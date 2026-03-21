"""
ComplianceChecker — data models.

Domain-agnostic: the only domain-specific input is ComplianceConfig.system_prompt.
Everything else (retrieval, judgment, RTM assembly) is invariant.
"""
import csv
import io
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, SecretStr, model_validator

from tilellm.models import Engine, LlmEmbeddingModel
from tilellm.models.llm import TEIConfig


# ---------------------------------------------------------------------------
# Default judgment → output-string mapping (Italian RTM format)
# ---------------------------------------------------------------------------

_DEFAULT_JUDGMENT_MAP: Dict[str, str] = {
    "compliant": "SI",
    "non_compliant": "NO",
    "partial": "PARZIALE",
    "not_verifiable": "N/V",
}

# Standard RTM CSV column headers (Italian public-procurement format)
_RTM_HEADERS = [
    "Requisito",
    "Presenza del requisito (SI/NO)",
    "Nome del documento allegato in Busta tecnica in cui è rinvenibile il requisito",
    "n. Pagina del documento allegato in Busta tecnica in cui è rinvenibile il requisito",
    "Note e/o descrizione aggiuntiva (eventuale)",
]


# ---------------------------------------------------------------------------
# CSV parsing helper
# ---------------------------------------------------------------------------

def _parse_rtm_csv(csv_text: str) -> List["RequirementItem"]:
    """
    Parse a raw RTM CSV string into a list of RequirementItem.

    Accepts two formats:
      1. Header row present  — first row contains "requisit" (case-insensitive) → skipped
      2. No header row       — every row is treated as a requirement

    Only the first column (requirement text) is read; other columns are ignored.
    Empty rows are skipped.  IDs are auto-generated as REQ-001, REQ-002, …
    """
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = list(reader)
    if not rows:
        return []

    # Detect and skip header row
    start = 0
    if rows[0] and "requisit" in rows[0][0].lower():
        start = 1

    items: List[RequirementItem] = []
    seq = 1
    for row in rows[start:]:
        if not row:
            continue
        text = row[0].strip()
        if not text:
            continue
        items.append(RequirementItem(
            id=f"REQ-{seq:03d}",
            text=text,
            mandatory=True,
        ))
        seq += 1
    return items


# ---------------------------------------------------------------------------
# Domain configuration (the ONLY thing that changes per use-case)
# ---------------------------------------------------------------------------

class ComplianceConfig(BaseModel):
    """
    Domain-specific prompt configuration.

    Swap this to change use-case:
      - e_procurement     → tender compliance
      - hr_assessment     → CV / competency evaluation
      - legal_audit       → regulatory / policy compliance
      - medical_devices   → MDR/MDD device procurement (gara ospedaliera)
      - custom            → any user-defined domain
    """
    domain: str = Field(..., description="Label for the domain (e.g. 'e_procurement').")
    system_prompt: str = Field(
        ...,
        description=(
            "System prompt injected into the judge LLM. "
            "Must instruct the LLM to return JSON with keys: "
            "judgment, confidence, evidence_text, justification."
        )
    )
    judgment_labels: List[str] = Field(
        default=["compliant", "non_compliant", "partial", "not_verifiable"],
        description="Valid values for the 'judgment' field in the LLM response."
    )
    judgment_map: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Mapping from judgment labels to output strings used in the RTM CSV. "
            "Defaults to Italian RTM format: "
            "{compliant: SI, non_compliant: NO, partial: PARZIALE, not_verifiable: N/V}."
        )
    )


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class RequirementItem(BaseModel):
    """A single requirement from the user-provided table."""
    id: str = Field(..., description="Unique requirement identifier (e.g. 'REQ-001', '3.2.1').")
    text: str = Field(..., description="Full text of the requirement.")
    category: Optional[str] = Field(None, description="Category/section the requirement belongs to.")
    mandatory: bool = Field(True, description="Whether the requirement is mandatory.")


class ComplianceRequest(BaseModel):
    """Full request for a compliance check run."""
    config: ComplianceConfig

    # Requirements: supply EITHER a structured list OR raw CSV text (not both).
    requirements: Optional[List[RequirementItem]] = Field(
        default=None,
        description=(
            "User-provided requirements as a structured list. "
            "Mutually exclusive with csv_requirements."
        )
    )
    csv_requirements: Optional[str] = Field(
        default=None,
        description=(
            "Requirements as raw RTM CSV text (Italian public-procurement format). "
            "First column is the requirement text; header row is auto-detected and skipped. "
            "IDs are auto-generated as REQ-001, REQ-002, …  "
            "Mutually exclusive with requirements."
        )
    )

    # Vector store / retrieval
    namespace: str
    engine: Engine
    embedding: Union[str,LlmEmbeddingModel] = Field(default_factory=lambda: "text-embedding-3-small")
    sparse_encoder: Union[str, "TEIConfig", None] = Field(default="splade") #bge-m3
    top_k: int = Field(default=8, description="Chunks retrieved per requirement.")

    # LLM (for judgment) — fields mirror QuestionAnswer so @inject_llm_chat_async works directly
    llm: Optional[str] = Field(default="openai")
    gptkey: Optional[SecretStr] = "sk"
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)
    debug: bool = Field(default=False)
    search_type: str = Field(default_factory=lambda: "similarity")

    # Behaviour
    max_concurrent_requirements: int = Field(
        default=3,
        description="Max requirements processed concurrently (rate-limit guard)."
    )

    @model_validator(mode="after")
    def _resolve_requirements(self) -> "ComplianceRequest":
        has_list = bool(self.requirements)
        has_csv = bool(self.csv_requirements and self.csv_requirements.strip())

        if has_list and has_csv:
            raise ValueError(
                "Provide either 'requirements' or 'csv_requirements', not both."
            )
        if has_csv:
            parsed = _parse_rtm_csv(self.csv_requirements)
            if not parsed:
                raise ValueError(
                    "csv_requirements was provided but no valid requirement rows were found."
                )
            self.requirements = parsed
        if not self.requirements:
            raise ValueError(
                "Either 'requirements' or 'csv_requirements' must be provided."
            )
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class ComplianceResult(BaseModel):
    """Judgment for a single requirement."""
    requirement_id: str
    requirement_text: str
    category: Optional[str]
    mandatory: bool

    judgment: str                       # one of config.judgment_labels
    confidence: float                   # 0.0 – 1.0
    evidence_text: str                  # direct quote from the document
    justification: str                  # LLM explanation

    evidence_document: str              # file_name of the source chunk
    evidence_page: int                  # page number (1-indexed)
    evidence_section: str               # heading_path of the source chunk
    evidence_chunk_ids: List[str] = []  # doc_ids of retrieved chunks (debug)


class ComplianceSummary(BaseModel):
    total: int
    compliant: int = 0
    non_compliant: int = 0
    partial: int = 0
    not_verifiable: int = 0
    compliance_rate: float = 0.0        # compliant / (total - not_verifiable)


class ComplianceReport(BaseModel):
    """Full output of a compliance check run."""
    domain: str
    namespace: str
    summary: ComplianceSummary
    results: List[ComplianceResult]

    def to_rtm_csv(self, judgment_map: Optional[Dict[str, str]] = None) -> str:
        """
        Render a filled RTM CSV table in Italian public-procurement format.

        Args:
            judgment_map: Override the judgment → output-string mapping.
                Defaults to {compliant: SI, non_compliant: NO,
                             partial: PARZIALE, not_verifiable: N/V}.

        Returns:
            UTF-8 CSV string with BOM (compatible with Microsoft Excel).
        """
        jmap = judgment_map or _DEFAULT_JUDGMENT_MAP
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(_RTM_HEADERS)
        for r in self.results:
            writer.writerow([
                r.requirement_text,
                jmap.get(r.judgment, r.judgment),
                r.evidence_document,
                str(r.evidence_page) if r.evidence_document else "",
                r.justification,
            ])
        # Prepend BOM for Excel UTF-8 compatibility
        return "\ufeff" + buf.getvalue()

    def to_markdown(self, judgment_map=None):
        """
        Genera una rappresentazione in formato Markdown dei risultati.

        Args:
            judgment_map: Dizionario opzionale per mappare i giudizi in formati personalizzati

        Returns:
            Stringa in formato Markdown con i risultati
        """
        jmap = judgment_map or _DEFAULT_JUDGMENT_MAP
        buf = io.StringIO()

        # Intestazione della tabella Markdown
        buf.write("| Requisito | Giudizio | Documento | Pagina | Giustificazione |\n")
        buf.write("|-----------|----------|-----------|--------|-----------------|\n")

        # Righe della tabella
        for r in self.results:
            # Gestione pagina vuota
            page = str(r.evidence_page) if r.evidence_document else ""

            # Escape dei caratteri speciali Markdown nelle celle
            requirement = r.requirement_text.replace('|', '\\|')
            judgment = jmap.get(r.judgment, r.judgment).replace('|', '\\|')
            document = r.evidence_document.replace('|', '\\|') if r.evidence_document else ""
            justification = r.justification.replace('|', '\\|')

            # Scrittura riga
            buf.write(f"| {requirement} | {judgment} | {document} | {page} | {justification} |\n")

        return buf.getvalue()
