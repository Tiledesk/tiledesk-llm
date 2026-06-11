"""
ComplianceChecker v2 — data models.

Estende il modulo v1 con:
- Criteri discrezionali (modalità variabile / proporzionale / on_off)
- Scoring: coefficiente 0–1 × max_points (L. 36/2023)
- human_review_required con reason esplicita
- YAML come formato di input per i requisiti di gara
- Estrazione xlsx → YAML (LLM-assisted)

I modelli v1 (models.py, logic.py) NON vengono modificati.
"""
import io
from enum import StrEnum
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from tilellm.models import Engine, LlmEmbeddingModel
from tilellm.models.llm import PineconeRerankerConfig, TEIConfig
from tilellm.modules.compliance_checker.models import (
    ComplianceResult,
    _DEFAULT_JUDGMENT_MAP,
)


# ---------------------------------------------------------------------------
# 1.1 — DiscretionaryMode
# ---------------------------------------------------------------------------

class DiscretionaryMode(StrEnum):
    VARIABILE = "variabile"
    PROPORZIONALE = "proporzionale"
    ON_OFF = "on_off"


# ---------------------------------------------------------------------------
# 1.2 — TabularRequirementV2
# ---------------------------------------------------------------------------

class TabularRequirementV2(BaseModel):
    """Requisito tabellare (presenza/assenza) — allineato a RequirementItem v1."""
    id: str
    text: str
    mandatory: bool = True


# ---------------------------------------------------------------------------
# 1.3 — DiscretionaryCriterion
# ---------------------------------------------------------------------------

class DiscretionaryCriterion(BaseModel):
    """Criterio discrezionale con punteggio massimo e modalità di attribuzione."""
    id: str
    text: str
    mode: DiscretionaryMode
    max_points: float
    human_only: bool = Field(
        default=False,
        description="Se True il criterio non viene mai valutato dall'IA (es. criteri soggettivi).",
    )
    notes: Optional[str] = None

    @field_validator("max_points")
    @classmethod
    def _max_points_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_points deve essere > 0, ricevuto {v}")
        return v


# ---------------------------------------------------------------------------
# 1.4 — TenderInfo
# ---------------------------------------------------------------------------

class TenderInfo(BaseModel):
    """Metadati del lotto/gara — radice del documento YAML."""
    title: str
    lot_id: str
    lot_name: str
    source_file: Optional[str] = None


# ---------------------------------------------------------------------------
# 1.5 — TenderLotRequirements (radice del documento YAML)
# ---------------------------------------------------------------------------

class _RequirementsBlock(BaseModel):
    tabular: List[TabularRequirementV2] = Field(default_factory=list)
    discretionary: List[DiscretionaryCriterion] = Field(default_factory=list)


class TenderLotRequirements(BaseModel):
    """
    Documento YAML che rappresenta i requisiti di un singolo lotto di gara.

    Usato sia come output dell'endpoint xlsx→YAML (dopo revisione umana)
    sia come input del check v2.
    """
    tender: TenderInfo
    requirements: _RequirementsBlock = Field(default_factory=_RequirementsBlock)

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> "TenderLotRequirements":
        tab_ids = [r.id for r in self.requirements.tabular]
        disc_ids = [c.id for c in self.requirements.discretionary]
        for id_list, label in [(tab_ids, "tabular"), (disc_ids, "discretionary")]:
            seen = set()
            for rid in id_list:
                if rid in seen:
                    raise ValueError(
                        f"ID duplicato nei requisiti {label}: '{rid}'. "
                        "Ogni requisito deve avere un id univoco."
                    )
                seen.add(rid)
        return self

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "TenderLotRequirements":
        """
        Costruisce l'istanza da una stringa YAML.

        Raises:
            ValueError: YAML malformato o schema non valido.
        """
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML malformato: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("YAML non valido: il documento deve essere un mapping.")
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(f"Schema YAML non valido: {e}") from e

    def to_yaml(self) -> str:
        """Serializza in stringa YAML UTF-8, pronta per revisione umana."""
        # mode="json" converte StrEnum → str, evitando tag !!python/object nel YAML
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# 1.6 — ExtractRequirementsRequest
# ---------------------------------------------------------------------------

class ExtractRequirementsRequest(BaseModel):
    """Richiesta di estrazione YAML da file xlsx."""
    source: str = Field(..., description="URL del file xlsx contenente i criteri di gara.")
    llm: Optional[str] = Field(default="openai")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)
    debug: bool = Field(default=False)


# ---------------------------------------------------------------------------
# 1.7 — ExtractRequirementsResponse / LotYamlDocument
# ---------------------------------------------------------------------------

class LotYamlDocument(BaseModel):
    """YAML dei requisiti per un singolo lotto, pronto per revisione umana."""
    lot_id: str
    lot_name: str
    yaml: str = Field(..., description="Contenuto YAML del documento TenderLotRequirements.")
    warnings: List[str] = Field(
        default_factory=list,
        description="Avvisi emessi durante l'estrazione (es. criteri marcati human_only).",
    )


class ExtractRequirementsResponse(BaseModel):
    """Risposta dell'endpoint /requirements/extract — un documento YAML per lotto."""
    lots: List[LotYamlDocument]


# ---------------------------------------------------------------------------
# 1.8 — ComplianceRequestV2
# ---------------------------------------------------------------------------

class ComplianceRequestV2(BaseModel):
    """
    Richiesta di check v2: tabellari + discrezionali.

    I requisiti vengono forniti come YAML (inline oppure via URL, mutuamente esclusivi).
    """
    # Requisiti: esattamente uno dei due deve essere fornito
    requirements_yaml: Optional[str] = Field(
        default=None,
        description="Contenuto YAML del documento TenderLotRequirements passato inline nel body.",
    )
    requirements_yaml_url: Optional[str] = Field(
        default=None,
        description="URL di un documento YAML TenderLotRequirements da scaricare.",
    )

    # Soglia confidence sotto la quale si attiva human_review_required
    min_confidence: float = Field(
        default=0.6,
        description="Soglia minima di confidenza: sotto questa soglia il risultato viene flaggato per revisione umana.",
    )

    # Vector store / retrieval
    namespace: str
    engine: Engine
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    sparse_encoder: Union[str, TEIConfig, None] = Field(default="splade")
    top_k: int = Field(default=8)
    search_type: str = Field(default="similarity")

    # LLM (judge)
    llm: Optional[str] = Field(default="openai")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)
    debug: bool = Field(default=False)

    # Reranking
    reranking: Union[bool, TEIConfig, PineconeRerankerConfig] = Field(default=False)
    reranking_multiplier: int = Field(default=3)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Concorrenza
    max_concurrent_requirements: int = Field(default=3)

    @model_validator(mode="after")
    def _validate_requirements_source(self) -> "ComplianceRequestV2":
        has_inline = bool(self.requirements_yaml and self.requirements_yaml.strip())
        has_url = bool(self.requirements_yaml_url and self.requirements_yaml_url.strip())
        if has_inline and has_url:
            raise ValueError(
                "'requirements_yaml' e 'requirements_yaml_url' sono mutuamente esclusivi: "
                "fornire uno solo dei due."
            )
        if not has_inline and not has_url:
            raise ValueError(
                "Fornire 'requirements_yaml' (YAML inline) oppure "
                "'requirements_yaml_url' (URL al documento YAML)."
            )
        return self


# ---------------------------------------------------------------------------
# 1.9 — DiscretionaryResult
# ---------------------------------------------------------------------------

class DiscretionaryResult(BaseModel):
    """Risultato della valutazione di un singolo criterio discrezionale."""
    criterion_id: str
    criterion_text: str
    mode: DiscretionaryMode
    max_points: float

    # Scoring (None per proporzionale e human_only)
    coefficient: Optional[float] = None
    score: Optional[float] = None

    # Solo per mode=proporzionale
    measured_value: Optional[str] = Field(
        default=None,
        description="Valore misurabile estratto dall'offerta (es. '14 misure 4.0–10.5 mm').",
    )

    motivation: str
    confidence: float

    human_review_required: bool = False
    human_review_reason: Optional[str] = Field(
        default=None,
        description="Motivo esplicito per cui è richiesta la revisione umana.",
    )

    # Audit trail (allineato a ComplianceResult v1)
    evidence_document: str = ""
    evidence_page: int = 1
    evidence_section: str = ""
    evidence_text: str = ""
    evidence_chunk_index: int = 0
    evidence_chunk_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1.10 — ComplianceSummaryV2
# ---------------------------------------------------------------------------

class TabularSummary(BaseModel):
    """Contatori per i requisiti tabellari (semantica v1)."""
    total: int = 0
    compliant: int = 0
    non_compliant: int = 0
    partial: int = 0
    not_verifiable: int = 0
    compliance_rate: float = 0.0

    @model_validator(mode="after")
    def _compute_rate(self) -> "TabularSummary":
        verifiable = self.total - self.not_verifiable
        if verifiable > 0 and self.compliance_rate == 0.0 and self.compliant > 0:
            self.compliance_rate = round(self.compliant / verifiable, 4)
        return self


class ComplianceSummaryV2(BaseModel):
    """Riepilogo completo: requisiti tabellari + criteri discrezionali."""
    tabular: TabularSummary = Field(default_factory=TabularSummary)

    discretionary_total: int = 0
    ai_scored_count: int = Field(
        default=0,
        description="Criteri discrezionali effettivamente valutati dall'IA (con score non None).",
    )
    ai_scored_points: float = Field(
        default=0.0,
        description="Somma dei punteggi assegnati dall'IA.",
    )
    ai_max_scorable_points: float = Field(
        default=0.0,
        description="Punteggio massimo ottenibile dai criteri valutati dall'IA.",
    )
    human_review_count: int = Field(
        default=0,
        description="Criteri che richiedono revisione umana.",
    )
    human_review_points: float = Field(
        default=0.0,
        description="Punteggio massimo in attesa di revisione umana.",
    )

    @classmethod
    def from_results(
        cls,
        tabular_results: List[ComplianceResult],
        disc_results: List[DiscretionaryResult],
    ) -> "ComplianceSummaryV2":
        # Tabular counters
        counts: Dict[str, int] = {
            "compliant": 0, "non_compliant": 0, "partial": 0, "not_verifiable": 0,
        }
        for r in tabular_results:
            key = r.judgment if r.judgment in counts else "not_verifiable"
            counts[key] += 1
        total_tab = len(tabular_results)
        verifiable = total_tab - counts["not_verifiable"]
        rate = round(counts["compliant"] / verifiable, 4) if verifiable > 0 else 0.0
        tab_summary = TabularSummary(
            total=total_tab,
            compliance_rate=rate,
            **counts,
        )

        # Discretionary counters
        ai_count = 0
        ai_pts = 0.0
        ai_max = 0.0
        hr_count = 0
        hr_pts = 0.0
        for d in disc_results:
            if d.human_review_required:
                hr_count += 1
                hr_pts += d.max_points
            elif d.score is not None:
                ai_count += 1
                ai_pts += d.score
                ai_max += d.max_points

        return cls(
            tabular=tab_summary,
            discretionary_total=len(disc_results),
            ai_scored_count=ai_count,
            ai_scored_points=round(ai_pts, 2),
            ai_max_scorable_points=round(ai_max, 2),
            human_review_count=hr_count,
            human_review_points=round(hr_pts, 2),
        )


# ---------------------------------------------------------------------------
# 1.11 — ComplianceReportV2
# ---------------------------------------------------------------------------

_JUDGMENT_LABELS: Dict[str, str] = _DEFAULT_JUDGMENT_MAP


class ComplianceReportV2(BaseModel):
    """Report completo di check v2 (tabellari + discrezionali)."""
    tender: TenderInfo
    namespace: str
    summary: ComplianceSummaryV2
    tabular_results: List[ComplianceResult]
    discretionary_results: List[DiscretionaryResult]

    def to_markdown(self, judgment_map: Optional[Dict[str, str]] = None) -> str:
        """
        Genera la tabella di restituzione in formato Markdown.

        Formato rif.: docs/gare/Tabella tipo restituzione da IA.docx
        """
        jmap = judgment_map or _JUDGMENT_LABELS
        buf = io.StringIO()

        buf.write(f"# Valutazione offerta tecnica — {self.tender.lot_name}\n\n")
        buf.write(f"**Namespace operatore:** `{self.namespace}`\n\n")

        # Sezione 1: Requisiti tabellari
        if self.tabular_results:
            buf.write("## Requisiti minimi (conformità)\n\n")
            buf.write("| Requisito | Presenza | Documento | Pagina | Note |\n")
            buf.write("|-----------|----------|-----------|--------|------|\n")
            for r in self.tabular_results:
                presence = jmap.get(r.judgment, r.judgment)
                doc = r.evidence_document.replace("|", "\\|")
                page = str(r.evidence_page) if r.evidence_document else ""
                note = r.justification.replace("|", "\\|")
                req = r.requirement_text.replace("|", "\\|")
                buf.write(f"| {req} | {presence} | {doc} | {page} | {note} |\n")
            buf.write("\n")

        # Sezione 2: Criteri discrezionali
        if self.discretionary_results:
            buf.write("## Criteri discrezionali (punteggio qualitativo)\n\n")
            buf.write(
                "| ID | Criterio | Punteggio | Max | Documento | Pagina | Motivazione | Revisione umana |\n"
            )
            buf.write(
                "|----|----------|-----------|-----|-----------|--------|-------------|----------------|\n"
            )
            for d in self.discretionary_results:
                if d.human_review_required:
                    score_str = f"⚠ REVISIONE UMANA"
                elif d.score is not None:
                    score_str = f"{d.score:.2f}"
                elif d.measured_value:
                    score_str = f"Valore: {d.measured_value}"
                else:
                    score_str = "N/V"
                doc = d.evidence_document.replace("|", "\\|")
                page = str(d.evidence_page) if d.evidence_document else ""
                mot = d.motivation.replace("|", "\\|")
                text = d.criterion_text.replace("|", "\\|")
                hr_reason = (d.human_review_reason or "").replace("|", "\\|")
                buf.write(
                    f"| {d.criterion_id} | {text} | {score_str} | {d.max_points} "
                    f"| {doc} | {page} | {mot} | {hr_reason} |\n"
                )
            buf.write("\n")

        # Riepilogo
        s = self.summary
        buf.write("## Riepilogo\n\n")
        if s.tabular.total > 0:
            buf.write(
                f"**Tabellari:** {s.tabular.compliant}/{s.tabular.total} conformi "
                f"(compliance rate {s.tabular.compliance_rate:.0%})\n\n"
            )
        if s.discretionary_total > 0:
            buf.write(
                f"**Discrezionali:** punteggio IA {s.ai_scored_points:.2f}/{s.ai_max_scorable_points:.2f} "
                f"| in attesa revisione umana: {s.human_review_count} criteri "
                f"({s.human_review_points:.2f} pt max)\n"
            )

        return buf.getvalue()
