"""
ComplianceChecker — FastAPI controller.

Endpoints:
  POST /api/compliance/check           — run a full compliance check (JSON response)
  POST /api/compliance/check/rtm       — run a check and return a filled RTM CSV file
  POST /api/compliance/ask             — natural-language compliance check (LLM extracts requirements)
  GET  /api/compliance/domains         — list built-in domain names
  GET  /api/compliance/config/{domain} — return the built-in ComplianceConfig for a domain
"""
import logging
from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, SecretStr

from tilellm.models.embedding import LlmEmbeddingModel
from tilellm.models.llm import TEIConfig
from tilellm.models.vector_store import Engine
from tilellm.modules.compliance_checker.models import (
    ComplianceConfig,
    ComplianceReport,
    ComplianceRequest,
)
from tilellm.modules.compliance_checker.logic import check_compliance
from tilellm.modules.compliance_checker.prompts import get_builtin_config, list_builtin_domains
from tilellm.modules.compliance_checker.prompts.e_procurement import E_PROCUREMENT_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/compliance",
    tags=["Compliance Checker"],
)


@router.post("/check", response_model=ComplianceReport)
async def run_compliance_check(request: ComplianceRequest):
    """
    Run a RAG-based compliance check against a knowledge base. Returns JSON.

    Supply requirements either as a structured list (`requirements`) or as raw
    RTM CSV text (`csv_requirements`).  The two fields are mutually exclusive.

    The caller provides:
    - **requirements** / **csv_requirements**: requirements to verify.
    - **config**: domain config with the system prompt that defines judgment semantics.
      Use one of the built-in domains (`/api/compliance/domains`) or supply a custom prompt.
    - **namespace / engine**: vector store where the document-under-review is indexed.
    - **LLM fields** (`llm`, `gptkey`, `model`, …): judge LLM to use for evaluation.
    """
    try:
        report = await check_compliance(request)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"ComplianceChecker error: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


@router.post("/check/rtm")
async def run_compliance_check_rtm(request: ComplianceRequest):
    """
    Run a compliance check and return the result as a **filled RTM CSV file**.

    The response is a downloadable CSV (UTF-8 with BOM for Excel compatibility)
    in the standard Italian public-procurement RTM format:

    | Requisito | Presenza del requisito (SI/NO) | Nome documento | Pagina | Note |
    |-----------|-------------------------------|----------------|--------|------|

    - `csv_requirements` is the most convenient input: paste the raw RTM table directly.
    - The `judgment_map` in `config` controls the SI/NO/PARZIALE/N/V labels
      (defaults to Italian RTM format).
    """
    try:
        report = await check_compliance(request)
        jmap = request.config.judgment_map  # may be None → to_rtm_csv uses default
        csv_content = report.to_rtm_csv(judgment_map=jmap)
        filename = f"rtm_{report.domain}_{report.namespace}.csv"
        return Response(
            content=csv_content.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"ComplianceChecker RTM error: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


class AskComplianceRequest(BaseModel):
    """
    Request for the natural-language compliance endpoint.

    The caller provides a free-text question that contains (or describes) the
    requirements to verify.  The LLM extracts the requirement list and the
    domain automatically; all other fields mirror ComplianceRequest.
    """
    question: str = Field(..., description="Free-text question containing the requirements to verify.")
    namespace: str
    engine: Engine
    embedding: Union[str, LlmEmbeddingModel] = Field(default="text-embedding-3-small")
    sparse_encoder: Union[str, TEIConfig, None] = Field(default="splade")
    gptkey: Optional[SecretStr] = Field(default=None)
    model: Union[str, LlmEmbeddingModel] = Field(default="gpt-4o-mini")
    llm: Optional[str] = Field(default="openai")
    temperature: float = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    max_tokens: int = Field(default=512)
    debug: bool = Field(default=False)
    id_project: Optional[str] = Field(default=None, description="Tiledesk project ID per analytics token.")
    request_id: Optional[str] = Field(default=None, description="ID richiesta Tiledesk per analytics.")
    top_k: int = Field(default=8)
    domain_hint: Optional[str] = Field(
        default=None,
        description="Force a specific domain instead of auto-detecting it. "
                    "Use one of the values returned by GET /api/compliance/domains.",
    )


@router.post("/ask", response_model=ComplianceReport)
async def ask_compliance(request: AskComplianceRequest):
    """
    Natural-language compliance check.

    The caller provides a free-text question (e.g. pasted directly from a chat).
    The LLM automatically:
    1. Detects whether the message contains compliance requirements.
    2. Extracts the requirement list (from inline CSV, bullet points, etc.).
    3. Selects the most appropriate domain (e_procurement, medical_devices, …).

    If the message does not appear to be a compliance request, a 422 is returned
    with a suggestion to use `/api/v2/qa` instead.
    """
    from tilellm.models import QuestionAnswer
    from tilellm.agents.nodes import _get_guard_llm, IntentExtractionResult
    from tilellm.shared import token_tracking
    from tilellm.shared.token_tracking import TokenUsageCollector, model_name_of

    # Single collector for the whole /ask flow (intent extraction + judge calls).
    tokens = TokenUsageCollector()
    model_name = model_name_of(request.model)

    # Build a minimal QuestionAnswer so _get_guard_llm can resolve the LLM
    qa = QuestionAnswer(
        question=request.question,
        namespace=request.namespace,
        engine=request.engine,
        embedding=request.embedding,
        sparse_encoder=request.sparse_encoder,
        gptkey=request.gptkey,
        model=request.model,
        llm=request.llm,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        debug=request.debug,
        top_k=request.top_k,
    )

    # Step 1 — intent extraction
    try:
        llm = await _get_guard_llm(qa)
        # include_raw keeps the AIMessage so intent-extraction tokens are captured.
        structured_llm = llm.with_structured_output(IntentExtractionResult, include_raw=True)
        prompt = (
            f"Analizza questa richiesta e determina l'intento:\n\n"
            f"Richiesta: {request.question}\n\n"
            f"Rispondi con un JSON strutturato secondo lo schema richiesto."
        )
        raw_intent = await structured_llm.ainvoke(prompt)
        tokens.record(raw_intent, operation="compliance_intent", model=model_name)
        # include_raw=True yields {"raw","parsed","parsing_error"}; fall back to the
        # parsed object directly for providers/mocks that don't honor include_raw.
        if isinstance(raw_intent, dict):
            if raw_intent.get("parsing_error"):
                raise ValueError(raw_intent["parsing_error"])
            intent_result = raw_intent["parsed"]
        else:
            intent_result = raw_intent
    except Exception as e:
        logger.warning(f"/ask intent extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intent extraction failed: {e}")

    if intent_result.intent != "compliance":
        raise HTTPException(
            status_code=422,
            detail=(
                "La domanda non sembra contenere requisiti di compliance da verificare. "
                "Per domande generiche usa /api/v2/qa."
            ),
        )

    if not intent_result.csv_requirements:
        raise HTTPException(
            status_code=422,
            detail=(
                "Non è stato possibile estrarre requisiti dalla domanda. "
                "Prova a strutturarli in forma di elenco o tabella CSV."
            ),
        )

    # Step 2 — resolve domain and config
    domain = request.domain_hint or intent_result.domain
    config = get_builtin_config(domain) or E_PROCUREMENT_CONFIG

    # Step 3 — run compliance check
    try:
        compliance_request = ComplianceRequest(
            config=config,
            csv_requirements=intent_result.csv_requirements,
            namespace=request.namespace,
            engine=request.engine,
            embedding=request.embedding,
            sparse_encoder=request.sparse_encoder,
            gptkey=request.gptkey,
            model=request.model,
            llm=request.llm,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            debug=request.debug,
            id_project=request.id_project,
            request_id=request.request_id,
            top_k=request.top_k,
        )
        # Pass the shared collector so judge tokens merge with intent tokens;
        # the controller owns analytics emission + the debug token_usage block.
        report = await check_compliance(compliance_request, token_collector=tokens)
        token_tracking.emit_analytics(
            tokens,
            id_project=request.id_project,
            source="compliance",
            provider=request.llm,
            request_id=request.request_id,
        )
        if request.debug:
            report.token_usage = tokens.to_dict()
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"/ask compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {e}")


@router.get("/domains")
async def list_domains():
    """
    List the built-in domain names available for ComplianceChecker.

    Pass one of these as `config.domain` and retrieve the corresponding
    pre-built system prompt via `GET /api/compliance/config/{domain}`.
    """
    return {"domains": list_builtin_domains()}


@router.get("/config/{domain}", response_model=ComplianceConfig)
async def get_domain_config(domain: str):
    """
    Return the built-in ComplianceConfig for a given domain.

    Useful to inspect the default system_prompt and judgment_labels before
    submitting a check request, or to copy it as a starting point for customisation.
    """
    config = get_builtin_config(domain)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No built-in config for domain '{domain}'. "
                   f"Available: {list_builtin_domains()}",
        )
    return config


# v2 endpoints — included here so the feature-router scanner picks up a single `router`
from tilellm.modules.compliance_checker.controllers_v2 import router_v2  # noqa: E402
router.include_router(router_v2)
