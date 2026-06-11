"""
DiscretionaryCheckService — evaluates a single lot (tabular + discretionary).

Tabular path  → delegates to v1 check_compliance (uses its own DI decorators).
Discretionary → uses pre-injected repo + llm directly; applies human_review rules.

Entry point: check_compliance_v2 (decorated with @inject_llm_chat_async @inject_repo_async).
"""
import asyncio
import json
import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from tilellm.models import QuestionAnswer
from tilellm.modules.compliance_checker.logic import (
    _build_evidence_block,
    _pick_best_source,
    check_compliance,
)
from tilellm.modules.compliance_checker.models import (
    ComplianceConfig,
    ComplianceRequest,
    RequirementItem,
)
from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceReportV2,
    ComplianceSummaryV2,
    ComplianceRequestV2,
    DiscretionaryCriterion,
    DiscretionaryMode,
    DiscretionaryResult,
    TenderLotRequirements,
)
from tilellm.modules.compliance_checker.prompts import (
    DISCRETIONARY_JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from tilellm.modules.compliance_checker.prompts.xlsx_extraction import _LLMLotExtractionResult
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import YamlRequirementsLoader
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async

logger = logging.getLogger(__name__)

_E_PROCUREMENT_DOMAIN = "e_procurement"


class DiscretionaryCheckService:
    def __init__(self, repo, llm, request: ComplianceRequestV2):
        self._repo = repo
        self._llm = llm
        self._request = request

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def evaluate_lot(self, lot: TenderLotRequirements) -> ComplianceReportV2:
        tabular_results = await self._check_tabular(lot)
        disc_results = await self._check_discretionary(lot)
        summary = ComplianceSummaryV2.from_results(tabular_results, disc_results)
        return ComplianceReportV2(
            tender=lot.tender,
            namespace=self._request.namespace,
            summary=summary,
            tabular_results=tabular_results,
            discretionary_results=disc_results,
        )

    # ------------------------------------------------------------------
    # Tabular: delegate to v1 check_compliance
    # ------------------------------------------------------------------

    async def _check_tabular(self, lot: TenderLotRequirements):
        if not lot.requirements.tabular:
            return []

        from tilellm.modules.compliance_checker.prompts import get_builtin_config
        config = get_builtin_config(_E_PROCUREMENT_DOMAIN)

        reqs = [
            RequirementItem(id=r.id, text=r.text, mandatory=r.mandatory)
            for r in lot.requirements.tabular
        ]
        v1_request = ComplianceRequest(
            config=config,
            requirements=reqs,
            namespace=self._request.namespace,
            engine=self._request.engine,
            embedding=self._request.embedding,
            sparse_encoder=self._request.sparse_encoder,
            top_k=self._request.top_k,
            llm=self._request.llm,
            gptkey=self._request.gptkey,
            model=self._request.model,
            temperature=self._request.temperature,
            max_tokens=self._request.max_tokens,
            search_type=self._request.search_type,
            reranking=self._request.reranking,
            reranking_multiplier=self._request.reranking_multiplier,
            reranker_model=self._request.reranker_model,
            max_concurrent_requirements=self._request.max_concurrent_requirements,
        )
        tabular_report = await check_compliance(v1_request)
        return tabular_report.results

    # ------------------------------------------------------------------
    # Discretionary: per-criterion retrieval + LLM judge
    # ------------------------------------------------------------------

    async def _check_discretionary(self, lot: TenderLotRequirements) -> List[DiscretionaryResult]:
        criteria = lot.requirements.discretionary
        if not criteria:
            return []

        semaphore = asyncio.Semaphore(self._request.max_concurrent_requirements)

        async def _process(criterion: DiscretionaryCriterion) -> DiscretionaryResult:
            async with semaphore:
                return await self._evaluate_criterion(criterion)

        return list(await asyncio.gather(*[_process(c) for c in criteria]))

    async def _evaluate_criterion(self, criterion: DiscretionaryCriterion) -> DiscretionaryResult:
        # human_only → no LLM call, flag immediately
        if criterion.human_only:
            return DiscretionaryResult(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                mode=criterion.mode,
                max_points=criterion.max_points,
                human_review_required=True,
                human_review_reason="Criterio marcato human_only: richiede valutazione soggettiva non automatizzabile.",
                motivation="Valutazione delegata alla commissione.",
                confidence=0.0,
            )

        # Retrieve evidence from vector store
        qa = QuestionAnswer(
            question=criterion.text,
            namespace=self._request.namespace,
            engine=self._request.engine,
            embedding=self._request.embedding,
            sparse_encoder=self._request.sparse_encoder,
            gptkey=self._request.gptkey,
            model=self._request.model,
            temperature=self._request.temperature,
            max_tokens=self._request.max_tokens,
            top_k=self._request.top_k,
            search_type=self._request.search_type,
        )
        try:
            retrieval = await self._repo.get_chunks_from_repo(qa)
            chunks = retrieval.chunks or []
            metadata = retrieval.metadata or []
        except Exception as e:
            logger.warning("Retrieval failed for criterion '%s': %s", criterion.id, e)
            chunks = []
            metadata = []

        # No evidence → flag without calling LLM
        if not chunks:
            return DiscretionaryResult(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                mode=criterion.mode,
                max_points=criterion.max_points,
                human_review_required=True,
                human_review_reason="Nessuna evidenza trovata nel namespace: impossibile valutare automaticamente.",
                motivation="Nessuna evidenza disponibile nel knowledge base.",
                confidence=0.0,
            )

        evidence_block = _build_evidence_block(chunks, metadata)
        user_prompt = build_judge_user_prompt(
            criterion_id=criterion.id,
            criterion_text=criterion.text,
            mode=criterion.mode.value,
            max_points=criterion.max_points,
            evidence_block=evidence_block,
        )
        raw_output = await self._invoke_judge(user_prompt)

        coefficient = raw_output.get("coefficient")
        if coefficient is not None:
            coefficient = max(0.0, min(1.0, float(coefficient)))
        measured_value = raw_output.get("measured_value")
        motivation = str(raw_output.get("motivation", ""))
        confidence = max(0.0, min(1.0, float(raw_output.get("confidence", 0.0))))
        source_index = int(raw_output.get("source_chunk_index", 0))
        evidence_text = str(raw_output.get("evidence_text", ""))

        evidence_doc, evidence_page, evidence_section, matched_idx = _pick_best_source(
            chunks, metadata, evidence_text, source_index
        )

        # Compute score based on mode
        score: Optional[float] = None
        human_review_required = False
        human_review_reason: Optional[str] = None

        if criterion.mode == DiscretionaryMode.VARIABILE:
            if coefficient is not None:
                score = round(coefficient * criterion.max_points, 2)
        elif criterion.mode == DiscretionaryMode.ON_OFF:
            if coefficient is not None:
                score = round(coefficient * criterion.max_points, 2)
        elif criterion.mode == DiscretionaryMode.PROPORZIONALE:
            # Never assign a score: needs cross-operator comparison
            score = None
            human_review_required = True
            human_review_reason = (
                "Modalità proporzionale: il punteggio finale richiede "
                "il confronto tra tutti gli operatori economici partecipanti."
            )

        # Low confidence → flag for human review
        if not human_review_required and confidence < self._request.min_confidence:
            human_review_required = True
            human_review_reason = (
                f"Confidenza ({confidence:.2f}) sotto la soglia minima "
                f"({self._request.min_confidence:.2f}): revisione umana raccomandata."
            )

        return DiscretionaryResult(
            criterion_id=criterion.id,
            criterion_text=criterion.text,
            mode=criterion.mode,
            max_points=criterion.max_points,
            coefficient=coefficient,
            score=score,
            measured_value=measured_value,
            motivation=motivation,
            confidence=confidence,
            human_review_required=human_review_required,
            human_review_reason=human_review_reason,
            evidence_document=evidence_doc,
            evidence_page=evidence_page,
            evidence_section=evidence_section,
            evidence_text=evidence_text,
            evidence_chunk_index=matched_idx,
        )

    async def _invoke_judge(self, user_prompt: str) -> dict:
        """Call the judge LLM and return a parsed dict."""
        response = await self._llm.ainvoke([
            SystemMessage(content=DISCRETIONARY_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part:
                        text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            raw = "\n".join(text_parts).strip()
        else:
            raw = str(content).strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Judge LLM returned non-JSON for criterion: %s", e)
            return {}


# ---------------------------------------------------------------------------
# Public entry point — decorated for DI
# ---------------------------------------------------------------------------

@inject_llm_chat_async
@inject_repo_async
async def check_compliance_v2(
    request: ComplianceRequestV2,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> ComplianceReportV2:
    """
    Full v2 compliance check: tabular requirements + discretionary scoring.

    Both LLM and repo are injected by the shared DI decorators.
    """
    loader = YamlRequirementsLoader()
    lot = await loader.load(
        yaml_inline=request.requirements_yaml,
        yaml_url=request.requirements_yaml_url,
    )
    svc = DiscretionaryCheckService(repo=repo, llm=llm, request=request)
    return await svc.evaluate_lot(lot)
