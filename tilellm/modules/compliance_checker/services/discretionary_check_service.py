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
    _rerank_chunks,
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
    L01CheckResult,
    TenderLotRequirements,
)
from tilellm.modules.compliance_checker.prompts import (
    DISCRETIONARY_JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from tilellm.modules.compliance_checker.prompts.xlsx_extraction import _LLMLotExtractionResult
from tilellm.modules.compliance_checker.services.yaml_requirements_loader import YamlRequirementsLoader
from tilellm.modules.compliance_checker.services.l01_service import run_l01_check
from tilellm.shared.cache.semantic_cache import SemanticCache
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.shared import token_tracking
from tilellm.shared.token_tracking import TokenUsageCollector, model_name_of

logger = logging.getLogger(__name__)

_E_PROCUREMENT_DOMAIN = "e_procurement"

# Judge invocation: transient provider failures (timeout, overload, empty/malformed
# response) are common enough in practice — see docs/COMPLIANCE_V2_IMPLEMENTATION_PLAN.md
# §7.9 — to warrant a short bounded retry before giving up on a single criterion.
_MAX_JUDGE_ATTEMPTS = 2
_JUDGE_RETRY_DELAY_S = 1.0


class JudgeInvocationError(Exception):
    """Raised when the judge LLM call fails (provider error, or unparseable response)
    after all retries. Distinct from a genuine "no evidence" judgment — the caller
    must never silently treat this the same as an empty-but-valid verdict."""


def _apply_l01_quantity(
    lot: TenderLotRequirements,
    results: List[DiscretionaryResult],
    l01_check: L01CheckResult,
) -> None:
    """Use the L01 product count as the comparable quantity for `quantity_from_l01` criteria.

    Ampiezza di gamma: the price list lives in the operator's L01, which is deliberately
    NOT indexed — the judge sees no chunk carrying it and returns `measured_quantity=None`,
    leaving the criterion unscored for every operator. The count is the measurement.
    A quantity the judge did manage to extract wins: this is a fallback, not an override.
    """
    if not l01_check or not l01_check.used:
        return
    flagged = {c.id for c in lot.requirements.discretionary if c.quantity_from_l01}
    if not flagged:
        return
    for r in results:
        if r.criterion_id in flagged and r.measured_quantity is None:
            r.measured_quantity = float(l01_check.l01_products_total)
            logger.info(
                "Criterio '%s': measured_quantity=%s dal conteggio prodotti L01.",
                r.criterion_id, r.measured_quantity,
            )


def _build_capitolato_evidence_block(chunks: List[str], metadata: List[dict]) -> str:
    """Format capitolato chunks with a [CAP-N] prefix — deliberately distinct from the
    offer's [N] blocks (_build_evidence_block, logic.py) so the judge never confuses a
    capitolato citation with an offer citation in source_chunk_index/evidence_text."""
    lines = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
        file_name = meta.get("file_name", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        lines.append(f"[CAP-{i}] {file_name} | page {page}")
        lines.append(chunk[:1500])
        lines.append("")
    return "\n".join(lines)


class DiscretionaryCheckService:
    def __init__(self, repo, llm, request: ComplianceRequestV2):
        self._repo = repo
        self._llm = llm
        self._request = request
        # Collects token usage from every LLM call in this check (tabular + discretionary).
        self.tokens = TokenUsageCollector()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def evaluate_lot(self, lot: TenderLotRequirements) -> ComplianceReportV2:
        tabular_results = await self._check_tabular(lot)
        disc_results = await self._check_discretionary(lot)
        l01_check = await self._check_l01()
        _apply_l01_quantity(lot, disc_results, l01_check)
        summary = ComplianceSummaryV2.from_results(tabular_results, disc_results)
        return ComplianceReportV2(
            tender=lot.tender,
            namespace=self._request.namespace,
            summary=summary,
            tabular_results=tabular_results,
            discretionary_results=disc_results,
            l01_check=l01_check,
        )

    # ------------------------------------------------------------------
    # L01 → PDF consistency check (opt-in via request.l01_xlsx_url)
    # ------------------------------------------------------------------

    async def _check_l01(self) -> L01CheckResult:
        """Explicit L01 flag: `used=False` unless an l01_xlsx_url was provided."""
        if not (self._request.l01_xlsx_url and self._request.l01_xlsx_url.strip()):
            return L01CheckResult(used=False)

        async def _retrieve(query: str):
            qa = QuestionAnswer(
                question=query,
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
            retrieval = await self._repo.get_chunks_from_repo(qa)
            return (retrieval.chunks or [], retrieval.metadata or [])

        return await run_l01_check(
            self._request.l01_xlsx_url,
            _retrieve,
            concurrency=self._request.max_concurrent_requirements,
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
        # Pass our collector so tabular judge tokens are merged into this check's total;
        # the v1 path records into it and leaves emission/attachment to us.
        tabular_report = await check_compliance(v1_request, token_collector=self.tokens)
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

        # return_exceptions=True: _evaluate_criterion already turns judge failures
        # into a flagged DiscretionaryResult (see JudgeInvocationError handling
        # above), but this is the safety net — one unexpected bug on one criterion
        # must never abort every other already-evaluated criterion for this operator.
        raw_results = await asyncio.gather(
            *[_process(c) for c in criteria], return_exceptions=True
        )
        results: List[DiscretionaryResult] = []
        for criterion, r in zip(criteria, raw_results):
            if isinstance(r, BaseException):
                logger.error(
                    "Criterio '%s': eccezione non gestita durante la valutazione — %s",
                    criterion.id, r,
                )
                results.append(DiscretionaryResult(
                    criterion_id=criterion.id,
                    criterion_text=criterion.text,
                    mode=criterion.mode,
                    max_points=criterion.max_points,
                    direction=criterion.direction,
                    human_review_required=True,
                    human_review_reason=f"Errore imprevisto durante la valutazione: {r}",
                    motivation="Valutazione non completata per un errore interno "
                               "(non un giudizio di merito sul criterio).",
                    confidence=0.0,
                ))
            else:
                results.append(r)
        return results

    async def _fetch_capitolato_evidence(self, criterion: DiscretionaryCriterion) -> Optional[str]:
        """Retrieve evidence for *criterion* from the shared capitolato_namespace, if any.

        The same criterion text is queried once per operator in a bulk check (the
        capitolato is shared across operators of the same lot) — cached via the
        existing Redis-backed SemanticCache (L1 exact match only: criterion text is
        byte-identical across operators, no need for the L2 embedding cost). Any
        failure (bad namespace, Redis down, retrieval error) degrades to "no
        capitolato evidence for this criterion" rather than breaking the check.
        """
        ns = self._request.capitolato_namespace
        if not ns:
            return None
        try:
            cached = await SemanticCache.lookup(ns, criterion.text, embedding=None, check_l2=False)
            if cached is not None:
                chunks, metadata = cached.get("chunks", []), cached.get("metadata", [])
            else:
                qa = QuestionAnswer(
                    question=criterion.text,
                    namespace=ns,
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
                retrieval = await self._repo.get_chunks_from_repo(qa)
                chunks, metadata = retrieval.chunks or [], retrieval.metadata or []
                await SemanticCache.store(
                    ns, criterion.text, embedding=None,
                    body={"chunks": chunks, "metadata": metadata}, store_l2=False,
                )
            if not chunks:
                return None
            return _build_capitolato_evidence_block(chunks, metadata)
        except Exception as e:
            logger.warning(
                "Capitolato retrieval failed for criterion '%s' (namespace=%r): %s — "
                "proceeding without capitolato evidence.", criterion.id, ns, e,
            )
            return None

    async def _evaluate_criterion(self, criterion: DiscretionaryCriterion) -> DiscretionaryResult:
        # human_only → no LLM call, flag immediately
        if criterion.human_only:
            return DiscretionaryResult(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                mode=criterion.mode,
                max_points=criterion.max_points,
                direction=criterion.direction,
                human_review_required=True,
                human_review_reason="Criterio marcato human_only: richiede valutazione soggettiva non automatizzabile.",
                motivation="Valutazione delegata alla commissione.",
                confidence=0.0,
            )

        # Retrieve evidence from vector store.
        # When reranking is enabled, oversample (top_k × multiplier) then rerank
        # down to top_k — mirrors the v1 tabular path (logic.py).
        reranker_config = self._request.reranker_config
        search_top_k = (
            self._request.top_k * self._request.reranking_multiplier
            if reranker_config
            else self._request.top_k
        )
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
            top_k=search_top_k,
            search_type=self._request.search_type,
        )
        if self._request.exclude_chiarimenti:
            # Clarification-response documents can only point back to existing offer
            # pages, never add new content — so they must never be judged as evidence
            # (see docs/COMPLIANCE_V2_IMPLEMENTATION_PLAN.md §7, pattern E). No-op on
            # chunks that were never tagged with doc_type (backward compatible).
            qa._metadata_filter = {"doc_type": {"$ne": "chiarimento"}}
        try:
            retrieval = await self._repo.get_chunks_from_repo(qa)
            chunks = retrieval.chunks or []
            metadata = retrieval.metadata or []
        except Exception as e:
            logger.warning("Retrieval failed for criterion '%s': %s", criterion.id, e)
            chunks = []
            metadata = []

        if reranker_config and chunks:
            try:
                chunks, metadata = await _rerank_chunks(
                    criterion.text, chunks, metadata, reranker_config, self._request.top_k
                )
            except Exception as e:
                logger.warning(
                    "Reranking failed for criterion '%s': %s — proceeding without reranking",
                    criterion.id, e,
                )

        # No evidence → flag without calling LLM
        if not chunks:
            return DiscretionaryResult(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                mode=criterion.mode,
                max_points=criterion.max_points,
                direction=criterion.direction,
                human_review_required=True,
                human_review_reason="Nessuna evidenza trovata nel namespace: impossibile valutare automaticamente.",
                motivation="Nessuna evidenza disponibile nel knowledge base.",
                confidence=0.0,
            )

        evidence_block = _build_evidence_block(chunks, metadata)
        capitolato_evidence_block = await self._fetch_capitolato_evidence(criterion)
        user_prompt = build_judge_user_prompt(
            criterion_id=criterion.id,
            criterion_text=criterion.text,
            mode=criterion.mode.value,
            max_points=criterion.max_points,
            evidence_block=evidence_block,
            capitolato_evidence_block=capitolato_evidence_block,
        )
        try:
            raw_output = await self._invoke_judge(user_prompt)
        except JudgeInvocationError as e:
            # Never let a provider failure masquerade as "nessuna evidenza trovata"
            # (see docs/COMPLIANCE_V2_IMPLEMENTATION_PLAN.md §7.9) — distinct reason,
            # always flagged for human review regardless of confidence/mode.
            logger.error("Criterio '%s': giudice LLM non disponibile — %s", criterion.id, e)
            return DiscretionaryResult(
                criterion_id=criterion.id,
                criterion_text=criterion.text,
                mode=criterion.mode,
                max_points=criterion.max_points,
                direction=criterion.direction,
                human_review_required=True,
                human_review_reason=f"Errore del provider LLM durante la valutazione: {e}",
                motivation="Valutazione non completata per un errore del provider LLM "
                           "(non un giudizio di merito sul criterio).",
                confidence=0.0,
            )

        coefficient = raw_output.get("coefficient")
        if coefficient is not None:
            coefficient = max(0.0, min(1.0, float(coefficient)))
        measured_value = raw_output.get("measured_value")
        measured_quantity = raw_output.get("measured_quantity")
        if measured_quantity is not None:
            try:
                measured_quantity = float(measured_quantity)
            except (TypeError, ValueError):
                measured_quantity = None
        motivation = str(raw_output.get("motivation", ""))
        confidence = max(0.0, min(1.0, float(raw_output.get("confidence", 0.0))))
        source_index = int(raw_output.get("source_chunk_index") or 0)
        evidence_text = str(raw_output.get("evidence_text", ""))
        capitolato_discrepancy = raw_output.get("capitolato_discrepancy") or None

        evidence_doc, evidence_page, evidence_section, matched_idx = _pick_best_source(
            chunks, metadata, evidence_text, source_index
        )
        # Audit: we got here only with non-empty chunks, so matched_idx == 0 means the
        # judge produced a verdict but did not anchor it to any chunk (empty evidence_text
        # / source_index out of range). The score is still usable, but the documental
        # citation is an unverified fallback → flag it for the human operator.
        citation_attributed = matched_idx > 0
        if not citation_attributed:
            logger.info(
                "Citation not attributable for criterion '%s' (model returned no usable "
                "evidence_text/source_chunk_index) — flagged for human audit.",
                criterion.id,
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

        # Coherence guard: the judge reported a positive result (score/measured value)
        # but did NOT anchor it to any real chunk — self-contradictory (see docs/
        # COMPLIANCE_V2_IMPLEMENTATION_PLAN.md §7, pattern F / Santucci r.59: "dice che
        # non trova niente" in one field, then reports a value in another). Legitimate
        # negative/absent results (coefficient=0, no evidence) are NOT flagged here —
        # that is the correct, expected shape for "requirement genuinely absent".
        claims_positive_result = (
            criterion.mode in (DiscretionaryMode.VARIABILE, DiscretionaryMode.ON_OFF)
            and coefficient is not None and coefficient > 0
        ) or (
            criterion.mode == DiscretionaryMode.PROPORZIONALE
            and (measured_value or measured_quantity is not None)
        )
        if not citation_attributed and claims_positive_result:
            human_review_required = True
            coherence_note = (
                "Incoerenza rilevata: risultato positivo riportato (punteggio/valore "
                "misurato) senza una citazione ancorabile a un chunk recuperato — "
                "verificare manualmente."
            )
            human_review_reason = (
                f"{human_review_reason} {coherence_note}" if human_review_reason else coherence_note
            )

        return DiscretionaryResult(
            criterion_id=criterion.id,
            criterion_text=criterion.text,
            mode=criterion.mode,
            max_points=criterion.max_points,
            direction=criterion.direction,
            coefficient=coefficient,
            score=score,
            measured_value=measured_value,
            measured_quantity=measured_quantity,
            motivation=motivation,
            confidence=confidence,
            capitolato_discrepancy=capitolato_discrepancy,
            human_review_required=human_review_required,
            human_review_reason=human_review_reason,
            citation_attributed=citation_attributed,
            evidence_document=evidence_doc,
            evidence_page=evidence_page,
            evidence_section=evidence_section,
            evidence_text=evidence_text,
            evidence_chunk_index=matched_idx,
        )

    async def _invoke_judge_once(self, user_prompt: str) -> dict:
        """Single attempt: call the judge LLM and parse its JSON response.

        Raises on any failure — provider error (ainvoke) or empty/unparseable
        content. Never swallows: the caller (_invoke_judge) decides whether to
        retry or give up, and giving up must never look like "no evidence".
        """
        response = await self._llm.ainvoke([
            SystemMessage(content=DISCRETIONARY_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        self.tokens.record(
            response, operation="discretionary_judge", model=model_name_of(self._request.model)
        )
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
        if not raw:
            # Empirically what an overloaded provider looks like (e.g. deepseek
            # dropping a queued request): HTTP 200, empty content — no exception
            # to catch, just nothing to parse.
            raise JudgeInvocationError("Risposta vuota dal modello giudice.")
        return json.loads(raw)  # json.JSONDecodeError propagates, caught by the retry loop

    async def _invoke_judge(self, user_prompt: str) -> dict:
        """Call the judge LLM, retrying a bounded number of times on transient
        failures (provider error, timeout, empty/malformed response) before
        giving up. Raises JudgeInvocationError — never returns a fake empty
        verdict, which the caller could otherwise mistake for a genuine
        "no evidence found" judgment.
        """
        last_error: Exception = JudgeInvocationError("nessun tentativo eseguito")
        for attempt in range(1, _MAX_JUDGE_ATTEMPTS + 1):
            try:
                return await self._invoke_judge_once(user_prompt)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Judge LLM invocation failed (tentativo %d/%d): %s",
                    attempt, _MAX_JUDGE_ATTEMPTS, e,
                )
                if attempt < _MAX_JUDGE_ATTEMPTS:
                    await asyncio.sleep(_JUDGE_RETRY_DELAY_S)
        raise JudgeInvocationError(
            f"Il giudice LLM non ha risposto correttamente dopo {_MAX_JUDGE_ATTEMPTS} "
            f"tentativi: {last_error}"
        ) from last_error


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
        xlsx_url=request.requirements_xlsx_url,
        lot_id=request.requirements_lot_id,
    )
    svc = DiscretionaryCheckService(repo=repo, llm=llm, request=request)
    report = await svc.evaluate_lot(lot)

    # Always attempt analytics (fire-and-forget); attach the token detail only on debug.
    token_tracking.emit_analytics(
        svc.tokens,
        id_project=request.id_project,
        source="compliance",
        provider=request.llm,
        request_id=request.request_id,
    )
    if request.debug:
        report.token_usage = svc.tokens.to_dict()

    return report
