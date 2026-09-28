"""
ComplianceChecker — core async logic.

For each requirement in ComplianceRequest:
  1. Hybrid-search the vector store for relevant chunks.
  2. Pass requirement + chunks to a judge LLM with a domain-specific system prompt.
  3. Parse the structured JSON response into ComplianceResult.
  4. Aggregate into ComplianceReport with summary statistics.

The algorithm is fully domain-agnostic: the only domain-specific input is
ComplianceConfig.system_prompt, set by the caller or picked from built-in prompts.

LLM and repository are injected by the shared decorators @inject_llm_chat_async and
@inject_repo_async, which handle caching, provider routing, and all supported backends.
"""
import asyncio
import json
import logging
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from tilellm.models import QuestionAnswer
from tilellm.modules.compliance_checker.models import (
    ComplianceConfig,
    ComplianceReport,
    ComplianceRequest,
    ComplianceResult,
    ComplianceSummary,
    RequirementItem,
)
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.shared import token_tracking
from tilellm.shared.token_tracking import TokenUsageCollector, model_name_of

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge-LLM user-turn prompt
# ---------------------------------------------------------------------------

_JUDGE_USER_PROMPT = """\
<requisito>
ID: {req_id}
Testo: {req_text}
</requisito>

<evidenze_recuperate>
{evidence_block}
</evidenze_recuperate>

Basandoti ESCLUSIVAMENTE sulle evidenze recuperate sopra, valuta se il requisito è soddisfatto.
Il requisito può comparire nelle evidenze con parole diverse, sinonimi, forme equivalenti o in \
un'altra lingua (le offerte sono spesso multilingue): valuta il significato, non la corrispondenza \
letterale. Una dichiarazione esplicita che il prodotto possiede la caratteristica richiesta, o che è \
conforme a una norma il cui titolo o contenuto, riportato nelle evidenze, riguarda quella \
caratteristica, è evidenza valida: non pretendere dati di test o certificati se il requisito non li \
chiede espressamente.
Rispondi con un singolo oggetto JSON valido (senza fence markdown) con esattamente queste chiavi:
  "judgment"           : uno tra {valid_judgments}
  "confidence"         : numero float tra 0.0 e 1.0
  "source_chunk_index" : intero — l'indice [N] del chunk (tra quelli sopra etichettati) il cui testo \
supporta meglio il tuo giudizio; usa 0 se nessun chunk è rilevante
  "evidence_text"      : citazione verbatim (copia-incolla) dal chunk scelto che supporta meglio \
il giudizio; stringa vuota se assente
  "justification"      : 1-3 frasi in italiano che spiegano il giudizio

Se non vi sono evidenze rilevanti, imposta judgment a "not_verifiable", confidence a 0.0, \
source_chunk_index a 0 ed evidence_text a ""."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_evidence_block(chunks: List[str], metadata: List[dict]) -> str:
    """Format retrieved chunks + source metadata into a numbered block for the LLM."""
    lines = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
        file_name = meta.get("file_name", meta.get("source", "unknown"))
        page = meta.get("page", "?")
        section = meta.get("heading_path", "")
        header = f"[{i}] {file_name} | page {page}"
        if section:
            header += f" | {section}"
        lines.append(header)
        # No truncation: chunks are already bounded upstream by table-aware chunking
        # (markdown_chunker.py), which deliberately keeps a table's rows together even
        # past the nominal chunk_size. A fixed-length cut here silently reintroduces
        # data loss for exactly the criteria that need a table's later rows — found on
        # a real tender (RC12/BIOCOMPOSITE, 2026-09-16): the cut landed mid-cell, right
        # before the numeric value the proportional criterion needed.
        lines.append(chunk)
        lines.append("")
    return "\n".join(lines)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation — for fuzzy comparison."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _meta_fields(meta: dict) -> Tuple[str, int, str]:
    return (
        meta.get("file_name", meta.get("source", "")),
        int(meta.get("page", 1)),
        meta.get("heading_path", ""),
    )


def _pick_best_source(
    chunks: List[str],
    metadata: List[dict],
    evidence_text: str,
    source_index: int = 0,
) -> Tuple[str, int, str, int]:
    """
    Return (file_name, page, heading_path, matched_chunk_1based) for the chunk
    that best matches the LLM-quoted evidence.

    Resolution order (most to least reliable):
    1. LLM-provided source_chunk_index  — direct, no string matching needed
    2. Exact substring match on evidence_text (tries 200 / 80 / 40 char snippets)
    3. Fuzzy difflib match (SequenceMatcher ratio >= 0.35)
    4. First chunk fallback (with a warning so the caller can inspect)
    """
    if not metadata:
        return ("", 1, "", 0)

    # 1. Primary: LLM explicitly told us which chunk it used
    if 1 <= source_index <= len(metadata):
        logger.debug("Citation resolved via source_chunk_index=%d", source_index)
        return (*_meta_fields(metadata[source_index - 1]), source_index)

    if evidence_text:
        # 2. Exact substring — progressively shorter snippets
        for snippet_len in (200, 80, 40):
            snippet = evidence_text[:snippet_len].strip()
            if not snippet:
                continue
            for i, (chunk, meta) in enumerate(zip(chunks, metadata), 1):
                if snippet in chunk:
                    logger.debug("Citation resolved via substring match (len=%d), chunk %d", snippet_len, i)
                    return (*_meta_fields(meta), i)

        # 3. Fuzzy match via difflib
        ev_norm = _normalize(evidence_text[:300])
        best_score, best_i = 0.0, -1
        for i, chunk in enumerate(chunks):
            score = SequenceMatcher(None, ev_norm, _normalize(chunk[:600])).ratio()
            if score > best_score:
                best_score, best_i = score, i
        if best_score >= 0.35 and best_i >= 0:
            logger.debug("Citation resolved via fuzzy match (score=%.2f), chunk %d", best_score, best_i + 1)
            return (*_meta_fields(metadata[best_i]), best_i + 1)

    # 4. Last resort — first chunk, with a warning
    logger.warning(
        "Could not attribute citation to any chunk — falling back to chunk 1. "
        "evidence_text[:80]=%r  source_index=%d",
        evidence_text[:80], source_index,
    )
    return (*_meta_fields(metadata[0]), 0)


async def _rerank_chunks(
    query: str,
    chunks: List[str],
    metadata: List[dict],
    reranker_config,
    top_k: int,
) -> Tuple[List[str], List[dict]]:
    """Re-order chunks by relevance to *query* using TileReranker, keep top_k."""
    from tilellm.tools.reranker import TileReranker  # deferred import

    docs = [Document(page_content=c, metadata=m) for c, m in zip(chunks, metadata)]
    reranker = TileReranker(reranker_config)
    loop = asyncio.get_event_loop()
    reranked: List[Document] = await loop.run_in_executor(
        None, lambda: reranker.rerank_documents(query, docs, top_k)
    )
    return [d.page_content for d in reranked], [d.metadata for d in reranked]


# Neighbours are re-attached only around the best-ranked chunks: that is where a split
# table row matters (the judge reads those first) and it caps the extra prompt size.
NEIGHBOR_EXPANSION_TOP_N = 5


def _chunk_position(meta: dict) -> Optional[Tuple[str, int]]:
    """(doc_id, chunk_index) of a chunk, or None when its position isn't recorded.
    chunk_index may come back as a float (Pinecone stores numbers as floats)."""
    doc_id = meta.get("doc_id") or meta.get("id")
    index = meta.get("chunk_index")
    if isinstance(index, float) and index.is_integer():
        index = int(index)
    if not doc_id or not isinstance(index, int) or isinstance(index, bool):
        return None
    return doc_id, index


async def _expand_with_neighbors(
    repo, engine, namespace: str, chunks: List[str], metadata: List[dict],
    top_n: int = NEIGHBOR_EXPANSION_TOP_N,
) -> Tuple[List[str], List[dict]]:
    """Merge each of the first `top_n` chunks with its neighbours (chunk_index +/- 1,
    same document), in document order. Document converters can split one table row
    into consecutive chunks — on a real tender the judge saw a requirement row
    without the manufacturer's answer, which sat in the next chunk.

    A neighbour already among the selected chunks, or already merged into an earlier
    one, is not repeated. Metadata is unchanged: the citation stays on the retrieved
    chunk. Best effort — any repository failure leaves the evidence as it was.
    """
    positions = [_chunk_position(m) for m in metadata]
    taken = {p for p in positions if p}
    owned: Dict[int, List[Tuple[str, int]]] = {}
    wanted_by_doc: Dict[str, set] = defaultdict(set)
    for i, position in enumerate(positions[:top_n]):
        if position is None:
            continue
        doc_id, index = position
        owned[i] = []
        for neighbour in ((doc_id, index - 1), (doc_id, index + 1)):
            if neighbour[1] >= 0 and neighbour not in taken:
                taken.add(neighbour)
                owned[i].append(neighbour)
                wanted_by_doc[doc_id].add(neighbour[1])
    if not wanted_by_doc:
        return chunks, metadata

    texts: Dict[Tuple[str, int], str] = {}
    try:
        for doc_id, indexes in wanted_by_doc.items():
            for doc in await repo.get_chunks_by_index(engine, namespace, doc_id, sorted(indexes)):
                position = _chunk_position(doc.metadata)
                if position:
                    texts[position] = doc.page_content
    except Exception as e:
        logger.warning("Neighbour expansion skipped, evidence left as retrieved: %s", e)
        return chunks, metadata

    expanded = list(chunks)
    for i, neighbours in owned.items():
        index = positions[i][1]
        before = [texts[n] for n in neighbours if n[1] < index and n in texts]
        after = [texts[n] for n in neighbours if n[1] > index and n in texts]
        expanded[i] = "\n\n".join(before + [chunks[i]] + after)
    return expanded, metadata


async def _retrieve_evidence(
    repo, qa: QuestionAnswer, rerank_query: str, reranker_config, top_k: int, label: str,
) -> Tuple[List[str], List[dict]]:
    """The one evidence pipeline every compliance judge uses: retrieve (qa.top_k is the
    oversampled pool) -> rerank down to `top_k` against `rerank_query` -> re-attach
    split neighbours. A retrieval failure yields no evidence; a reranking failure
    proceeds with the retrieved order. `label` only names the item in logs."""
    try:
        retrieval = await repo.get_chunks_from_repo(qa)
        chunks, metadata = retrieval.chunks or [], retrieval.metadata or []
    except Exception as e:
        logger.warning("Retrieval failed for %s: %s", label, e)
        return [], []

    if reranker_config and chunks:
        try:
            chunks, metadata = await _rerank_chunks(rerank_query, chunks, metadata, reranker_config, top_k)
        except Exception as e:
            logger.warning("Reranking failed for %s: %s — proceeding without reranking", label, e)

    return await _expand_with_neighbors(repo, qa.engine, qa.namespace, chunks, metadata)


async def _judge_requirement(
    req: RequirementItem,
    chunks: List[str],
    metadata: List[dict],
    config: ComplianceConfig,
    llm,
    token_collector: TokenUsageCollector = None,
    model_name: str = "",
) -> ComplianceResult:
    """Run the judge LLM for a single requirement and return a ComplianceResult."""
    chunk_ids = [str(m.get("id", m.get("doc_id", ""))) for m in metadata]

    if not chunks:
        return ComplianceResult(
            requirement_id=req.id,
            requirement_text=req.text,
            category=req.category,
            mandatory=req.mandatory,
            judgment="not_verifiable",
            confidence=0.0,
            evidence_text="",
            justification="No relevant evidence found in the knowledge base.",
            evidence_document="",
            evidence_page=1,
            evidence_section="",
            evidence_chunk_ids=chunk_ids,
        )

    evidence_block = _build_evidence_block(chunks, metadata)
    user_msg = _JUDGE_USER_PROMPT.format(
        req_id=req.id,
        req_text=req.text,
        evidence_block=evidence_block,
        valid_judgments=str(config.judgment_labels),
    )

    parsed: dict = {}
    try:
        response = await llm.ainvoke([
            SystemMessage(content=config.system_prompt),
            HumanMessage(content=user_msg),
        ])
        if token_collector is not None:
            token_collector.record(response, operation="compliance_judge", model=model_name)
        content = response.content
        # Reasoning models (gpt-5.x, o-series) return content as a list of blocks
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
        # Strip markdown code fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning(f"Judge LLM call failed for requirement '{req.id}': {e}")

    judgment = parsed.get("judgment", "not_verifiable")
    if judgment not in config.judgment_labels:
        judgment = "not_verifiable"

    confidence = float(parsed.get("confidence", 0.0))
    source_index = int(parsed.get("source_chunk_index") or 0)
    evidence_text = str(parsed.get("evidence_text", ""))
    justification = str(parsed.get("justification", "LLM response could not be parsed."))

    evidence_doc, evidence_page, evidence_section, matched_idx = _pick_best_source(
        chunks, metadata, evidence_text, source_index
    )
    citation_attributed = matched_idx > 0
    if not citation_attributed:
        # _pick_best_source's last resort is simply the first retrieved chunk — not
        # evidence. Reporting it as the source put unrelated documents next to
        # "not found" verdicts on a real tender.
        evidence_doc, evidence_page, evidence_section = "", 1, ""

    return ComplianceResult(
        requirement_id=req.id,
        requirement_text=req.text,
        category=req.category,
        mandatory=req.mandatory,
        judgment=judgment,
        confidence=confidence,
        evidence_text=evidence_text,
        justification=justification,
        evidence_document=evidence_doc,
        evidence_page=evidence_page,
        evidence_section=evidence_section,
        evidence_chunk_index=matched_idx,
        evidence_chunk_ids=chunk_ids,
        citation_attributed=citation_attributed,
    )


def _compute_summary(results: List[ComplianceResult]) -> ComplianceSummary:
    total = len(results)
    counts = {"compliant": 0, "non_compliant": 0, "partial": 0, "not_verifiable": 0}
    for r in results:
        key = r.judgment if r.judgment in counts else "not_verifiable"
        counts[key] += 1
    verifiable = total - counts["not_verifiable"]
    rate = counts["compliant"] / verifiable if verifiable > 0 else 0.0
    return ComplianceSummary(
        total=total,
        compliance_rate=round(rate, 4),
        **counts,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@inject_llm_chat_async
@inject_repo_async
async def check_compliance(
    request: ComplianceRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,    # injected by @inject_llm_chat_async, not used directly
    callback_handler=None,  # injected by @inject_llm_chat_async, not used directly
    embedding_config_key=None,
    token_collector: TokenUsageCollector = None,
    **kwargs,
) -> ComplianceReport:
    """
    Run a full compliance check for all requirements in *request*.

    Both the vector-store repository and the judge LLM are resolved by the shared
    infrastructure decorators:
    - ``@inject_repo_async`` — resolves the correct vector-store backend from
      ``request.engine`` (Pinecone serverless/pod, Qdrant, Milvus) with caching.
    - ``@inject_llm_chat_async`` — resolves the judge LLM from ``request.llm`` /
      ``request.model`` / ``request.gptkey`` with ``TimedCache`` (all providers
      supported by the platform: openai, anthropic, google, cohere, mistral,
      groq, deepseek, ollama, vllm, …).

    Requirements are evaluated concurrently up to ``request.max_concurrent_requirements``.
    """
    # When a caller (e.g. the v2 discretionary service) passes its own collector,
    # we only RECORD into it and let the caller own analytics emission + the
    # debug token_usage block. Standalone (v1 endpoint) we own both.
    owns_collector = token_collector is None
    collector = token_collector if token_collector is not None else TokenUsageCollector()
    model_name = model_name_of(request.model)

    semaphore = asyncio.Semaphore(request.max_concurrent_requirements)

    async def _process_one(req: RequirementItem) -> ComplianceResult:
        async with semaphore:
            reranker_config = request.reranker_config
            search_top_k = (
                request.top_k * request.reranking_multiplier
                if reranker_config
                else request.top_k
            )
            qa = QuestionAnswer(
                question=req.text,
                namespace=request.namespace,
                engine=request.engine,
                embedding=request.embedding,
                sparse_encoder=request.sparse_encoder,
                gptkey=request.gptkey,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_k=search_top_k,
                search_type=request.search_type,
            )
            chunks, metadata = await _retrieve_evidence(
                repo, qa, req.text, reranker_config, request.top_k, f"requirement '{req.id}'"
            )

            return await _judge_requirement(
                req, chunks, metadata, request.config, llm,
                token_collector=collector, model_name=model_name,
            )

    results = list(await asyncio.gather(*[_process_one(r) for r in request.requirements]))
    summary = _compute_summary(results)

    report = ComplianceReport(
        domain=request.config.domain,
        namespace=request.namespace,
        summary=summary,
        results=results,
    )

    if owns_collector:
        # Always attempt analytics (fire-and-forget); attach token detail only on debug.
        token_tracking.emit_analytics(
            collector,
            id_project=getattr(request, "id_project", None),
            source="compliance",
            provider=request.llm,
            request_id=getattr(request, "request_id", None),
        )
        if request.debug:
            report.token_usage = collector.to_dict()

    return report
