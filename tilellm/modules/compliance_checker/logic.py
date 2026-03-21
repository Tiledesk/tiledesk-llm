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
from typing import List, Tuple

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge-LLM user-turn prompt
# ---------------------------------------------------------------------------

_JUDGE_USER_PROMPT = """\
<requirement>
ID: {req_id}
Text: {req_text}
</requirement>

<retrieved_evidence>
{evidence_block}
</retrieved_evidence>

Based ONLY on the retrieved evidence above, evaluate whether the requirement is satisfied.
Respond with a single valid JSON object (no markdown fences) with exactly these keys:
  "judgment"      : one of {valid_judgments}
  "confidence"    : float between 0.0 and 1.0
  "evidence_text" : direct quote from the evidence that best supports your judgment (empty string if none)
  "justification" : 1-3 sentences explaining your judgment

If there is no relevant evidence, set judgment to "not_verifiable" and confidence to 0.0."""


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
        lines.append(chunk[:1500])
        lines.append("")
    return "\n".join(lines)


def _pick_best_source(
    chunks: List[str], metadata: List[dict], evidence_text: str
) -> Tuple[str, int, str]:
    """
    Return (file_name, page, heading_path) of the chunk whose text best matches
    the LLM-quoted *evidence_text*.  Falls back to the first chunk.
    """
    if not metadata:
        return ("", 1, "")

    if evidence_text:
        snippet = evidence_text[:80]
        for chunk, meta in zip(chunks, metadata):
            if snippet in chunk:
                return (
                    meta.get("file_name", meta.get("source", "")),
                    int(meta.get("page", 1)),
                    meta.get("heading_path", ""),
                )

    meta = metadata[0]
    return (
        meta.get("file_name", meta.get("source", "")),
        int(meta.get("page", 1)),
        meta.get("heading_path", ""),
    )


async def _judge_requirement(
    req: RequirementItem,
    chunks: List[str],
    metadata: List[dict],
    config: ComplianceConfig,
    llm,
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
        raw = response.content.strip()
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
    evidence_text = str(parsed.get("evidence_text", ""))
    justification = str(parsed.get("justification", "LLM response could not be parsed."))

    evidence_doc, evidence_page, evidence_section = _pick_best_source(
        chunks, metadata, evidence_text
    )

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
        evidence_chunk_ids=chunk_ids,
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
    semaphore = asyncio.Semaphore(request.max_concurrent_requirements)

    async def _process_one(req: RequirementItem) -> ComplianceResult:
        async with semaphore:
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
                top_k=request.top_k,
                search_type=request.search_type,
            )
            try:
                retrieval = await repo.get_chunks_from_repo(qa)
                chunks = retrieval.chunks
                metadata = retrieval.metadata
            except Exception as e:
                logger.warning(f"Retrieval failed for requirement '{req.id}': {e}")
                chunks = []
                metadata = []

            return await _judge_requirement(req, chunks, metadata, request.config, llm)

    results = list(await asyncio.gather(*[_process_one(r) for r in request.requirements]))
    summary = _compute_summary(results)

    return ComplianceReport(
        domain=request.config.domain,
        namespace=request.namespace,
        summary=summary,
        results=results,
    )
