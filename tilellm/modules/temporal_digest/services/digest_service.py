"""
DigestService — core logic for generating and querying temporal digests.

Digest generation pipeline (per time window):
  1. Retrieve source chunks from vector store (dense filter by date).
  2. [Optional] Augment with lgraph PPR-seeded chunks (entity-linked, DATE_IT).
  3. [Optional] Augment with community summary context from {ns}__lgraph_communities.
  4. [Optional] Run LLM act_type classifier on chunks with missing type metadata.
  5. Build a structured evidence block and call the judge LLM.
  6. Index the summary vector with digest metadata for future retrieval.

Rollup strategy:
  - granularity='weekly':  synthesises from pre-existing daily digests when available;
    falls back to raw chunks when daily coverage is incomplete.
  - granularity='monthly': synthesises from pre-existing weekly digests when available;
    falls back to daily digests, then raw chunks.

Digest query:
  - 'temporal' mode: filter by type='digest' + date range → LLM synthesises answer.
  - 'semantic' mode: standard vector search on raw chunks (type != 'digest').
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from tilellm.models import QuestionAnswer
from tilellm.modules.temporal_digest.models.schemas import (
    DigestAgentRequest,
    DigestAgentResponse,
    DigestGenerationRequest,
    DigestGenerationResponse,
    DigestGenerationResult,
    DigestQueryRequest,
    DigestQueryResponse,
)
from tilellm.modules.temporal_digest.services.domain_prompts import get_domain_prompts
from tilellm.modules.temporal_digest.services.query_router import classify_query, classify_query_debug

logger = logging.getLogger(__name__)

_DIGEST_TYPE_FIELD = "digest_type"
_DIGEST_TYPE_VALUE = "digest"
_COMMUNITY_NS_SUFFIX = "__lgraph_communities"
_MAX_CHUNK_CHARS = 900
_MAX_EVIDENCE_CHARS = 140_000


def _format_chat_history(chat_history_dict: dict, max_messages: int = 10) -> str:
    if not chat_history_dict:
        return ""
    sorted_keys = sorted(chat_history_dict.keys(), key=lambda x: int(x))[-max_messages:]
    lines = []
    for k in sorted_keys:
        entry = chat_history_dict[k]
        if hasattr(entry, "get_question_text"):
            q_text = entry.get_question_text()
        elif isinstance(entry, dict):
            q_text = entry.get("question", "")
        else:
            q_text = str(getattr(entry, "question", entry))
        answer = getattr(entry, "answer", entry.get("answer", "")) if isinstance(entry, dict) else entry.answer
        lines.append(f"Utente: {q_text}")
        lines.append(f"Assistente: {answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_date_windows(
    date_from: date,
    date_to: date,
    granularity: str,
) -> List[Tuple[date, date]]:
    windows: List[Tuple[date, date]] = []
    current = date_from

    while current <= date_to:
        if granularity == "daily":
            windows.append((current, current))
            current += timedelta(days=1)
        elif granularity == "weekly":
            week_end = current + timedelta(days=6 - current.weekday())
            windows.append((current, min(week_end, date_to)))
            current = week_end + timedelta(days=1)
        elif granularity == "monthly":
            import calendar
            last_day = calendar.monthrange(current.year, current.month)[1]
            month_end = current.replace(day=last_day)
            windows.append((current, min(month_end, date_to)))
            current = (month_end + timedelta(days=1)).replace(day=1)
        else:
            windows.append((current, date_to))
            break

    return windows


def _build_evidence_block(chunks: List[str], metadatas: List[dict]) -> str:
    """Build a numbered evidence block for the LLM, respecting total char budget."""
    lines: List[str] = []
    total = 0
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        # Header: prefer structured metadata if available
        date_from = meta.get("digest_date_from")
        date_to = meta.get("digest_date_to")
        if date_from:
            header = f"[{i}] Periodo: {date_from}"
            if date_to and date_to != date_from:
                header += f"–{date_to}"
        else:
            source = meta.get("file_name", meta.get("source", "documento sconosciuto"))
            header = f"[{i}] {source} | pag. {meta.get('page', '?')}"

        act_type = meta.get("act_type")
        if act_type:
            header += f" | {act_type}"
        amount = meta.get("amount")
        if amount:
            try:
                header += f" | €{float(amount):,.2f}"
            except (ValueError, TypeError):
                pass

        snippet = chunk[:_MAX_CHUNK_CHARS]
        entry = f"{header}\n{snippet}\n"
        if total + len(entry) > _MAX_EVIDENCE_CHARS:
            remaining = len(chunks) - i + 1
            lines.append(f"[…] {remaining} atti aggiuntivi omessi per limiti di contesto.")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def _extract_llm_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return str(content).strip()


def _aggregate_metadata(metadatas: List[dict]) -> Tuple[Dict[str, int], Optional[float]]:
    act_counter: Counter = Counter()
    total_amount = 0.0
    has_amount = False

    for meta in metadatas:
        at = meta.get("act_type")
        if at:
            act_counter[at] += 1
        amt = meta.get("amount")
        if amt is not None:
            try:
                total_amount += float(amt)
                has_amount = True
            except (ValueError, TypeError):
                pass

    return dict(act_counter), total_amount if has_amount else None


# ---------------------------------------------------------------------------
# LLM-based act_type classifier
# ---------------------------------------------------------------------------

_ACT_TYPE_CLASSIFIER_SYSTEM = (
    "Sei un esperto in atti amministrativi di strutture sanitarie pubbliche italiane (ASL). "
    "Per ogni atto fornito, assegna la categoria dalla tassonomia indicata. "
    "Rispondi SOLO con un array JSON (senza markdown). "
    "Formato: [{\"idx\": 0, \"act_type\": \"LIQUIDAZIONE\"}, ...]\n\n"
    "Tassonomia:\n"
    "LIQUIDAZIONE — pagamenti a fornitori, liquidazione fatture\n"
    "DELIBERA_GARA — aggiudicazione appalti\n"
    "GARA_DESERTA — gare andate deserte\n"
    "BANDO — pubblicazione bandi\n"
    "PROCEDURA_NEGOZIATA — affidamenti con procedura negoziata\n"
    "AFFIDAMENTO_DIRETTO — forniture dirette sotto soglia\n"
    "COMPENSO — compensi a professionisti\n"
    "ASSUNZIONE — contratti di lavoro, nomine\n"
    "INCARICO_PROFESSIONALE — consulenze, collaborazioni esterne\n"
    "ACQUISTO_FARMACI — acquisto medicinali\n"
    "ACQUISTO_DISPOSITIVI_MEDICI — attrezzature, DPI, dispositivi\n"
    "RINNOVO_CONTRATTO — proroghe, rinnovi\n"
    "DELIBERA_PROGRAMMATICA — atti di indirizzo, piani\n"
    "VARIAZIONE_BILANCIO — variazioni budget, storni\n"
    "ALTRO — non classificabile"
)

_ACT_TYPE_BATCH = 20  # max chunks per classification call


async def _classify_act_types(
    chunks: List[str],
    metadatas: List[dict],
    llm,
) -> List[dict]:
    """Run LLM classifier on chunks missing act_type; returns updated metadatas list."""
    to_classify = [
        (i, chunks[i])
        for i, m in enumerate(metadatas)
        if not m.get("act_type")
    ]

    if not to_classify or llm is None:
        return metadatas

    updated = list(metadatas)
    for batch_start in range(0, len(to_classify), _ACT_TYPE_BATCH):
        batch = to_classify[batch_start: batch_start + _ACT_TYPE_BATCH]
        items_txt = "\n\n".join(
            f"[{local_idx}] {chunk[:400]}"
            for local_idx, (_, chunk) in enumerate(batch)
        )
        user_msg = f"Classifica i seguenti atti (indice 0-{len(batch)-1}):\n\n{items_txt}"
        try:
            resp = await llm.ainvoke([
                SystemMessage(content=_ACT_TYPE_CLASSIFIER_SYSTEM),
                HumanMessage(content=user_msg),
            ])
            raw = _extract_llm_text(resp).strip()
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`\n ")
            classifications = json.loads(raw)
            for item in classifications:
                local_idx = item.get("idx")
                act_type = item.get("act_type", "ALTRO")
                if local_idx is not None and 0 <= local_idx < len(batch):
                    original_idx = batch[local_idx][0]
                    updated[original_idx] = {**updated[original_idx], "act_type": act_type}
        except Exception as e:
            logger.warning(f"[classifier] batch failed: {e}")

    return updated


# ---------------------------------------------------------------------------
# DigestService
# ---------------------------------------------------------------------------

class DigestService:

    async def generate(
        self,
        request: DigestGenerationRequest,
        repo,
        llm,
        llm_embeddings,
    ) -> DigestGenerationResponse:
        """Generate digests for each time window in the request date range."""
        date_to = request.date_to or request.date_from
        windows = _iter_date_windows(request.date_from, date_to, request.granularity)

        results: List[DigestGenerationResult] = []
        total_chunks = 0
        skipped = 0

        for win_start, win_end in windows:
            result = await self._generate_window(
                request=request,
                win_start=win_start,
                win_end=win_end,
                repo=repo,
                llm=llm,
                llm_embeddings=llm_embeddings,
            )
            results.append(result)
            total_chunks += result.chunk_count
            if result.already_existed:
                skipped += 1

        return DigestGenerationResponse(
            namespace=request.namespace,
            digests=results,
            total_chunks_processed=total_chunks,
            total_windows=len(windows),
            skipped_windows=skipped,
        )

    # ------------------------------------------------------------------
    # lgraph augmentation (DATE_IT-linked chunks via PPR)
    # ------------------------------------------------------------------

    async def _fetch_chunks_from_lgraph(
        self,
        request: DigestGenerationRequest,
        date_from_str: str,
        date_to_str: str,
    ) -> Tuple[List[str], List[dict]]:
        """
        Query FalkorDB for chunks linked to DATE_IT entities in [date_from, date_to].
        Also retrieves chunks from the same Leiden communities as the matched entities
        to provide broader thematic context (LinearRAG-style).
        """
        try:
            from tilellm.modules.lgraph.logic import _get_falkor_repo
            from tilellm.modules.lgraph.services.graph_builder import make_graph_name as _mgn
            import re
            from datetime import date as _date

            graph_name = _mgn(request.namespace, request.engine.index_name)
            falkor = _get_falkor_repo()

            df = _date.fromisoformat(date_from_str)
            dt = _date.fromisoformat(date_to_str)

            date_re = re.compile(r'^(\d{1,2})[/\-\.](\d{2})[/\-\.](\d{4})$')

            def _parse_it(s: str):
                m = date_re.match(s.strip())
                if not m:
                    return None
                try:
                    return _date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                except ValueError:
                    return None

            # Fetch DATE_IT entities + connected chunks
            rows = await falkor._execute_query(
                "MATCH (e:LEntity {entity_type: 'DATE_IT'})-[:HAS_ENTITY]-(c:LChunk) "
                "WHERE e.namespace=$ns AND e.index_name=$idx "
                "RETURN e.name AS date_str, e.community_id AS community_id, "
                "c.chunk_id AS chunk_id, c.text AS text, "
                "c.source AS source, c.metadata_id AS metadata_id",
                {"ns": request.namespace, "idx": request.engine.index_name},
                graph_name=graph_name,
            )

            seen_ids: set = set()
            chunks_text: List[str] = []
            chunks_meta: List[dict] = []
            matched_communities: set = set()

            for row in rows:
                d = _parse_it(row.get("date_str", ""))
                if d is None or d < df or d > dt:
                    continue
                cid = row.get("chunk_id", "")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                chunks_text.append(row.get("text") or "")
                chunks_meta.append({
                    "chunk_id": cid,
                    "source": row.get("source", ""),
                    "metadata_id": row.get("metadata_id", ""),
                    "lgraph_date": row.get("date_str", ""),
                })
                comm_id = row.get("community_id")
                if comm_id is not None:
                    matched_communities.add(comm_id)

            # Extend with community-sibling chunks (LinearRAG-style broadening)
            if matched_communities:
                community_rows = await falkor._execute_query(
                    "MATCH (e:LEntity)-[:HAS_ENTITY]-(c:LChunk) "
                    "WHERE e.namespace=$ns AND e.index_name=$idx "
                    "AND e.community_id IN $comm_ids "
                    "RETURN c.chunk_id AS chunk_id, c.text AS text, "
                    "c.source AS source, c.metadata_id AS metadata_id",
                    {
                        "ns": request.namespace,
                        "idx": request.engine.index_name,
                        "comm_ids": list(matched_communities),
                    },
                    graph_name=graph_name,
                )
                for row in community_rows:
                    cid = row.get("chunk_id", "")
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    chunks_text.append(row.get("text") or "")
                    chunks_meta.append({
                        "chunk_id": cid,
                        "source": row.get("source", ""),
                        "metadata_id": row.get("metadata_id", ""),
                        "lgraph_community": True,
                    })

            logger.info(
                f"[lgraph augment] {date_from_str}–{date_to_str}: "
                f"{len(chunks_text)} chunks ({len(matched_communities)} communities)"
            )
            return chunks_text, chunks_meta

        except Exception as e:
            logger.warning(f"[lgraph augment] failed, skipping: {e}")
            return [], []

    # ------------------------------------------------------------------
    # Community summary context (vector store search on __lgraph_communities)
    # ------------------------------------------------------------------

    async def _fetch_community_context(
        self,
        request: DigestGenerationRequest,
        date_from_str: str,
        date_to_str: str,
        repo,
    ) -> str:
        """
        Retrieve community summaries relevant to this time window from the
        {namespace}__lgraph_communities vector store namespace.
        Returns a formatted string for injection into the evidence block, or ''.
        """
        try:
            community_ns = f"{request.namespace}{_COMMUNITY_NS_SUFFIX}"
            query = f"attività amministrativa {date_from_str} {date_to_str}"
            qa = self._build_qa(request, question=query)
            qa.namespace = community_ns
            qa.top_k = 5
            qa._metadata_filter = None  # no date filter on community summaries

            retrieval = await repo.get_chunks_from_repo(qa)
            if not retrieval.chunks:
                return ""

            lines = ["### Contesto tematico (comunità di entità rilevanti):"]
            for i, (chunk, meta) in enumerate(zip(retrieval.chunks, retrieval.metadata), 1):
                comm_label = meta.get("community_label", f"Comunità {meta.get('community_id', i)}")
                lines.append(f"- **{comm_label}**: {chunk[:600]}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[community context] not available: {e}")
            return ""

    # ------------------------------------------------------------------
    # Hierarchical rollup
    # ------------------------------------------------------------------

    async def _try_rollup(
        self,
        request: DigestGenerationRequest,
        win_start: date,
        win_end: date,
        repo,
        llm,
        granularity_source: str,
    ) -> Optional[Tuple[List[str], List[dict]]]:
        """
        Fetch pre-generated digests at finer granularity for rollup synthesis.
        Returns (chunks, metadatas) if sufficient coverage exists, else None.
        """
        child_windows = _iter_date_windows(win_start, win_end, granularity_source)
        child_ds = win_start.isoformat()
        child_de = win_end.isoformat()

        qa = self._build_qa(request, question=f"rollup {child_ds} {child_de}")
        qa.top_k = len(child_windows) * 2 + 5
        qa._metadata_filter = {
            _DIGEST_TYPE_FIELD: {"$eq": _DIGEST_TYPE_VALUE},
            "digest_granularity": {"$eq": granularity_source},
            "digest_date_from": {"$gte": child_ds},
            "digest_date_to": {"$lte": child_de},
        }

        try:
            retrieval = await repo.get_chunks_from_repo(qa)
            if not retrieval.chunks:
                return None
            # Accept rollup only when coverage is ≥ 50% of expected child windows
            if len(retrieval.chunks) < max(1, len(child_windows) // 2):
                return None
            return list(retrieval.chunks), list(retrieval.metadata)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core window generation
    # ------------------------------------------------------------------

    async def _generate_window(
        self,
        request: DigestGenerationRequest,
        win_start: date,
        win_end: date,
        repo,
        llm,
        llm_embeddings,
    ) -> DigestGenerationResult:
        ds = win_start.isoformat()
        de = win_end.isoformat()

        if not request.force_regenerate:
            existing = await self._fetch_digests(
                repo=repo, request=request,
                date_from_str=ds, date_to_str=de, top_k=1,
            )
            if existing.chunks:
                return DigestGenerationResult(
                    namespace=request.namespace,
                    date_from=ds, date_to=de,
                    granularity=request.granularity,
                    content=existing.chunks[0] if existing.chunks else "",
                    chunk_count=0,
                    already_existed=True,
                )

        chunks: List[str] = []
        metadatas: List[dict] = []
        used_rollup = False

        # --- Try hierarchical rollup first -----------------------------------
        if request.granularity == "weekly":
            rollup = await self._try_rollup(request, win_start, win_end, repo, llm, "daily")
            if rollup:
                chunks, metadatas = rollup
                used_rollup = True
        elif request.granularity == "monthly":
            rollup = await self._try_rollup(request, win_start, win_end, repo, llm, "weekly")
            if not rollup:
                rollup = await self._try_rollup(request, win_start, win_end, repo, llm, "daily")
            if rollup:
                chunks, metadatas = rollup
                used_rollup = True

        # --- Raw chunk retrieval when rollup not available -------------------
        if not chunks:
            retrieval = await self._fetch_source_chunks(
                repo=repo, request=request,
                date_from_str=ds, date_to_str=de,
            )
            chunks = list(retrieval.chunks)
            metadatas = list(retrieval.metadata)

            # Augment with lgraph DATE_IT + community chunks
            if getattr(request, "use_lgraph", False):
                lgraph_texts, lgraph_metas = await self._fetch_chunks_from_lgraph(
                    request=request, date_from_str=ds, date_to_str=de,
                )
                existing_ids = {m.get("chunk_id", m.get("metadata_id", "")) for m in metadatas}
                for text, meta in zip(lgraph_texts, lgraph_metas):
                    cid = meta.get("chunk_id", "")
                    if cid not in existing_ids:
                        chunks.append(text)
                        metadatas.append(meta)
                        existing_ids.add(cid)

        if not chunks:
            return DigestGenerationResult(
                namespace=request.namespace,
                date_from=ds, date_to=de,
                granularity=request.granularity,
                content="Nessun atto trovato per questo periodo.",
                chunk_count=0,
            )

        # --- LLM act_type classifier ----------------------------------------
        if not used_rollup and getattr(request, "classify_act_types", True):
            metadatas = await _classify_act_types(chunks, metadatas, llm)

        act_types, total_amount = _aggregate_metadata(metadatas)

        # --- Community context injection (Fase C) ---------------------------
        community_context = ""
        if getattr(request, "use_lgraph", False):
            community_context = await self._fetch_community_context(
                request=request, date_from_str=ds, date_to_str=de, repo=repo,
            )

        # --- Build evidence + LLM call --------------------------------------
        evidence = _build_evidence_block(chunks, metadatas)
        if community_context:
            evidence = community_context + "\n\n---\n\n" + evidence

        domain_prompts = get_domain_prompts(request.domain)
        system_prompt = request.system_prompt or domain_prompts["system"]
        user_prompt = domain_prompts["user_template"].format(
            chunk_count=len(chunks),
            namespace=request.namespace,
            date_from=ds,
            date_to=de,
            evidence=evidence,
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            digest_text = _extract_llm_text(response)
        except Exception as e:
            logger.error(f"LLM call failed for digest {ds}–{de}: {e}")
            digest_text = f"[Errore generazione: {e}]"

        vector_id = await self._index_digest(
            repo=repo, llm_embeddings=llm_embeddings,
            request=request, digest_text=digest_text,
            win_start=win_start, win_end=win_end,
            chunk_count=len(chunks), act_types=act_types,
            total_amount=total_amount,
        )

        return DigestGenerationResult(
            namespace=request.namespace,
            date_from=ds, date_to=de,
            granularity=request.granularity,
            content=digest_text,
            chunk_count=len(chunks),
            act_types=act_types,
            total_amount=total_amount,
            digest_vector_id=vector_id,
        )

    async def _fetch_source_chunks(self, repo, request, date_from_str, date_to_str):
        qa = self._build_qa(request, question=f"atti amministrativi {date_from_str} {date_to_str}")
        qa.top_k = request.top_k
        qa._metadata_filter = {
            request.date_metadata_field: {"$gte": date_from_str, "$lte": date_to_str},
            _DIGEST_TYPE_FIELD: {"$ne": _DIGEST_TYPE_VALUE},
        }
        try:
            return await repo.get_chunks_from_repo(qa)
        except Exception as e:
            logger.warning(f"Chunk retrieval failed: {e}")
            from tilellm.models.schemas import RetrievalChunksResult
            return RetrievalChunksResult(chunks=[], metadata=[], namespace=qa.namespace)

    async def _fetch_digests(self, repo, request, date_from_str, date_to_str, top_k=20):
        qa = self._build_qa(request, question=f"rapporto {date_from_str} {date_to_str}")
        qa.top_k = top_k
        qa._metadata_filter = {
            _DIGEST_TYPE_FIELD: {"$eq": _DIGEST_TYPE_VALUE},
            "digest_date_from": {"$gte": date_from_str},
            "digest_date_to": {"$lte": date_to_str},
        }
        try:
            return await repo.get_chunks_from_repo(qa)
        except Exception as e:
            logger.warning(f"Digest retrieval failed: {e}")
            from tilellm.models.schemas import RetrievalChunksResult
            return RetrievalChunksResult(chunks=[], metadata=[], namespace=qa.namespace)

    async def _index_digest(
        self, repo, llm_embeddings, request,
        digest_text: str, win_start: date, win_end: date,
        chunk_count: int, act_types: Dict[str, int], total_amount: Optional[float],
    ) -> Optional[str]:
        doc = Document(
            page_content=digest_text,
            metadata={
                "id": f"digest_{request.namespace}_{win_start.isoformat()}_{win_end.isoformat()}",
                "metadata_id": f"digest_{request.namespace}_{win_start.isoformat()}",
                "doc_id": f"digest_{request.namespace}_{win_start.isoformat()}",
                _DIGEST_TYPE_FIELD: _DIGEST_TYPE_VALUE,
                "digest_date_from": win_start.isoformat(),
                "digest_date_to": win_end.isoformat(),
                "digest_granularity": request.granularity,
                "namespace": request.namespace,
                "chunk_count": chunk_count,
                "act_types_json": json.dumps(act_types, ensure_ascii=False),
                "total_amount": total_amount,
                "domain": request.domain or "generic",
            },
        )
        try:
            ids = await repo.aadd_documents(
                engine=request.engine,
                documents=[doc],
                namespace=request.namespace,
                embedding_model=llm_embeddings,
                metadata_id=doc.metadata["metadata_id"],
            )
            return ids[0] if ids else None
        except Exception as e:
            logger.error(f"Failed to index digest: {e}")
            return None

    # ------------------------------------------------------------------
    # Query path
    # ------------------------------------------------------------------

    async def query(
        self,
        request: DigestQueryRequest,
        repo,
        llm,
        llm_embeddings,
    ) -> DigestQueryResponse:
        mode = request.query_mode
        if mode == "auto":
            mode, matched_pattern = classify_query_debug(request.question)
            logger.info(f"[query_router] auto → {mode!r} | pattern={matched_pattern!r}")
        else:
            logger.info(f"[query_router] explicit mode={mode!r}")

        if mode == "temporal":
            return await self._query_temporal(request, repo, llm)
        else:
            return await self._query_semantic(request, repo, llm)

    async def _query_temporal(self, request: DigestQueryRequest, repo, llm) -> DigestQueryResponse:
        """Retrieve pre-computed PA rapports and synthesise an answer."""
        date_from_str = request.date_from.isoformat() if request.date_from else None
        date_to_str = (request.date_to or request.date_from).isoformat() if request.date_from else None

        qa = self._build_qa(request, question=request.question)
        qa.top_k = 20
        qa._metadata_filter = {_DIGEST_TYPE_FIELD: {"$eq": _DIGEST_TYPE_VALUE}}
        if date_from_str:
            qa._metadata_filter["digest_date_from"] = {"$gte": date_from_str}
        if date_to_str:
            qa._metadata_filter["digest_date_to"] = {"$lte": date_to_str}

        try:
            retrieval = await repo.get_chunks_from_repo(qa)
        except Exception as e:
            logger.warning(f"Temporal retrieval failed: {e}")
            return DigestQueryResponse(
                answer="Impossibile recuperare i rapporti per il periodo richiesto.",
                query_mode="temporal",
            )

        chunks = retrieval.chunks
        metadatas = retrieval.metadata

        if not chunks:
            return DigestQueryResponse(
                answer=(
                    "Nessun rapporto trovato per il periodo richiesto. "
                    "Generare prima i rapporti con l'endpoint /api/digest/generate."
                ),
                query_mode="temporal",
                chunk_count=0,
            )

        chunks, metadatas = await self._apply_reranking(
            request.question, chunks, metadatas, len(chunks), qa
        )

        domain_prompts = get_domain_prompts(getattr(request, "domain", None))
        query_system = domain_prompts.get("query_system", domain_prompts["system"])

        history_text = _format_chat_history(
            request.chat_history_dict or {},
            getattr(request, "max_history_messages", 10),
        )
        history_section = f"Storico conversazione:\n{history_text}\n\n" if history_text else ""

        evidence = _build_evidence_block(chunks, metadatas)
        user_prompt = (
            f"{history_section}"
            f"Di seguito sono riportati i rapporti sull'attività amministrativa "
            f"per il periodo richiesto ({len(chunks)} rapporti).\n\n"
            f"{evidence}\n\n"
            f"Domanda: {request.question}"
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=query_system),
                HumanMessage(content=user_prompt),
            ])
            answer = _extract_llm_text(response)
        except Exception as e:
            answer = f"[Errore risposta LLM: {e}]"

        dates_used = [m.get("digest_date_from", "") for m in metadatas if m.get("digest_date_from")]

        return DigestQueryResponse(
            answer=answer,
            query_mode="temporal",
            sources=[
                {k: v for k, v in m.items() if k not in ("id", "metadata_id", "doc_id")}
                for m in metadatas
            ],
            digests_used=sorted(set(dates_used)),
            chunk_count=len(chunks),
        )

    async def _query_semantic(self, request: DigestQueryRequest, repo, llm) -> DigestQueryResponse:
        """Standard vector search on raw documents (excludes pre-generated rapports)."""
        qa = self._build_qa(request, question=request.question)
        original_top_k = qa.top_k
        qa._metadata_filter = {_DIGEST_TYPE_FIELD: {"$ne": _DIGEST_TYPE_VALUE}}
        if request.date_from:
            qa._metadata_filter[request.date_metadata_field] = {
                "$gte": request.date_from.isoformat(),
                "$lte": (request.date_to or request.date_from).isoformat(),
            }

        if qa.reranker_config:
            qa.top_k = original_top_k * qa.reranking_multiplier

        try:
            retrieval = await repo.get_chunks_from_repo(qa)
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return DigestQueryResponse(
                answer="Recupero degli atti non riuscito.",
                query_mode="semantic",
            )

        chunks = retrieval.chunks
        metadatas = retrieval.metadata

        if not chunks:
            return DigestQueryResponse(
                answer="Nessun atto rilevante trovato.",
                query_mode="semantic",
                chunk_count=0,
            )

        chunks, metadatas = await self._apply_reranking(
            request.question, chunks, metadatas, original_top_k, qa
        )

        domain_prompts = get_domain_prompts(getattr(request, "domain", None))
        query_system = domain_prompts.get("query_system", domain_prompts["system"])

        history_text = _format_chat_history(
            request.chat_history_dict or {},
            getattr(request, "max_history_messages", 10),
        )
        history_section = f"Storico conversazione:\n{history_text}\n\n" if history_text else ""

        evidence = _build_evidence_block(chunks, metadatas)
        user_prompt = (
            f"{history_section}"
            f"Di seguito sono riportati {len(chunks)} atti amministrativi pertinenti alla domanda.\n\n"
            f"{evidence}\n\n"
            f"Domanda: {request.question}"
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=query_system),
                HumanMessage(content=user_prompt),
            ])
            answer = _extract_llm_text(response)
        except Exception as e:
            answer = f"[Errore risposta LLM: {e}]"

        return DigestQueryResponse(
            answer=answer,
            query_mode="semantic",
            sources=[
                {k: v for k, v in m.items() if k not in ("id", "metadata_id", "doc_id")}
                for m in metadatas
            ],
            chunk_count=len(chunks),
        )

    # ------------------------------------------------------------------
    # Agentic query
    # ------------------------------------------------------------------

    async def _extract_query_params(
        self,
        question: str,
        history_text: str,
        today: "date",
        llm,
    ) -> dict:
        from datetime import date as _date, timedelta

        yesterday = (today - timedelta(days=1)).isoformat()
        days_since_monday = today.weekday()
        last_mon = (today - timedelta(days=days_since_monday + 7)).isoformat()
        last_sun = (today - timedelta(days=days_since_monday + 1)).isoformat()
        month_start = today.replace(day=1).isoformat()

        system_prompt = (
            f"Sei un assistente che analizza domande su atti amministrativi di strutture sanitarie italiane.\n"
            f"Oggi è {today.isoformat()}.\n\n"
            "Estrai i parametri di ricerca dalla domanda. "
            "Rispondi SOLO con JSON valido (niente markdown, niente testo extra).\n\n"
            "Schema:\n"
            '{"date_from": "YYYY-MM-DD o null", "date_to": "YYYY-MM-DD o null", '
            '"query_mode": "temporal|semantic", "reasoning": "motivazione breve"}\n\n'
            "Regole query_mode:\n"
            "- temporal: riassunti, conteggi, cosa hanno fatto/deliberato/speso in un periodo\n"
            "- semantic: ricerca di contenuto specifico ('hanno acquistato X?', 'c'è un atto su Y?')\n\n"
            "Regole date (usa sempre ISO YYYY-MM-DD):\n"
            f"- 'ieri' → {yesterday}\n"
            f"- 'oggi' → {today.isoformat()}\n"
            f"- 'la settimana scorsa' → {last_mon} / {last_sun}\n"
            f"- 'questo mese' → {month_start} / {today.isoformat()}\n"
            "- 'ad aprile' (anno corrente) → 2026-04-01 / 2026-04-30\n"
            "- nessuna data menzionata → date_from: null, date_to: null"
        )

        history_section = f"Storico conversazione:\n{history_text}\n\n" if history_text else ""
        user_content = f"{history_section}Domanda utente: {question}"

        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            text = _extract_llm_text(response).strip()
            text = re.sub(r"```[a-z]*\n?", "", text).strip("`\n ")
            params = json.loads(text)
            for field in ("date_from", "date_to"):
                val = params.get(field)
                if val and val != "null":
                    try:
                        _date.fromisoformat(val)
                    except (ValueError, TypeError):
                        params[field] = None
                else:
                    params[field] = None
            params.setdefault("query_mode", "auto")
            params.setdefault("reasoning", "")
            return params
        except Exception as exc:
            logger.warning(f"[agent] param extraction failed: {exc}")
            return {"date_from": None, "date_to": None, "query_mode": "auto", "reasoning": str(exc)}

    async def agent_query(
        self,
        request: DigestAgentRequest,
        repo,
        llm,
        llm_embeddings,
    ) -> DigestAgentResponse:
        from datetime import date as _date

        today = request.today or _date.today()
        history_text = _format_chat_history(
            request.chat_history_dict or {},
            request.max_history_messages,
        )

        params = await self._extract_query_params(request.question, history_text, today, llm)
        logger.info(
            f"[agent] extracted: date_from={params['date_from']} "
            f"date_to={params['date_to']} mode={params['query_mode']}"
        )

        date_from = _date.fromisoformat(params["date_from"]) if params.get("date_from") else None
        date_to = _date.fromisoformat(params["date_to"]) if params.get("date_to") else None

        query_request = DigestQueryRequest(
            question=request.question,
            namespace=request.namespace,
            engine=request.engine,
            embedding=request.embedding,
            gptkey=request.gptkey,
            model=request.model,
            llm=request.llm,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            debug=request.debug,
            top_k=request.top_k,
            sparse_encoder=request.sparse_encoder,
            search_type=request.search_type,
            reranking=request.reranking,
            reranker_model=request.reranker_model,
            reranking_multiplier=request.reranking_multiplier,
            date_metadata_field=request.date_metadata_field,
            tags=request.tags,
            date_from=date_from,
            date_to=date_to,
            query_mode=params["query_mode"],
            chat_history_dict=request.chat_history_dict,
            max_history_messages=request.max_history_messages,
        )

        base_response = await self.query(query_request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)

        return DigestAgentResponse(
            **base_response.model_dump(),
            extracted_date_from=params.get("date_from"),
            extracted_date_to=params.get("date_to"),
            extracted_query_mode=params.get("query_mode", "auto"),
            agent_reasoning=params.get("reasoning"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _apply_reranking(
        self,
        question: str,
        chunks: list,
        metadatas: list,
        top_k: int,
        qa: QuestionAnswer,
    ) -> tuple:
        if not qa.reranker_config or len(chunks) <= 1:
            return chunks[:top_k], metadatas[:top_k]

        from tilellm.tools.reranker import TileReranker
        docs = [Document(page_content=c, metadata=m) for c, m in zip(chunks, metadatas)]
        reranker = TileReranker(model_name=qa.reranker_config)
        reranked = await reranker.arerank_documents(
            query=question, documents=docs, top_k=top_k
        )
        return (
            [d.page_content for d in reranked],
            [d.metadata for d in reranked],
        )

    def _build_qa(self, request, question: str) -> QuestionAnswer:
        gptkey = getattr(request, "gptkey", None) or "sk"
        sparse_encoder = getattr(request, "sparse_encoder", None)
        search_type = getattr(request, "search_type", "similarity")
        return QuestionAnswer(
            question=question,
            namespace=request.namespace,
            engine=request.engine,
            embedding=request.embedding,
            gptkey=gptkey,
            model=getattr(request, "model", "gpt-4o-mini"),
            top_k=getattr(request, "top_k", 5),
            sparse_encoder=sparse_encoder,
            search_type=search_type,
            reranking=getattr(request, "reranking", False),
            reranker_model=getattr(request, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            reranking_multiplier=getattr(request, "reranking_multiplier", 3),
            tags=getattr(request, "tags", None),
        )
