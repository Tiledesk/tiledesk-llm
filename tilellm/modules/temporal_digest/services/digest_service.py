"""
DigestService — core logic for generating and querying temporal digests.

Digest generation:
  1. Retrieve all chunks for the date window via metadata filter + high top_k.
  2. Build an evidence block (truncated per chunk for context budget).
  3. Call the judge LLM to produce a structured summary.
  4. Index the summary as a vector with type="digest" metadata for future retrieval.

Digest query:
  - 'temporal' mode: filter by type="digest" + date range → LLM synthesizes answer.
  - 'semantic' mode: standard vector search on raw chunks (type != "digest").
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

_DIGEST_TYPE_FIELD = "digest_type"   # "digest" marks a pre-generated summary vector
_DIGEST_TYPE_VALUE = "digest"
_MAX_CHUNK_CHARS = 800               # chars per chunk in evidence block
_MAX_EVIDENCE_CHARS = 120_000        # total evidence block budget (increased for lgraph augmentation)


def _format_chat_history(chat_history_dict: dict, max_messages: int = 10) -> str:
    """Format conversation history as plain text for LLM prompts."""
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
    """Yield (start, end) pairs for each window in the range."""
    windows: List[Tuple[date, date]] = []
    current = date_from

    while current <= date_to:
        if granularity == "daily":
            windows.append((current, current))
            current += timedelta(days=1)
        elif granularity == "weekly":
            # ISO week: Monday → Sunday
            week_end = current + timedelta(days=6 - current.weekday())
            windows.append((current, min(week_end, date_to)))
            current = week_end + timedelta(days=1)
        elif granularity == "monthly":
            # First → last day of month
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
        date_from = meta.get("digest_date_from")
        date_to = meta.get("digest_date_to")
        if date_from:
            header = f"[{i}] Digest {date_from}"
            if date_to and date_to != date_from:
                header += f"–{date_to}"
        else:
            header = f"[{i}] {meta.get('file_name', meta.get('source', 'unknown'))} | pag. {meta.get('page', '?')}"
        act_type = meta.get("act_type")
        if act_type:
            header += f" | tipo: {act_type}"
        amount = meta.get("amount")
        if amount:
            header += f" | €{amount:,.2f}"
        snippet = chunk[:_MAX_CHUNK_CHARS]
        entry = f"{header}\n{snippet}\n"
        if total + len(entry) > _MAX_EVIDENCE_CHARS:
            lines.append(f"[…] {len(chunks) - i + 1} ulteriori frammenti omessi per limiti di contesto.")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def _extract_llm_text(response) -> str:
    """Robustly extract string from LangChain LLM response (handles reasoning models)."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return str(content).strip()


def _aggregate_metadata(metadatas: List[dict]) -> Tuple[Dict[str, int], Optional[float]]:
    """Aggregate act_type counts and total amount from chunk metadata."""
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

    async def _fetch_chunks_from_lgraph(
        self,
        request: DigestGenerationRequest,
        date_from_str: str,
        date_to_str: str,
    ) -> Tuple[List[str], List[dict]]:
        """
        Query FalkorDB for DATE_IT entities matching [date_from_str, date_to_str]
        and return the text + metadata of the associated LChunk nodes.

        Dates in FalkorDB are stored as strings in DD/MM/YYYY format (Italian PA style).
        We match entities whose name (normalised date) falls within the requested range.
        """
        try:
            from tilellm.modules.lgraph.logic import _get_falkor_repo
            from tilellm.modules.lgraph.services.graph_builder import make_graph_name as _mgn
            import re
            from datetime import date as _date

            graph_name = _mgn(request.namespace, request.engine.index_name)
            falkor = _get_falkor_repo()

            # Parse requested range
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

            # Fetch all DATE_IT entities + their connected LChunks
            rows = await falkor._execute_query(
                "MATCH (e:LEntity {entity_type: 'DATE_IT'})-[:HAS_ENTITY]-(c:LChunk) "
                "WHERE e.namespace=$ns AND e.index_name=$idx "
                "RETURN e.name AS date_str, c.chunk_id AS chunk_id, "
                "c.text AS text, c.source AS source, c.metadata_id AS metadata_id",
                {"ns": request.namespace, "idx": request.engine.index_name},
                graph_name=graph_name,
            )

            seen_ids: set = set()
            chunks_text: List[str] = []
            chunks_meta: List[dict] = []

            for row in rows:
                d = _parse_it(row.get("date_str", ""))
                if d is None:
                    continue
                if d < df or d > dt:
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

            logger.info(
                f"[lgraph augment] namespace={request.namespace} "
                f"{date_from_str}–{date_to_str}: {len(chunks_text)} extra chunks from FalkorDB"
            )
            return chunks_text, chunks_meta

        except Exception as e:
            logger.warning(f"[lgraph augment] failed, skipping: {e}")
            return [], []

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

        # Check if digest already exists (skip unless force_regenerate)
        if not request.force_regenerate:
            existing = await self._fetch_digests(
                repo=repo,
                request=request,
                date_from_str=ds,
                date_to_str=de,
                top_k=1,
            )
            if existing.chunks:
                logger.info(f"Digest for {ds}–{de} already exists in '{request.namespace}', skipping.")
                return DigestGenerationResult(
                    namespace=request.namespace,
                    date_from=ds,
                    date_to=de,
                    granularity=request.granularity,
                    content=existing.chunks[0] if existing.chunks else "",
                    chunk_count=0,
                    already_existed=True,
                )

        # Retrieve source chunks from vector store
        retrieval = await self._fetch_source_chunks(
            repo=repo,
            request=request,
            date_from_str=ds,
            date_to_str=de,
        )
        chunks = list(retrieval.chunks)
        metadatas = list(retrieval.metadata)

        # Augment with lgraph DATE_IT chunks if requested
        if getattr(request, "use_lgraph", False):
            lgraph_texts, lgraph_metas = await self._fetch_chunks_from_lgraph(
                request=request,
                date_from_str=ds,
                date_to_str=de,
            )
            # Dedup: avoid adding chunks already retrieved from vector store
            existing_ids = {m.get("chunk_id", m.get("metadata_id", "")) for m in metadatas}
            for text, meta in zip(lgraph_texts, lgraph_metas):
                cid = meta.get("chunk_id", "")
                if cid not in existing_ids:
                    chunks.append(text)
                    metadatas.append(meta)
                    existing_ids.add(cid)

        if not chunks:
            logger.info(f"No chunks found for {ds}–{de} in '{request.namespace}'.")
            return DigestGenerationResult(
                namespace=request.namespace,
                date_from=ds,
                date_to=de,
                granularity=request.granularity,
                content="Nessun atto trovato per questo periodo.",
                chunk_count=0,
            )

        # Aggregate metadata stats
        act_types, total_amount = _aggregate_metadata(metadatas)

        # Build evidence block and call LLM
        evidence = _build_evidence_block(chunks, metadatas)
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
            digest_text = f"[Errore generazione digest: {e}]"

        # Index the digest as a vector
        vector_id = await self._index_digest(
            repo=repo,
            llm_embeddings=llm_embeddings,
            request=request,
            digest_text=digest_text,
            win_start=win_start,
            win_end=win_end,
            chunk_count=len(chunks),
            act_types=act_types,
            total_amount=total_amount,
        )

        logger.info(
            f"Digest generated for {ds}–{de}: {len(chunks)} chunks → vector {vector_id}"
        )
        return DigestGenerationResult(
            namespace=request.namespace,
            date_from=ds,
            date_to=de,
            granularity=request.granularity,
            content=digest_text,
            chunk_count=len(chunks),
            act_types=act_types,
            total_amount=total_amount,
            digest_vector_id=vector_id,
        )

    async def _fetch_source_chunks(self, repo, request, date_from_str, date_to_str):
        """Retrieve all raw (non-digest) chunks in the date window."""
        qa = self._build_qa(request, question=f"digest {date_from_str} {date_to_str}")
        qa.top_k = request.top_k
        # Store date filter hint in a custom attribute that repos may honour
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
        """Retrieve existing digest vectors for the given window."""
        qa = self._build_qa(request, question=f"digest {date_from_str} {date_to_str}")
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
        self,
        repo,
        llm_embeddings,
        request,
        digest_text: str,
        win_start: date,
        win_end: date,
        chunk_count: int,
        act_types: Dict[str, int],
        total_amount: Optional[float],
    ) -> Optional[str]:
        """Index the digest text as a single vector with digest metadata."""
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
        """Route query to temporal or semantic path and return an answer."""
        mode = request.query_mode
        if mode == "auto":
            mode, matched_pattern = classify_query_debug(request.question)
            logger.info(
                f"[query_router] auto → {mode!r} | "
                f"pattern={matched_pattern!r} | "
                f"question={request.question!r}"
            )
        else:
            logger.info(f"[query_router] explicit mode={mode!r} | question={request.question!r}")

        if mode == "temporal":
            return await self._query_temporal(request, repo, llm)
        else:
            return await self._query_semantic(request, repo, llm)

    async def _query_temporal(self, request: DigestQueryRequest, repo, llm) -> DigestQueryResponse:
        """Retrieve pre-computed digests and synthesize an answer."""
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
                answer="Impossibile recuperare i digest per il periodo richiesto.",
                query_mode="temporal",
            )

        chunks = retrieval.chunks
        metadatas = retrieval.metadata

        if not chunks:
            return DigestQueryResponse(
                answer="Nessun digest trovato per il periodo richiesto. Prova a generare i digest prima.",
                query_mode="temporal",
                chunk_count=0,
            )

        # Rerank digest candidates if enabled (keeps all digests unless top_k is tighter)
        chunks, metadatas = await self._apply_reranking(
            request.question, chunks, metadatas, len(chunks), qa
        )

        history_text = _format_chat_history(
            request.chat_history_dict or {},
            getattr(request, "max_history_messages", 10),
        )
        history_section = f"Storico conversazione:\n{history_text}\n\n" if history_text else ""

        evidence = _build_evidence_block(chunks, metadatas)
        prompt = (
            f"{history_section}"
            f"Di seguito sono riportati TUTTI i digest degli atti amministrativi "
            f"relativi al periodo richiesto ({len(chunks)} digest).\n\n{evidence}\n\n"
            f"ISTRUZIONI:\n"
            f"- Rispondi in modo ESAUSTIVO e COMPLETO basandoti su tutti i digest forniti.\n"
            f"- Non scrivere mai 'non ho informazioni sufficienti' o 'le informazioni sono parziali'.\n"
            f"- Se un'informazione specifica non compare nei digest, dì esplicitamente "
            f"'Nei digest analizzati non risulta alcun riferimento a [X]'.\n"
            f"- Riporta TUTTI gli importi, CIG, CUP, nomi di persone/enti che compaiono.\n\n"
            f"Domanda: {request.question}"
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
        """Standard vector search on raw chunks (excludes digests)."""
        qa = self._build_qa(request, question=request.question)
        original_top_k = qa.top_k
        qa._metadata_filter = {_DIGEST_TYPE_FIELD: {"$ne": _DIGEST_TYPE_VALUE}}
        if request.date_from:
            qa._metadata_filter[request.date_metadata_field] = {
                "$gte": request.date_from.isoformat(),
                "$lte": (request.date_to or request.date_from).isoformat(),
            }

        # Fetch more candidates when reranking is enabled
        if qa.reranker_config:
            qa.top_k = original_top_k * qa.reranking_multiplier

        try:
            retrieval = await repo.get_chunks_from_repo(qa)
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return DigestQueryResponse(
                answer="Recupero vettoriale non riuscito.",
                query_mode="semantic",
            )

        chunks = retrieval.chunks
        metadatas = retrieval.metadata

        if not chunks:
            return DigestQueryResponse(
                answer="Nessun documento rilevante trovato.",
                query_mode="semantic",
                chunk_count=0,
            )

        chunks, metadatas = await self._apply_reranking(
            request.question, chunks, metadatas, original_top_k, qa
        )

        history_text = _format_chat_history(
            request.chat_history_dict or {},
            getattr(request, "max_history_messages", 10),
        )
        history_section = f"Storico conversazione:\n{history_text}\n\n" if history_text else ""

        evidence = _build_evidence_block(chunks, metadatas)
        prompt = (
            f"{history_section}"
            f"Di seguito sono riportati i frammenti di documenti più rilevanti ({len(chunks)} frammenti).\n\n"
            f"{evidence}\n\n"
            f"ISTRUZIONI:\n"
            f"- Rispondi in modo ESAUSTIVO e COMPLETO basandoti su tutti i frammenti forniti.\n"
            f"- Non scrivere mai 'non ho informazioni sufficienti' o 'le informazioni sono parziali'.\n"
            f"- Se un'informazione specifica non compare nei frammenti, dì esplicitamente "
            f"'Nei documenti analizzati non risulta alcun riferimento a [X]'.\n"
            f"- Riporta TUTTI gli importi, CIG, CUP, nomi di persone/enti che compaiono.\n\n"
            f"Domanda: {request.question}"
        )

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
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
        """Call LLM to extract date_from, date_to, query_mode from the user question."""
        from datetime import date as _date, timedelta

        yesterday = (today - timedelta(days=1)).isoformat()
        # Last week: Monday–Sunday
        days_since_monday = today.weekday()
        last_mon = (today - timedelta(days=days_since_monday + 7)).isoformat()
        last_sun = (today - timedelta(days=days_since_monday + 1)).isoformat()
        month_start = today.replace(day=1).isoformat()

        system_prompt = (
            f"Sei un assistente che analizza domande su atti e delibere amministrative italiane.\n"
            f"Oggi è {today.isoformat()}.\n\n"
            "Estrai i parametri di ricerca dalla domanda. "
            "Rispondi SOLO con JSON valido (niente markdown, niente testo extra).\n\n"
            "Schema:\n"
            '{"date_from": "YYYY-MM-DD o null", "date_to": "YYYY-MM-DD o null", '
            '"query_mode": "temporal|semantic", "reasoning": "motivazione breve"}\n\n'
            "Regole query_mode:\n"
            "- temporal: riassunti, conteggi, cosa hanno fatto/deliberato/deciso in un periodo\n"
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
            # Strip markdown fences if present
            text = re.sub(r"```[a-z]*\n?", "", text).strip("`\n ")
            params = json.loads(text)
            # Sanitize dates
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
            logger.warning(f"[agent] param extraction failed: {exc} — falling back to auto")
            return {"date_from": None, "date_to": None, "query_mode": "auto", "reasoning": f"extraction error: {exc}"}

    async def agent_query(
        self,
        request: DigestAgentRequest,
        repo,
        llm,
        llm_embeddings,
    ) -> DigestAgentResponse:
        """Extract query parameters with LLM, then route to temporal or semantic path."""
        from datetime import date as _date

        today = request.today or _date.today()
        history_text = _format_chat_history(
            request.chat_history_dict or {},
            request.max_history_messages,
        )

        params = await self._extract_query_params(request.question, history_text, today, llm)
        logger.info(
            f"[agent] extracted params: date_from={params['date_from']} "
            f"date_to={params['date_to']} mode={params['query_mode']} "
            f"reasoning={params['reasoning']!r}"
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
        """Rerank chunks with TileReranker if enabled, then truncate to top_k."""
        if not qa.reranker_config or len(chunks) <= 1:
            return chunks[:top_k], metadatas[:top_k]

        from tilellm.tools.reranker import TileReranker

        docs = [
            Document(page_content=c, metadata=m)
            for c, m in zip(chunks, metadatas)
        ]
        reranker = TileReranker(model_name=qa.reranker_config)
        reranked = await reranker.arerank_documents(
            query=question, documents=docs, top_k=top_k
        )
        logger.info(
            f"[reranker] {len(docs)} → {len(reranked)} chunks "
            f"(model={qa.reranker_config})"
        )
        return (
            [d.page_content for d in reranked],
            [d.metadata for d in reranked],
        )

    def _build_qa(self, request, question: str) -> QuestionAnswer:
        """Build a minimal QuestionAnswer for repo calls."""
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
