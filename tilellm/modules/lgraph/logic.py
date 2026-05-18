"""
Business logic for the Light GraphRAG (lgraph) module.

``build_lgraph`` is decorated with @inject_repo_async so that the vector-store
repository matching ``request.engine`` is automatically injected.
``qa_lgraph`` is decorated with @inject_llm_chat_async (no vector-store needed:
retrieval is entirely via FalkorDB PPR).
``search_lgraph``, ``delete_lgraph``, ``get_lgraph_network``, and
``leiden_lgraph`` interact directly with FalkorDB.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from tilellm.shared.utility import inject_repo_async, inject_llm_chat_async

from .models.schemas import (
    ChunkResult,
    LGraphBuildRequest,
    LGraphDeleteResponse,
    LGraphLeidenRequest,
    LGraphLeidenResponse,
    LGraphNetworkResponse,
    LGraphQARequest,
    LGraphQAResponse,
    LGraphSearchRequest,
    LGraphSearchResponse,
)
from .services.entity_extractor import (
    build_chunk_entity_matrix,
    extract_entities,
    expand_date_references,
    extract_query_keywords,
)
from .services.graph_builder import build_light_graph, make_graph_name
from .services.ppr_retriever import ppr_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FalkorDB repository (shared singleton, lazy-initialized)
# ---------------------------------------------------------------------------
_falkor_repo = None


def _get_falkor_repo():
    global _falkor_repo
    if _falkor_repo is None:
        from tilellm.modules.knowledge_graph_falkor.repository import AsyncFalkorGraphRepository
        _falkor_repo = AsyncFalkorGraphRepository()
    return _falkor_repo


# ---------------------------------------------------------------------------
# BUILD  (runs inside TaskIQ worker)
# ---------------------------------------------------------------------------

@inject_repo_async
async def build_lgraph(
    request: LGraphBuildRequest,
    repo=None,
    **kwargs,
) -> Dict[str, Any]:
    """Fetch all chunks from the vector store and build the light graph in FalkorDB."""
    logger.info(
        f"[lgraph] build started namespace='{request.namespace}' "
        f"engine='{request.engine.name}' index='{request.engine.index_name}'"
    )

    from tilellm.models.schemas import RepositoryItems
    items: RepositoryItems = await repo.get_all_obj_namespace(
        engine=request.engine,
        namespace=request.namespace,
        with_text=True,
    )
    chunks = [
        {
            "id": m.id,
            "text": m.text or "",
            "metadata_id": m.metadata_id or "",
            "source": m.metadata_source or "",
        }
        for m in items.matches
    ]

    if not chunks:
        logger.warning(f"[lgraph] no chunks found in namespace='{request.namespace}'")
        return {
            "status": "empty",
            "namespace": request.namespace,
            "graph_name": make_graph_name(request.namespace, request.engine.index_name),
            "chunks_processed": 0,
            "entities_created": 0,
            "entity_chunk_edges": 0,
            "entity_entity_edges": 0,
            "message": "No chunks found in namespace",
        }

    logger.info(f"[lgraph] {len(chunks)} chunks fetched — extracting entities")

    chunk_entities, entity_doc_freq = build_chunk_entity_matrix(
        chunks=chunks,
        spacy_model=request.spacy_model,
        include_types=request.include_entity_types,
        use_noun_chunks=request.use_noun_chunks,
        sub_window_size=request.sub_window_size,
        sub_window_overlap=request.sub_window_overlap,
    )

    falkor = _get_falkor_repo()
    stats = await build_light_graph(
        repo=falkor,
        chunks=chunks,
        chunk_entities=chunk_entities,
        entity_doc_freq=entity_doc_freq,
        namespace=request.namespace,
        index_name=request.engine.index_name,
        npmi_threshold=request.npmi_threshold,
        npmi_min_count=request.npmi_min_count,
        overwrite=request.overwrite,
    )

    logger.info(
        f"[lgraph] build complete — {stats['chunks_processed']} chunks, "
        f"{stats['entities_created']} entities, "
        f"{stats['entity_chunk_edges']} HAS_ENTITY edges, "
        f"{stats['entity_entity_edges']} CO_OCCURS edges"
    )

    return {
        "status": "success",
        "namespace": request.namespace,
        **stats,
        "message": "Light graph built successfully",
    }


# ---------------------------------------------------------------------------
# SEARCH  (PPR only, no LLM)
# ---------------------------------------------------------------------------

def _build_seed_entity_names(question: str, query_entities) -> List[str]:
    """Merge NLP entities, DATE_IT expansions, and keyword fallback into a seed list."""
    seen: set = set()
    names: List[str] = []

    for name, _ in query_entities:
        if name not in seen:
            seen.add(name)
            names.append(name)

    # Fix 3: expand "aprile 2026" → ["01/04/2026", …, "30/04/2026"]
    for d in expand_date_references(question):
        if d not in seen:
            seen.add(d)
            names.append(d)

    # Last resort: raw keywords from query text when NLP found nothing
    if not names:
        for kw in extract_query_keywords(question):
            if kw not in seen:
                seen.add(kw)
                names.append(kw)

    return names


async def search_lgraph(request: LGraphSearchRequest) -> LGraphSearchResponse:
    """Query the light graph with PPR seeded by query entities."""
    query_entities = extract_entities(
        text=request.question,
        spacy_model=request.spacy_model,
        include_types=request.include_entity_types,
        use_noun_chunks=request.use_noun_chunks,
    )
    entity_names = _build_seed_entity_names(request.question, query_entities)

    gname = make_graph_name(request.namespace, request.engine.index_name)
    falkor = _get_falkor_repo()

    chunk_results = await ppr_search(
        repo=falkor,
        namespace=request.namespace,
        index_name=request.engine.index_name,
        seed_chunk_ids=[],
        seed_entity_names=entity_names,
        top_k=request.top_k,
        alpha=request.ppr_alpha,
        max_iter=request.ppr_max_iter,
        graph_name=gname,
    )

    chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            text=c["text"],
            metadata_id=c["metadata_id"],
            source=c["source"],
            ppr_score=c["ppr_score"],
        )
        for c in chunk_results
    ]

    return LGraphSearchResponse(
        chunks=chunks,
        entities_found=entity_names,
        graph_name=gname,
        query=request.question,
    )


# ---------------------------------------------------------------------------
# QA  (PPR + LLM)
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = """\
Sei un assistente esperto in atti e delibere amministrative della Pubblica Amministrazione italiana.
Ti vengono forniti TUTTI i frammenti di documenti rilevanti per rispondere alla domanda.

REGOLE FONDAMENTALI:
1. Rispondi in modo ESAUSTIVO e COMPLETO basandoti sui frammenti forniti.
2. Non scrivere MAI "non ho informazioni sufficienti" o "le informazioni sono parziali".
3. Se un'informazione specifica non è presente nei frammenti forniti, dì esplicitamente:
   "Nei documenti analizzati non risulta alcun riferimento a [X]."
4. Riporta TUTTI gli importi, CIG, CUP, date e nomi di persone/enti che compaiono nei frammenti.
5. Usa il formato markdown con bullet points per lista di atti/acquisti/importi.
6. Concludi sempre con un riepilogo sintetico se la risposta contiene più voci.
{system_context}"""

_QA_USER_TEMPLATE = """\
Frammenti di documenti recuperati ({chunk_count} risultati, ordinati per rilevanza PPR):

{evidence}

Domanda: {question}"""

_MAX_CHUNK_CHARS = 1200
_MAX_EVIDENCE_CHARS = 80_000


def _build_evidence(chunks: List[ChunkResult]) -> str:
    lines = []
    total = 0
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] fonte: {c.source or 'n/a'} | chunk_id: {c.chunk_id} | ppr: {c.ppr_score:.4f}"
        snippet = c.text[:_MAX_CHUNK_CHARS]
        entry = f"{header}\n{snippet}\n"
        if total + len(entry) > _MAX_EVIDENCE_CHARS:
            lines.append(f"[…] {len(chunks) - i + 1} frammenti omessi per budget contesto.")
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


async def _filter_chunks_by_date(
    chunks: List[ChunkResult],
    date_from: Optional[str],
    date_to: Optional[str],
) -> List[ChunkResult]:
    """
    If date_from/date_to are set, keep only chunks that contain a DATE_IT
    entity within [date_from, date_to].  Falls back to returning all chunks
    if no date filtering is possible.
    """
    if not date_from and not date_to:
        return chunks

    import re
    from datetime import date as _date

    date_re = re.compile(r'(\d{1,2})[/\-\.](\d{2})[/\-\.](\d{4})')

    def _parse_it_date(s: str) -> Optional[_date]:
        m = date_re.match(s)
        if not m:
            return None
        try:
            return _date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None

    try:
        df = _date.fromisoformat(date_from) if date_from else None
        dt = _date.fromisoformat(date_to) if date_to else None
    except ValueError:
        return chunks

    filtered = []
    for chunk in chunks:
        dates_in_text = [_parse_it_date(m.group(0)) for m in date_re.finditer(chunk.text)]
        dates_in_text = [d for d in dates_in_text if d]
        if not dates_in_text:
            # No date found → keep chunk (conservative)
            filtered.append(chunk)
            continue
        in_range = any(
            (df is None or d >= df) and (dt is None or d <= dt)
            for d in dates_in_text
        )
        if in_range:
            filtered.append(chunk)

    return filtered if filtered else chunks


@inject_llm_chat_async
async def qa_lgraph(
    request: LGraphQARequest,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    **kwargs,
) -> LGraphQAResponse:
    """PPR retrieval + LLM answer for Italian PA documents."""
    # ---- 1. Extract entities from question ---------------------------------
    query_entities = extract_entities(
        text=request.question,
        spacy_model=request.spacy_model,
        include_types=request.include_entity_types,
        use_noun_chunks=request.use_noun_chunks,
    )
    entity_names = _build_seed_entity_names(request.question, query_entities)

    gname = make_graph_name(request.namespace, request.engine.index_name)
    falkor = _get_falkor_repo()

    # ---- 2. PPR retrieval --------------------------------------------------
    chunk_results_raw = await ppr_search(
        repo=falkor,
        namespace=request.namespace,
        index_name=request.engine.index_name,
        seed_chunk_ids=[],
        seed_entity_names=entity_names,
        top_k=request.top_k,
        alpha=request.ppr_alpha,
        max_iter=request.ppr_max_iter,
        graph_name=gname,
    )

    chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            text=c["text"],
            metadata_id=c["metadata_id"],
            source=c["source"],
            ppr_score=c["ppr_score"],
        )
        for c in chunk_results_raw
    ]

    # ---- 3. Optional temporal filter ---------------------------------------
    if request.date_from or request.date_to:
        chunks = await _filter_chunks_by_date(chunks, request.date_from, request.date_to)

    # ---- 4. LLM call -------------------------------------------------------
    if not chunks:
        return LGraphQAResponse(
            answer=(
                f"Nessun frammento rilevante trovato nel grafo '{gname}'. "
                "Verifica che il grafo sia stato costruito con /api/lgraph/build."
            ),
            entities_found=entity_names,
            graph_name=gname,
            chunk_count=0,
            chunks_used=[] if request.debug else None,
        )

    evidence = _build_evidence(chunks)
    system_prompt = _QA_SYSTEM_PROMPT.format(
        system_context=f"\n{request.system_context}" if request.system_context else ""
    )
    user_prompt = _QA_USER_TEMPLATE.format(
        chunk_count=len(chunks),
        evidence=evidence,
        question=request.question,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        answer = _extract_llm_text(response)
    except Exception as e:
        logger.error(f"[lgraph/qa] LLM error: {e}", exc_info=True)
        answer = f"[Errore LLM: {e}]"

    return LGraphQAResponse(
        answer=answer,
        entities_found=entity_names,
        graph_name=gname,
        chunk_count=len(chunks),
        chunks_used=chunks if request.debug else None,
    )


# ---------------------------------------------------------------------------
# LEIDEN CLUSTERING
# ---------------------------------------------------------------------------

async def leiden_lgraph(request: LGraphLeidenRequest) -> LGraphLeidenResponse:
    """Run Leiden community detection on LEntity CO_OCCURS graph."""
    from .services.leiden_service import run_leiden
    falkor = _get_falkor_repo()
    stats = await run_leiden(
        repo=falkor,
        namespace=request.namespace,
        index_name=request.engine.index_name,
        resolution=request.resolution,
        min_community_size=request.min_community_size,
    )
    return LGraphLeidenResponse(
        status=stats["status"],
        graph_name=stats["graph_name"],
        community_count=stats["community_count"],
        entities_updated=stats["entities_updated"],
        message=stats.get("message", ""),
    )


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

async def delete_lgraph(namespace: str, index_name: str) -> LGraphDeleteResponse:
    gname = make_graph_name(namespace, index_name)
    falkor = _get_falkor_repo()
    ok = await falkor.delete_graph(gname)
    return LGraphDeleteResponse(
        status="success" if ok else "failed",
        graph_name=gname,
        message=f"Graph '{gname}' deleted" if ok else f"Failed to delete graph '{gname}'",
    )


# ---------------------------------------------------------------------------
# NETWORK (visualization)
# ---------------------------------------------------------------------------

async def get_lgraph_network(
    namespace: str,
    index_name: str,
    node_limit: int = 500,
    edge_limit: int = 2000,
) -> LGraphNetworkResponse:
    gname = make_graph_name(namespace, index_name)
    falkor = _get_falkor_repo()

    node_rows = await falkor._execute_query(
        "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT $lim",
        {"lim": node_limit},
        graph_name=gname,
    )
    nodes = [
        {
            "id": str(r["id"]),
            "label": (r.get("labels") or ["?"])[0],
            "properties": r.get("props") or {},
        }
        for r in node_rows
    ]

    edge_rows = await falkor._execute_query(
        "MATCH (a)-[r]->(b) RETURN id(r) AS id, type(r) AS type, "
        "id(a) AS src, id(b) AS tgt, properties(r) AS props LIMIT $lim",
        {"lim": edge_limit},
        graph_name=gname,
    )
    edges = [
        {
            "id": str(r["id"]),
            "type": r.get("type", ""),
            "source": str(r["src"]),
            "target": str(r["tgt"]),
            "properties": r.get("props") or {},
        }
        for r in edge_rows
    ]

    return LGraphNetworkResponse(
        nodes=nodes,
        edges=edges,
        stats={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_name": gname,
        },
    )
