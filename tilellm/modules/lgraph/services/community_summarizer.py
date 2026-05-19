"""
LinearRAG-style community summarization for the lgraph module.

For each Leiden community in the CO_OCCURS graph:
  1. Load all LEntity nodes assigned to that community.
  2. Load all LChunk nodes connected to those entities (HAS_ENTITY edges).
  3. Generate an LLM summary describing the thematic area covered by the community.
  4. Store the summary as an LCommunitySummary node in FalkorDB.
  5. Index the summary as a LangChain Document in the vector store under the
     namespace ``{original_namespace}__lgraph_communities`` so it can be retrieved
     via dense + sparse search alongside regular RAG results.

Requires that Leiden community detection has been run beforehand
(i.e. LEntity nodes have the community_id property set).
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_COMMUNITY_NS_SUFFIX = "__lgraph_communities"
_MAX_CHUNK_CHARS = 600
_MAX_CONTEXT_CHARS = 60_000
_BATCH_SIZE = 200

_SUMMARIZE_SYSTEM = (
    "Sei un analista esperto in atti amministrativi di strutture sanitarie pubbliche italiane (ASL). "
    "Ti vengono forniti un gruppo di entità correlate (persone, organizzazioni, codici CIG/CUP, "
    "importi, date) e i frammenti degli atti che le citano. "
    "Produci un sommario tematico del dominio trattato da questo gruppo, in modo da poter "
    "rispondere a domande su: chi è coinvolto, quale tipo di attività amministrativa è rappresentata, "
    "quali importi e oggetti ricorrono.\n\n"
    "REGOLE:\n"
    "1. Il sommario deve essere chiaro, denso di informazioni e non più lungo di 400 parole.\n"
    "2. Inizia con un titolo sintetico che cattura il tema principale (es. 'Appalti dispositivi medici - Fornitore X').\n"
    "3. Elenca le entità chiave: fornitori, responsabili, CIG/CUP, importi ricorrenti.\n"
    "4. Descrivi il pattern di attività (liquidazioni periodiche, gare ripetute, ecc.).\n"
    "5. Segnala eventuali elementi di attenzione (stesso fornitore + importi frazionati, gare deserte, ecc.).\n"
    "Formato: markdown. Niente prefissi come 'Sommario:', vai diretto al titolo."
)


async def generate_community_summaries(
    repo,
    llm,
    llm_embeddings,
    vector_repo,
    engine,
    namespace: str,
    index_name: str,
    min_community_size: int = 3,
    max_chunks_per_community: int = 30,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate LLM summaries for each Leiden community and persist them in:
    - FalkorDB: LCommunitySummary node linked to community entities.
    - Vector store: Document in ``{namespace}__lgraph_communities`` namespace.

    Parameters
    ----------
    repo           : AsyncFalkorGraphRepository
    llm            : LangChain chat model
    llm_embeddings : LangChain embeddings model
    vector_repo    : vector store repository (aadd_documents interface)
    engine         : Engine config for the vector store
    namespace / index_name : identify the lgraph
    min_community_size     : skip communities with fewer entities
    max_chunks_per_community : context budget per LLM call
    overwrite      : if True, regenerate summaries that already exist

    Returns a stats dict.
    """
    from .graph_builder import make_graph_name

    graph_name = make_graph_name(namespace, index_name)
    community_ns = f"{namespace}{_COMMUNITY_NS_SUFFIX}"

    # ---- 1. Discover communities ----------------------------------------
    community_rows = await repo._execute_query(
        "MATCH (e:LEntity) "
        "WHERE e.namespace=$ns AND e.index_name=$idx AND e.community_id IS NOT NULL "
        "RETURN e.community_id AS comm_id, count(e) AS entity_count",
        {"ns": namespace, "idx": index_name},
        graph_name=graph_name,
    )

    if not community_rows:
        return {
            "status": "no_communities",
            "message": "No Leiden communities found. Run /api/lgraph/leiden first.",
            "communities_processed": 0,
        }

    communities = [
        (int(r["comm_id"]), int(r["entity_count"]))
        for r in community_rows
        if int(r["entity_count"]) >= min_community_size
    ]
    logger.info(f"[community_summarizer] {len(communities)} communities ≥ {min_community_size} entities")

    # ---- 2. Check for existing summaries (skip unless overwrite) --------
    if not overwrite:
        existing_rows = await repo._execute_query(
            "MATCH (cs:LCommunitySummary) "
            "WHERE cs.namespace=$ns AND cs.index_name=$idx "
            "RETURN cs.community_id AS comm_id",
            {"ns": namespace, "idx": index_name},
            graph_name=graph_name,
        )
        existing_ids = {int(r["comm_id"]) for r in existing_rows}
        communities = [(cid, cnt) for cid, cnt in communities if cid not in existing_ids]
        logger.info(f"[community_summarizer] {len(communities)} communities to generate (skipping {len(existing_ids)} existing)")

    processed = 0
    errors = 0
    vector_docs: List[Document] = []

    for comm_id, entity_count in communities:
        try:
            summary_text, label = await _summarize_community(
                repo=repo,
                llm=llm,
                graph_name=graph_name,
                namespace=namespace,
                index_name=index_name,
                comm_id=comm_id,
                max_chunks=max_chunks_per_community,
            )

            # --- Store in FalkorDB ---
            await _upsert_community_summary_node(
                repo=repo,
                graph_name=graph_name,
                namespace=namespace,
                index_name=index_name,
                comm_id=comm_id,
                label=label,
                summary_text=summary_text,
                entity_count=entity_count,
            )

            # --- Collect for batch vector indexing ---
            vector_docs.append(Document(
                page_content=summary_text,
                metadata={
                    "id": f"comm_summary_{namespace}_{comm_id}",
                    "metadata_id": f"comm_summary_{namespace}_{comm_id}",
                    "doc_id": f"comm_summary_{namespace}_{comm_id}",
                    "community_id": comm_id,
                    "community_label": label,
                    "entity_count": entity_count,
                    "namespace": community_ns,
                    "source_namespace": namespace,
                    "index_name": index_name,
                    "summary_type": "community",
                },
            ))

            processed += 1
            logger.debug(f"[community_summarizer] comm {comm_id}: '{label}' — {len(summary_text)} chars")

        except Exception as e:
            logger.error(f"[community_summarizer] comm {comm_id} failed: {e}", exc_info=True)
            errors += 1

    # ---- 3. Batch index to vector store ----------------------------------
    indexed_count = 0
    if vector_docs:
        for offset in range(0, len(vector_docs), 50):
            batch = vector_docs[offset: offset + 50]
            try:
                ids = await vector_repo.aadd_documents(
                    engine=engine,
                    documents=batch,
                    namespace=community_ns,
                    embedding_model=llm_embeddings,
                )
                indexed_count += len(ids)
            except Exception as e:
                logger.error(f"[community_summarizer] vector store batch failed: {e}", exc_info=True)
                errors += 1

    logger.info(
        f"[community_summarizer] done: {processed} summaries generated, "
        f"{indexed_count} indexed to vector store, {errors} errors"
    )

    return {
        "status": "success" if errors == 0 else "partial",
        "graph_name": graph_name,
        "community_ns": community_ns,
        "communities_processed": processed,
        "communities_indexed": indexed_count,
        "errors": errors,
        "message": (
            f"{processed} community summaries generated and indexed to '{community_ns}'."
            if errors == 0
            else f"{processed} processed, {errors} errors — check logs."
        ),
    }


async def _summarize_community(
    repo,
    llm,
    graph_name: str,
    namespace: str,
    index_name: str,
    comm_id: int,
    max_chunks: int,
) -> Tuple[str, str]:
    """Generate an LLM summary for one community. Returns (summary_text, label)."""

    # Load entities in this community
    entity_rows = await repo._execute_query(
        "MATCH (e:LEntity) "
        "WHERE e.namespace=$ns AND e.index_name=$idx AND e.community_id=$cid "
        "RETURN e.name AS name, e.entity_type AS etype "
        "ORDER BY e.entity_type",
        {"ns": namespace, "idx": index_name, "cid": comm_id},
        graph_name=graph_name,
    )

    entities_by_type: Dict[str, List[str]] = {}
    for r in entity_rows:
        etype = r.get("etype", "MISC")
        entities_by_type.setdefault(etype, []).append(r.get("name", ""))

    entity_summary_lines = []
    for etype, names in sorted(entities_by_type.items()):
        entity_summary_lines.append(f"  {etype}: {', '.join(names[:20])}")
    entity_block = "\n".join(entity_summary_lines) if entity_summary_lines else "(nessuna entità)"

    # Load chunk texts from entities in this community (limit for context budget)
    chunk_rows = await repo._execute_query(
        "MATCH (e:LEntity)-[:HAS_ENTITY]-(c:LChunk) "
        "WHERE e.namespace=$ns AND e.index_name=$idx AND e.community_id=$cid "
        "RETURN DISTINCT c.chunk_id AS cid, c.text AS text, c.source AS source "
        f"LIMIT {max_chunks}",
        {"ns": namespace, "idx": index_name, "cid": comm_id},
        graph_name=graph_name,
    )

    lines: List[str] = []
    total_chars = 0
    for i, row in enumerate(chunk_rows, 1):
        snippet = (row.get("text") or "")[:_MAX_CHUNK_CHARS]
        source = row.get("source", "")
        entry = f"[{i}] {source}\n{snippet}\n"
        if total_chars + len(entry) > _MAX_CONTEXT_CHARS:
            break
        lines.append(entry)
        total_chars += len(entry)

    chunks_block = "\n".join(lines) if lines else "(nessun frammento disponibile)"

    user_content = (
        f"Entità della comunità {comm_id}:\n{entity_block}\n\n"
        f"Frammenti di atti che citano queste entità:\n{chunks_block}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=_SUMMARIZE_SYSTEM),
        HumanMessage(content=user_content),
    ])

    summary_text = _extract_llm_text(response)

    # Derive a short label from the first non-empty line of the summary
    label = _extract_label(summary_text, comm_id)

    return summary_text, label


def _extract_label(text: str, comm_id: int) -> str:
    for line in text.splitlines():
        stripped = line.lstrip("#").strip(" *_")
        if stripped and len(stripped) > 3:
            return stripped[:80]
    return f"Comunità {comm_id}"


def _extract_llm_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return str(content).strip()


async def _upsert_community_summary_node(
    repo,
    graph_name: str,
    namespace: str,
    index_name: str,
    comm_id: int,
    label: str,
    summary_text: str,
    entity_count: int,
) -> None:
    """Persist LCommunitySummary node in FalkorDB (MERGE to handle re-runs)."""
    # Truncate summary stored in graph to 2000 chars — full text is in vector store
    stored_text = summary_text[:2000]
    await repo._execute_query(
        "MERGE (cs:LCommunitySummary {community_id: $cid, namespace: $ns, index_name: $idx}) "
        "SET cs.label = $label, cs.summary = $summary, cs.entity_count = $entity_count",
        {
            "cid": comm_id,
            "ns": namespace,
            "idx": index_name,
            "label": label,
            "summary": stored_text,
            "entity_count": entity_count,
        },
        graph_name=graph_name,
    )
    # Link community entities to their summary
    await repo._execute_query(
        "MATCH (cs:LCommunitySummary {community_id: $cid, namespace: $ns, index_name: $idx}) "
        "MATCH (e:LEntity {community_id: $cid, namespace: $ns, index_name: $idx}) "
        "MERGE (cs)-[:SUMMARIZES]->(e)",
        {"cid": comm_id, "ns": namespace, "idx": index_name},
        graph_name=graph_name,
    )
