"""
Builds a Light GraphRAG structure in FalkorDB from pre-extracted entity data.

Node types
----------
LChunk  — one node per vector-store chunk
LEntity — one node per unique (normalized_name, entity_type) pair

Edge types
----------
HAS_ENTITY  LChunk → LEntity  weight = term-frequency of entity in chunk
CO_OCCURS   LEntity — LEntity  weight = NPMI (Normalized PMI) of co-occurrence

Graph naming: ``lgraph_{namespace}_{index_name}``
"""

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LGRAPH_PREFIX = "lgraph"
_TEXT_SNIPPET_LEN = 1500  # chars stored in LChunk.text
_BATCH_SIZE = 200          # UNWIND batch size for FalkorDB queries


def make_graph_name(namespace: str, index_name: str) -> str:
    safe_ns = namespace.replace(" ", "_")
    safe_idx = index_name.replace(" ", "_")
    return f"{_LGRAPH_PREFIX}_{safe_ns}_{safe_idx}"


async def build_light_graph(
    repo,
    chunks: List[Dict[str, Any]],
    chunk_entities: Dict[str, List[Tuple[str, str]]],
    entity_doc_freq: Dict[Tuple[str, str], int],
    namespace: str,
    index_name: str,
    npmi_threshold: float,
    npmi_min_count: int,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Persist the light graph into FalkorDB.

    Steps
    -----
    1. Optionally delete existing graph (overwrite=True).
    2. Batch-create LChunk nodes.
    3. Batch-create LEntity nodes.
    4. Create HAS_ENTITY edges weighted by TF.
    5. Compute NPMI for entity pairs and create CO_OCCURS edges.

    Returns a stats dict consumed by the caller.
    """
    graph_name = make_graph_name(namespace, index_name)
    n_chunks = len(chunks)

    if overwrite:
        try:
            await repo.delete_graph(graph_name)
            logger.info(f"Deleted existing graph '{graph_name}' for overwrite")
        except Exception as e:
            logger.warning(f"Could not delete graph '{graph_name}': {e}")

    # ---- 1. LChunk nodes ------------------------------------------------
    chunk_params = [
        {
            "chunk_id": c["id"],
            "text": (c.get("text") or "")[:_TEXT_SNIPPET_LEN],
            "metadata_id": c.get("metadata_id") or "",
            "source": c.get("source") or "",
            "namespace": namespace,
            "index_name": index_name,
        }
        for c in chunks
    ]

    chunk_id_to_falkor: Dict[str, str] = {}
    for offset in range(0, len(chunk_params), _BATCH_SIZE):
        batch = chunk_params[offset: offset + _BATCH_SIZE]
        q = """
        UNWIND $nodes AS nd
        MERGE (c:LChunk {chunk_id: nd.chunk_id, namespace: nd.namespace, index_name: nd.index_name})
        SET c.text = nd.text, c.metadata_id = nd.metadata_id, c.source = nd.source
        RETURN id(c) AS id, c.chunk_id AS chunk_id
        """
        rows = await repo._execute_query(q, {"nodes": batch}, graph_name=graph_name)
        for r in rows:
            chunk_id_to_falkor[r["chunk_id"]] = str(r["id"])

    logger.info(f"LChunk nodes: {len(chunk_id_to_falkor)} created/merged in '{graph_name}'")

    # ---- 2. LEntity nodes -----------------------------------------------
    unique_entities: Dict[Tuple[str, str], None] = {}
    for ents in chunk_entities.values():
        for e in ents:
            unique_entities[e] = None

    entity_params = [
        {"name": name, "entity_type": etype, "namespace": namespace, "index_name": index_name}
        for (name, etype) in unique_entities
    ]

    entity_name_to_falkor: Dict[str, str] = {}  # normalized_name → falkor_id
    for offset in range(0, len(entity_params), _BATCH_SIZE):
        batch = entity_params[offset: offset + _BATCH_SIZE]
        q = """
        UNWIND $nodes AS nd
        MERGE (e:LEntity {name: nd.name, namespace: nd.namespace, index_name: nd.index_name})
        SET e.entity_type = nd.entity_type
        RETURN id(e) AS id, e.name AS name
        """
        rows = await repo._execute_query(q, {"nodes": batch}, graph_name=graph_name)
        for r in rows:
            entity_name_to_falkor[r["name"]] = str(r["id"])

    logger.info(f"LEntity nodes: {len(entity_name_to_falkor)} created/merged in '{graph_name}'")

    # ---- 3. HAS_ENTITY edges (TF-weighted) --------------------------------
    has_entity_params: List[Dict[str, Any]] = []
    for chunk in chunks:
        cid = chunk["id"]
        ents = chunk_entities.get(cid, [])
        if not ents:
            continue

        counts = Counter(name for (name, _) in ents)
        total = sum(counts.values()) or 1
        for (name, _etype), cnt in Counter((n, e) for (n, e) in ents).items():
            tf = counts[name] / total
            c_fid = chunk_id_to_falkor.get(cid)
            e_fid = entity_name_to_falkor.get(name)
            if c_fid and e_fid:
                has_entity_params.append(
                    {"chunk_fid": int(c_fid), "entity_fid": int(e_fid), "tf": tf}
                )

    for offset in range(0, len(has_entity_params), _BATCH_SIZE):
        batch = has_entity_params[offset: offset + _BATCH_SIZE]
        q = """
        UNWIND $edges AS ed
        MATCH (c:LChunk) WHERE id(c) = ed.chunk_fid
        MATCH (e:LEntity) WHERE id(e) = ed.entity_fid
        MERGE (c)-[r:HAS_ENTITY]->(e)
        SET r.tf = ed.tf
        """
        await repo._execute_query(q, {"edges": batch}, graph_name=graph_name)

    logger.info(f"HAS_ENTITY edges: {len(has_entity_params)} in '{graph_name}'")

    # ---- 4. CO_OCCURS edges (NPMI) ----------------------------------------
    entity_freq: Dict[str, int] = defaultdict(int)
    co_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cid, ents in chunk_entities.items():
        names = list({name for (name, _) in ents})
        for name in names:
            entity_freq[name] += 1
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                co_counts[n1][n2] += 1
                co_counts[n2][n1] += 1

    co_occurs_params: List[Dict[str, Any]] = []
    for n1, partners in co_counts.items():
        for n2, count in partners.items():
            if n2 <= n1:  # process each pair once
                continue
            if count < npmi_min_count:
                continue
            p_xy = count / n_chunks
            p_x = entity_freq[n1] / n_chunks
            p_y = entity_freq[n2] / n_chunks
            if p_xy == 0 or p_x == 0 or p_y == 0:
                continue
            pmi = math.log(p_xy / (p_x * p_y))
            # NPMI normalised to [-1, 1]: pmi / -log(P(x,y))
            npmi = pmi / (-math.log(p_xy + 1e-12))
            if npmi < npmi_threshold:
                continue
            e1_fid = entity_name_to_falkor.get(n1)
            e2_fid = entity_name_to_falkor.get(n2)
            if e1_fid and e2_fid:
                co_occurs_params.append(
                    {"e1_fid": int(e1_fid), "e2_fid": int(e2_fid), "npmi": npmi, "count": count}
                )

    for offset in range(0, len(co_occurs_params), _BATCH_SIZE):
        batch = co_occurs_params[offset: offset + _BATCH_SIZE]
        q = """
        UNWIND $edges AS ed
        MATCH (e1:LEntity) WHERE id(e1) = ed.e1_fid
        MATCH (e2:LEntity) WHERE id(e2) = ed.e2_fid
        MERGE (e1)-[r:CO_OCCURS]-(e2)
        SET r.npmi = ed.npmi, r.count = ed.count
        """
        await repo._execute_query(q, {"edges": batch}, graph_name=graph_name)

    logger.info(f"CO_OCCURS edges: {len(co_occurs_params)} in '{graph_name}'")

    return {
        "graph_name": graph_name,
        "chunks_processed": n_chunks,
        "entities_created": len(entity_name_to_falkor),
        "entity_chunk_edges": len(has_entity_params),
        "entity_entity_edges": len(co_occurs_params),
    }
