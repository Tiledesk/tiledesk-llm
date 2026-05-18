"""
Leiden community detection on the LEntity CO_OCCURS graph stored in FalkorDB.

Reads only LEntity nodes and CO_OCCURS edges (NPMI-weighted), builds an igraph
Graph, runs g.community_leiden(), then writes community_id back to each LEntity node.
"""
import logging
from typing import Any, Dict

from .graph_builder import make_graph_name

logger = logging.getLogger(__name__)

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    ig = None  # type: ignore


async def run_leiden(
    repo,
    namespace: str,
    index_name: str,
    resolution: float = 1.0,
    min_community_size: int = 3,
) -> Dict[str, Any]:
    """
    Run Leiden clustering on LEntity nodes (CO_OCCURS graph) and persist
    community_id as a property on each LEntity.

    Returns a stats dict: graph_name, community_count, entities_updated.
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError(
            "igraph is required for Leiden clustering. "
            "Install with: pip install igraph"
        )

    graph_name = make_graph_name(namespace, index_name)

    # ---- 1. Load LEntity nodes --------------------------------------------
    node_rows = await repo._execute_query(
        "MATCH (e:LEntity) WHERE e.namespace=$ns AND e.index_name=$idx "
        "RETURN id(e) AS fid, e.name AS name",
        {"ns": namespace, "idx": index_name},
        graph_name=graph_name,
    )

    if not node_rows:
        logger.warning(f"[leiden] No LEntity nodes found in '{graph_name}'")
        return {"status": "empty", "community_count": 0, "entities_updated": 0}

    # ---- 2. Load CO_OCCURS edges ------------------------------------------
    edge_rows = await repo._execute_query(
        "MATCH (e1:LEntity)-[r:CO_OCCURS]-(e2:LEntity) "
        "WHERE e1.namespace=$ns AND e1.index_name=$idx "
        "AND e2.namespace=$ns AND e2.index_name=$idx "
        "AND id(e1) < id(e2) "
        "RETURN id(e1) AS src, id(e2) AS tgt, coalesce(r.npmi, 1.0) AS weight",
        {"ns": namespace, "idx": index_name},
        graph_name=graph_name,
    )

    # ---- 3. Build igraph --------------------------------------------------
    fid_to_idx = {int(r["fid"]): i for i, r in enumerate(node_rows)}
    idx_to_fid = {i: int(r["fid"]) for i, r in enumerate(node_rows)}

    edge_list = []
    weights = []
    for e in edge_rows:
        src_idx = fid_to_idx.get(int(e["src"]))
        tgt_idx = fid_to_idx.get(int(e["tgt"]))
        if src_idx is not None and tgt_idx is not None:
            edge_list.append((src_idx, tgt_idx))
            weights.append(float(e.get("weight") or 1.0))

    g = ig.Graph(len(node_rows), edge_list)  # type: ignore
    g.es["weight"] = weights if weights else [1.0] * len(edge_list)

    # ---- 4. Run Leiden ----------------------------------------------------
    partition = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution_parameter=resolution,
    )
    logger.info(
        f"[leiden] '{graph_name}': {len(partition)} communities "
        f"(res={resolution}, nodes={len(node_rows)}, edges={len(edge_list)})"
    )

    # ---- 5. Write community_id back to FalkorDB ---------------------------
    _BATCH = 200
    updates: list = []
    for comm_id, members in enumerate(partition):
        if len(members) < min_community_size:
            continue
        for node_idx in members:
            fid = idx_to_fid[node_idx]
            updates.append({"fid": fid, "community_id": comm_id})

    entities_updated = 0
    for offset in range(0, len(updates), _BATCH):
        batch = updates[offset: offset + _BATCH]
        await repo._execute_query(
            "UNWIND $rows AS row "
            "MATCH (e:LEntity) WHERE id(e) = row.fid "
            "SET e.community_id = row.community_id",
            {"rows": batch},
            graph_name=graph_name,
        )
        entities_updated += len(batch)

    valid_communities = len([p for p in partition if len(p) >= min_community_size])
    logger.info(f"[leiden] Updated community_id on {entities_updated} LEntity nodes in '{graph_name}'")

    return {
        "status": "success",
        "graph_name": graph_name,
        "community_count": valid_communities,
        "entities_updated": entities_updated,
        "message": f"Leiden complete: {valid_communities} communities, {entities_updated} entities updated",
    }
