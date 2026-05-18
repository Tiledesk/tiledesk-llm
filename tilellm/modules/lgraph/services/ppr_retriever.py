"""
Personalized PageRank retrieval on the LGraph entity-chunk bipartite graph.

The subgraph is loaded from FalkorDB on demand, converted to a NetworkX graph,
and PPR is run with seed nodes equal to the query entities and (optionally) the
top-k chunk seeds from a prior vector-store search.

Nodes are addressed by composite key ``"chunk::{chunk_id}"`` or ``"entity::{name}"``.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Minimal Italian stop-word set for keyword splitting (self-contained, no import needed)
_IT_STOP_WORDS: set = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "da", "in", "con", "su", "per", "tra", "fra", "a",
    "e", "o", "ma", "che", "se", "del", "della", "dei", "degli",
    "delle", "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dalla", "dai", "dagli", "dalle", "nel", "nella",
    "nei", "negli", "nelle", "sul", "sulla", "sui", "sugli",
    "sulle", "col", "come", "non", "si", "ho", "ha",
}

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    nx = None  # type: ignore


async def _keyword_entity_fallback(
    repo,
    graph_name: str,
    namespace: str,
    index_name: str,
    entity_names: List[str],
) -> Tuple[List[int], Dict[int, str]]:
    """CONTAINS-based fallback when exact entity name match returns no seeds.

    Splits entity_names into individual keywords (min 4 chars, no stop words),
    then queries FalkorDB for LEntity nodes whose name contains any keyword.
    Returns (fids, fid→name) for the matched entities.
    """
    keywords: List[str] = []
    seen_kw: set = set()
    for name in entity_names:
        for word in name.split():
            word = word.strip().lower()
            if len(word) >= 4 and word not in _IT_STOP_WORDS and word not in seen_kw:
                seen_kw.add(word)
                keywords.append(word)

    if not keywords:
        return [], {}

    logger.debug(f"[lgraph/ppr] keyword fallback — keywords: {keywords}")

    # openCypher: any(kw IN $kws WHERE e.name CONTAINS kw)
    kw_q = """
    MATCH (e:LEntity)
    WHERE e.namespace = $ns AND e.index_name = $idx
      AND any(kw IN $kws WHERE e.name CONTAINS kw)
    RETURN id(e) AS id, e.name AS name
    LIMIT 50
    """
    try:
        rows = await repo._execute_query(
            kw_q,
            {"ns": namespace, "idx": index_name, "kws": keywords},
            graph_name=graph_name,
        )
    except Exception as e:
        logger.warning(f"[lgraph/ppr] keyword fallback query failed: {e}")
        return [], {}

    fids = [int(r["id"]) for r in rows]
    names = {int(r["id"]): r["name"] for r in rows}
    logger.debug(f"[lgraph/ppr] keyword fallback matched {len(fids)} entities: {list(names.values())[:10]}")
    return fids, names


async def _load_subgraph(
    repo,
    graph_name: str,
    namespace: str,
    index_name: str,
    seed_entity_names: List[str],
    seed_chunk_ids: List[str],
) -> Tuple[Any, Dict[str, Dict], Set[str]]:
    """Load the neighbourhood of seed nodes from FalkorDB into NetworkX.

    Returns
    -------
    G : nx.Graph
    chunk_data : falkor_node_id → {chunk_id, text, metadata_id, source}
    effective_seed_node_ids : set of nx node IDs (e.g. "entity::albo pretorio")
        that carry weight 1.0 in the PPR personalization vector.
        Includes both exact-match seeds and keyword-fallback seeds.
    """
    if not NX_AVAILABLE:
        raise ImportError("networkx is required. Install with: pip install networkx")

    # ---- Resolve seed entity nodes ----------------------------------------
    entity_q = """
    MATCH (e:LEntity)
    WHERE e.namespace = $ns AND e.index_name = $idx AND e.name IN $names
    RETURN id(e) AS id, e.name AS name
    """
    entity_rows = await repo._execute_query(
        entity_q,
        {"ns": namespace, "idx": index_name, "names": seed_entity_names},
        graph_name=graph_name,
    )
    seed_entity_fids = [int(r["id"]) for r in entity_rows]
    entity_name_by_fid = {int(r["id"]): r["name"] for r in entity_rows}

    # ---- Keyword fallback: when exact match finds nothing ------------------
    fallback_entity_names: Dict[int, str] = {}
    if not seed_entity_fids and seed_entity_names:
        fb_fids, fb_names = await _keyword_entity_fallback(
            repo, graph_name, namespace, index_name, seed_entity_names
        )
        seed_entity_fids = fb_fids
        entity_name_by_fid.update(fb_names)
        fallback_entity_names = fb_names

    # ---- Resolve seed chunk nodes ------------------------------------------
    chunk_q = """
    MATCH (c:LChunk)
    WHERE c.namespace = $ns AND c.index_name = $idx AND c.chunk_id IN $cids
    RETURN id(c) AS id, c.chunk_id AS chunk_id, c.text AS text,
           c.metadata_id AS metadata_id, c.source AS source
    """
    chunk_rows = await repo._execute_query(
        chunk_q,
        {"ns": namespace, "idx": index_name, "cids": seed_chunk_ids},
        graph_name=graph_name,
    )
    seed_chunk_fids = [int(r["id"]) for r in chunk_rows]
    chunk_data_by_fid = {
        int(r["id"]): {
            "chunk_id": r["chunk_id"],
            "text": r.get("text") or "",
            "metadata_id": r.get("metadata_id") or "",
            "source": r.get("source") or "",
        }
        for r in chunk_rows
    }

    all_seed_fids = list(set(seed_entity_fids + seed_chunk_fids))
    if not all_seed_fids:
        return nx.Graph(), {}, set()  # type: ignore

    # ---- Fetch 1-hop neighbourhood of all seeds ---------------------------
    nb_q = """
    MATCH (s)-[r]-(n)
    WHERE id(s) IN $seed_ids
    RETURN
        id(s) AS s_id, labels(s) AS s_labels,
        id(n) AS n_id, labels(n) AS n_labels,
        type(r) AS rel_type,
        coalesce(r.tf, r.npmi, 1.0) AS weight,
        coalesce(s.chunk_id, s.name, '') AS s_key,
        coalesce(n.chunk_id, n.name, '') AS n_key,
        coalesce(n.text, '')          AS n_text,
        coalesce(n.metadata_id, '')   AS n_metadata_id,
        coalesce(n.source, '')        AS n_source
    """
    nb_rows = await repo._execute_query(
        nb_q, {"seed_ids": all_seed_fids}, graph_name=graph_name
    )

    G = nx.Graph()  # type: ignore

    def _add_chunk_node(fid: int, key: str, text: str, metadata_id: str, source: str):
        node_id = f"chunk::{key}"
        chunk_data_by_fid[fid] = {
            "chunk_id": key,
            "text": text,
            "metadata_id": metadata_id,
            "source": source,
        }
        G.add_node(node_id, ntype="chunk", fid=fid, chunk_id=key,
                   text=text, metadata_id=metadata_id, source=source)
        return node_id

    def _add_entity_node(fid: int, key: str):
        node_id = f"entity::{key}"
        G.add_node(node_id, ntype="entity", fid=fid, name=key)
        return node_id

    # Pre-add known seed nodes
    for r in chunk_rows:
        fid = int(r["id"])
        _add_chunk_node(fid, r["chunk_id"], r.get("text") or "",
                        r.get("metadata_id") or "", r.get("source") or "")
    for r in entity_rows:
        fid = int(r["id"])
        _add_entity_node(fid, r["name"])
    # Pre-add fallback entity nodes (keyword-matched, not in entity_rows)
    for fid, name in fallback_entity_names.items():
        node_id = f"entity::{name}"
        if node_id not in G:
            _add_entity_node(fid, name)

    for row in nb_rows:
        s_fid = int(row["s_id"])
        n_fid = int(row["n_id"])
        s_labels = row.get("s_labels") or []
        n_labels = row.get("n_labels") or []
        s_key = row.get("s_key") or str(s_fid)
        n_key = row.get("n_key") or str(n_fid)
        weight = float(row.get("weight") or 1.0)

        s_is_chunk = "LChunk" in s_labels
        n_is_chunk = "LChunk" in n_labels

        s_nid = (f"chunk::{s_key}" if s_is_chunk else f"entity::{s_key}")
        n_nid = (f"chunk::{n_key}" if n_is_chunk else f"entity::{n_key}")

        if s_nid not in G:
            if s_is_chunk:
                _add_chunk_node(s_fid, s_key, "", "", "")
            else:
                _add_entity_node(s_fid, s_key)

        if n_nid not in G:
            if n_is_chunk:
                _add_chunk_node(n_fid, n_key,
                                row.get("n_text") or "",
                                row.get("n_metadata_id") or "",
                                row.get("n_source") or "")
            else:
                _add_entity_node(n_fid, n_key)

        G.add_edge(s_nid, n_nid, weight=weight)

    # Build the set of seed node IDs for personalization (exact + fallback)
    effective_seed_node_ids: Set[str] = set()
    for fid in seed_chunk_fids:
        d = chunk_data_by_fid.get(fid)
        if d:
            effective_seed_node_ids.add(f"chunk::{d['chunk_id']}")
    for fid, name in entity_name_by_fid.items():
        effective_seed_node_ids.add(f"entity::{name}")

    return G, chunk_data_by_fid, effective_seed_node_ids


async def ppr_search(
    repo,
    namespace: str,
    index_name: str,
    seed_chunk_ids: List[str],
    seed_entity_names: List[str],
    top_k: int,
    alpha: float,
    max_iter: int,
    graph_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run PPR from seeds and return top_k chunk nodes ranked by score.

    Parameters
    ----------
    seed_chunk_ids    : chunk IDs (from a prior dense/sparse search)
    seed_entity_names : normalized entity names extracted from query
    """
    from .graph_builder import make_graph_name as _make_name

    if not NX_AVAILABLE:
        raise ImportError("networkx is required. Install with: pip install networkx")

    gname = graph_name or _make_name(namespace, index_name)
    G, chunk_data, effective_seed_node_ids = await _load_subgraph(
        repo, gname, namespace, index_name, seed_entity_names, seed_chunk_ids
    )

    if G.number_of_nodes() == 0:
        logger.warning(f"PPR subgraph is empty for namespace='{namespace}'")
        return []

    # Build personalization vector using effective seeds (exact + keyword fallback)
    personalization: Dict[str, float] = {}
    for node in G.nodes():
        personalization[node] = 1.0 if node in effective_seed_node_ids else 0.0

    total_weight = sum(personalization.values())
    if total_weight == 0:
        # No seeds found in graph — fall back to uniform PPR
        personalization = None  # type: ignore
    else:
        personalization = {k: v / total_weight for k, v in personalization.items()}

    try:
        scores = nx.pagerank(  # type: ignore
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            weight="weight",
        )
    except Exception as e:
        logger.warning(f"PPR convergence issue: {e} — using uniform scores")
        n = G.number_of_nodes()
        scores = {node: 1.0 / n for node in G.nodes()}

    results: List[Dict[str, Any]] = []
    for node, score in scores.items():
        if not node.startswith("chunk::"):
            continue
        ndata = G.nodes[node]
        results.append(
            {
                "chunk_id": ndata.get("chunk_id", ""),
                "text": ndata.get("text", ""),
                "metadata_id": ndata.get("metadata_id", ""),
                "source": ndata.get("source", ""),
                "ppr_score": score,
            }
        )

    results.sort(key=lambda x: x["ppr_score"], reverse=True)
    return results[:top_k]
