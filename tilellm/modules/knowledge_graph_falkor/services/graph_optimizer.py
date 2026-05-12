"""
Graph Optimizer Service.

Pipeline:
  1. Export nodes + relationships from FalkorDB → Parquet bytes (via DuckDB)
  2. Batch-embed entity names+descriptions with the configured LLM embedder (TEI)
  3. Find near-duplicate pairs with SimSIMD cosine similarity
  4. Build an optimized merge plan entirely in DuckDB (no pandas):
       - canonical node keeps merged source_ids from all duplicates
       - relationships redirected to canonical, self-loops removed, duplicates collapsed
  5. Save optimized snapshot to MinIO
  6. Wipe FalkorDB graph and reimport from snapshot (community reports preserved)
"""

import io
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import simsimd
    SIMSIMD_AVAILABLE = True
except ImportError:
    SIMSIMD_AVAILABLE = False

logger = logging.getLogger(__name__)

# Chunked pairwise search: rows processed per SimSIMD call to cap peak memory.
# At 768-dim float32: 2000 × 33000 × 4 bytes ≈ 240 MB per chunk — acceptable.
_SIMSIMD_CHUNK = 2000


class GraphOptimizer:
    def __init__(self, repository, minio_storage_service, llm_embeddings=None):
        self.repository = repository
        self.minio = minio_storage_service
        self.llm_embeddings = llm_embeddings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        namespace: str,
        graph_name: str,
        similarity_threshold: float = 0.92,
        embedding_batch_size: int = 256,
        dry_run: bool = False,
        engine=None,
        vector_store_repo=None,
    ) -> Dict[str, Any]:
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required for graph optimization")
        if not SIMSIMD_AVAILABLE:
            raise ImportError("simsimd is required for graph optimization")
        if self.llm_embeddings is None:
            raise RuntimeError("llm_embeddings must be provided for graph optimization")

        logger.info(f"GraphOptimizer.run: graph='{graph_name}', threshold={similarity_threshold}, dry_run={dry_run}")

        # 1. Export from FalkorDB → in-memory Parquet bytes
        nodes_bytes, rels_bytes = await self._export_to_parquet(namespace, graph_name)

        # 2. Load Parquet into pandas (avoids DuckDB StringDtype incompatibility on pandas 2.0)
        import pandas as pd
        _nodes_df = pd.read_parquet(io.BytesIO(nodes_bytes))
        _rels_df = pd.read_parquet(io.BytesIO(rels_bytes))
        node_count = len(_nodes_df)
        rel_count = len(_rels_df)
        logger.info(f"Exported: {node_count} nodes, {rel_count} relationships")

        if node_count == 0:
            return {"status": "empty", "nodes_before": 0, "nodes_after": 0,
                    "nodes_merged": 0, "relationships_before": 0,
                    "relationships_after": 0, "merge_pairs": 0, "dry_run": dry_run}

        # 3. Compute embeddings
        node_ids = [str(v) for v in _nodes_df["id"]]
        texts = [
            f"{str(row.get('name') or '')}: {str(row.get('description') or '')}"[:512]
            for row in _nodes_df.to_dict("records")
        ]

        logger.info(f"Computing embeddings for {len(texts)} entities (batch={embedding_batch_size})...")
        embeddings = await self._batch_embed(texts, embedding_batch_size)

        # 4. Find duplicate pairs via SimSIMD
        merge_pairs = self._find_merge_pairs(embeddings, node_ids, similarity_threshold)
        logger.info(f"Found {len(merge_pairs)} merge pairs (threshold={similarity_threshold})")

        if dry_run:
            return {
                "status": "dry_run",
                "nodes_before": node_count,
                "nodes_after": node_count - len(set(dup for _, dup in merge_pairs)),
                "nodes_merged": len(set(dup for _, dup in merge_pairs)),
                "relationships_before": rel_count,
                "relationships_after": rel_count,
                "merge_pairs": len(merge_pairs),
                "dry_run": True,
            }

        if not merge_pairs:
            snapshot = self.minio.save_graph_snapshot(graph_name, nodes_bytes, rels_bytes)
            return {
                "status": "no_duplicates",
                "nodes_before": node_count, "nodes_after": node_count,
                "nodes_merged": 0, "relationships_before": rel_count,
                "relationships_after": rel_count, "merge_pairs": 0,
                "dry_run": False, "snapshot_timestamp": snapshot["timestamp"],
            }

        # 5. Build optimized graph
        opt_nodes_bytes, opt_rels_bytes, stats = self._apply_merge(merge_pairs, nodes_bytes, rels_bytes)

        # 6. Save optimized snapshot to MinIO
        snapshot = self.minio.save_graph_snapshot(graph_name, opt_nodes_bytes, opt_rels_bytes)
        logger.info(f"Optimized snapshot saved: {snapshot['timestamp']}")

        # 7. Load community reports (preserve them)
        community_reports = self.minio.load_community_reports(graph_name)

        # 8. Wipe FalkorDB + reimport
        await self._reimport(
            namespace=namespace,
            graph_name=graph_name,
            nodes_bytes=opt_nodes_bytes,
            rels_bytes=opt_rels_bytes,
            community_reports=community_reports,
        )

        return {
            "status": "success",
            "nodes_before": node_count,
            "nodes_after": stats["nodes_after"],
            "nodes_merged": stats["nodes_merged"],
            "relationships_before": rel_count,
            "relationships_after": stats["rels_after"],
            "merge_pairs": len(merge_pairs),
            "dry_run": False,
            "snapshot_timestamp": snapshot["timestamp"],
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    async def _export_to_parquet(self, namespace: str, graph_name: str) -> Tuple[bytes, bytes]:
        """Fetch full graph from FalkorDB and serialise to Parquet bytes via DuckDB."""
        import pandas as pd

        graph_data = await self.repository.get_all_nodes_and_relationships(
            namespace=namespace, graph_name=graph_name
        )
        nodes = graph_data["nodes"]
        rels = graph_data["relationships"]

        def _to_parquet(records: List[Dict]) -> bytes:
            if not records:
                return pd.DataFrame().to_parquet()
            # Flatten list fields to JSON strings for Parquet compatibility
            import json
            flat = []
            for r in records:
                row = dict(r)
                for k, v in row.items():
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v)
                flat.append(row)
            buf = io.BytesIO()
            pd.DataFrame(flat).to_parquet(buf, index=False)
            return buf.getvalue()

        return _to_parquet(nodes), _to_parquet(rels)

    # ------------------------------------------------------------------
    # DuckDB helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def _batch_embed(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed texts in batches using the configured LLM embedder."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = await self.llm_embeddings.aembed_documents(batch)
            all_embeddings.extend(vecs)
            if (i // batch_size) % 10 == 0:
                logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} entities")
        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-normalise so cosine similarity = dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    # ------------------------------------------------------------------
    # SimSIMD similarity search
    # ------------------------------------------------------------------

    def _find_merge_pairs(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        threshold: float,
    ) -> List[Tuple[str, str]]:
        """
        Find near-duplicate pairs using SimSIMD inner-product (= cosine on normalised vectors).
        Uses Union-Find to build transitive merge groups; each group gets one canonical node
        (the one that appears first in the original order = most likely the oldest).
        Returns a list of (canonical_id, duplicate_id) pairs.
        """
        n = len(embeddings)
        distance_threshold = 1.0 - threshold  # simsimd cdist returns distances

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        logger.info(f"Running SimSIMD pairwise search ({n} × {n}, chunk={_SIMSIMD_CHUNK})...")
        for chunk_start in range(0, n, _SIMSIMD_CHUNK):
            chunk = embeddings[chunk_start: chunk_start + _SIMSIMD_CHUNK]
            # Returns distance matrix (n_chunk × n)
            dists = np.array(simsimd.cdist(chunk, embeddings, metric="cosine"), dtype=np.float32)
            rows_idx, cols_idx = np.where(dists < distance_threshold)
            for r, c in zip(rows_idx, cols_idx):
                global_r = chunk_start + int(r)
                global_c = int(c)
                if global_r != global_c:
                    union(global_r, global_c)

        # Build merge pairs from Union-Find groups
        from collections import defaultdict
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        pairs = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            canonical_idx = min(members)  # first = canonical
            for m in members:
                if m != canonical_idx:
                    pairs.append((node_ids[canonical_idx], node_ids[m]))

        return pairs

    # ------------------------------------------------------------------
    # DuckDB merge plan
    # ------------------------------------------------------------------

    def _apply_merge(
        self,
        merge_pairs: List[Tuple[str, str]],
        nodes_bytes: bytes,
        rels_bytes: bytes,
    ) -> Tuple[bytes, bytes, Dict[str, int]]:
        """
        Apply the merge plan in pandas.

        Canonical nodes: all fields from the canonical node, with source_ids merged
        from all duplicates (union of JSON arrays).
        Relationships: source/target redirected to canonical; self-loops and exact
        duplicates (same type, source, target) removed.
        """
        import pandas as pd
        import json

        merge_df = pd.DataFrame(merge_pairs, columns=["canonical_id", "duplicate_id"])

        # ---- Nodes: build optimised set ----
        # For each duplicate, collect its source_ids into the canonical node.
        # We do this in Python since DuckDB list operations on JSON strings need care.
        nodes_df = pd.read_parquet(io.BytesIO(nodes_bytes))

        dup_ids = set(merge_df["duplicate_id"].tolist())
        canonical_ids = set(merge_df["canonical_id"].tolist())

        # Build per-canonical maps: source_ids to merge + best description ("longer wins").
        # Future improvement: "longer wins + append unique sentences" (token-overlap dedup
        # to preserve genuinely complementary information without description bloat).
        extra_sources: Dict[str, List[str]] = {}
        best_description: Dict[str, str] = {}  # canonical_id → longest description seen
        id_to_idx = {str(row["id"]): idx for idx, row in nodes_df.iterrows()}

        for _, pair in merge_df.iterrows():
            can_id, dup_id = pair["canonical_id"], pair["duplicate_id"]
            if dup_id not in id_to_idx:
                continue
            dup_row = nodes_df.iloc[id_to_idx[dup_id]]

            # source_ids accumulation
            src = dup_row.get("source_ids", "[]")
            try:
                src_list = json.loads(src) if isinstance(src, str) else (src or [])
            except Exception:
                src_list = []
            extra_sources.setdefault(can_id, []).extend(src_list)

            # description: keep the longer one between canonical and duplicate
            dup_desc = str(dup_row.get("description") or "")
            current_best = best_description.get(can_id)
            if current_best is None:
                # seed with canonical's own description
                if can_id in id_to_idx:
                    can_desc = str(nodes_df.iloc[id_to_idx[can_id]].get("description") or "")
                    current_best = can_desc
                else:
                    current_best = ""
            if len(dup_desc) > len(current_best):
                best_description[can_id] = dup_desc
            else:
                best_description[can_id] = current_best

        def merge_source_ids(row) -> str:
            node_id = str(row["id"])
            existing = row.get("source_ids", "[]")
            try:
                base = json.loads(existing) if isinstance(existing, str) else (existing or [])
            except Exception:
                base = []
            extra = extra_sources.get(node_id, [])
            merged = list(dict.fromkeys(base + extra))  # deduplicate, preserve order
            return json.dumps(merged)

        opt_nodes = nodes_df[~nodes_df["id"].astype(str).isin(dup_ids)].copy()
        opt_nodes["source_ids"] = opt_nodes.apply(merge_source_ids, axis=1)

        # Apply "longer wins" description to canonical nodes that absorbed duplicates
        if "description" in opt_nodes.columns and best_description:
            def apply_best_description(row):
                node_id = str(row["id"])
                return best_description.get(node_id, row["description"])
            opt_nodes["description"] = opt_nodes.apply(apply_best_description, axis=1)

        # ---- Relationships: redirect + deduplicate ----
        rels_df = pd.read_parquet(io.BytesIO(rels_bytes))
        can_map = dict(zip(merge_df["duplicate_id"], merge_df["canonical_id"]))

        def remap(node_id: str) -> str:
            return can_map.get(str(node_id), str(node_id))

        rels_df["source_id"] = rels_df["source_id"].astype(str).map(remap)
        rels_df["target_id"] = rels_df["target_id"].astype(str).map(remap)

        # Remove self-loops and deduplicate (same type + source + target)
        rels_df = rels_df[rels_df["source_id"] != rels_df["target_id"]]
        rels_df = rels_df.drop_duplicates(subset=["type", "source_id", "target_id"])

        # Serialise back to Parquet
        buf_nodes = io.BytesIO()
        opt_nodes.to_parquet(buf_nodes, index=False)

        buf_rels = io.BytesIO()
        rels_df.to_parquet(buf_rels, index=False)

        stats = {
            "nodes_after": len(opt_nodes),
            "nodes_merged": len(dup_ids),
            "rels_after": len(rels_df),
        }
        logger.info(f"Merge applied: {stats}")
        return buf_nodes.getvalue(), buf_rels.getvalue(), stats

    # ------------------------------------------------------------------
    # Reimport
    # ------------------------------------------------------------------

    async def _reimport(
        self,
        namespace: str,
        graph_name: str,
        nodes_bytes: bytes,
        rels_bytes: bytes,
        community_reports: List[Dict[str, Any]],
    ):
        """
        Wipe FalkorDB graph and reimport from optimised Parquet.
        Community reports are re-saved to FalkorDB as-is (no re-clustering needed).
        """
        import pandas as pd
        import json

        logger.info(f"Reimporting optimised graph into FalkorDB (graph='{graph_name}')...")

        # Wipe existing nodes for this namespace
        await self.repository.delete_all_nodes(namespace=namespace, graph_name=graph_name)

        # ---- Reimport nodes ----
        nodes_df = pd.read_parquet(io.BytesIO(nodes_bytes))
        entity_node_map: Dict[str, str] = {}

        for _, row in nodes_df.iterrows():
            props = row.to_dict()
            label = props.pop("label", "ENTITY")
            node_id_original = str(props.pop("id", ""))
            # Deserialise source_ids
            src = props.get("source_ids", "[]")
            try:
                props["source_ids"] = json.loads(src) if isinstance(src, str) else (src or [])
            except Exception:
                props["source_ids"] = []

            from tilellm.modules.knowledge_graph.models import Node
            node = Node(label=label, properties=props)
            created = await self.repository.create_node(
                node, namespace=namespace, graph_name=graph_name
            )
            if created.id and props.get("name"):
                entity_node_map[self.repository._normalize_name(props["name"])] = created.id

        logger.info(f"Reimported {len(nodes_df)} nodes")

        # ---- Reimport relationships ----
        rels_df = pd.read_parquet(io.BytesIO(rels_bytes))
        # Build old_id → new_id map using names (entity_node_map already has new IDs)
        # The rels reference old FalkorDB node IDs; we need to resolve via names.
        # Since we remapped in _apply_merge using canonical IDs, and the nodes Parquet
        # still carries the original IDs, build a direct old_id → new_falkor_id map.
        old_to_new: Dict[str, str] = {}
        for _, row in nodes_df.iterrows():
            old_to_new[str(row["id"])] = entity_node_map.get(
                self.repository._normalize_name(str(row.get("name", "")))
            ) or ""

        rels_created = 0
        from tilellm.modules.knowledge_graph.models import Relationship, RelationshipProperties
        for _, row in rels_df.iterrows():
            src_new = old_to_new.get(str(row["source_id"]))
            tgt_new = old_to_new.get(str(row["target_id"]))
            if not src_new or not tgt_new:
                continue
            try:
                props = row.to_dict()
                for drop_col in ("id", "source_id", "target_id", "type"):
                    props.pop(drop_col, None)
                # Deserialise source_ids if present
                src_ids = props.get("source_ids", "[]")
                try:
                    props["source_ids"] = json.loads(src_ids) if isinstance(src_ids, str) else (src_ids or [])
                except Exception:
                    props["source_ids"] = []
                rel = Relationship(
                    source_id=src_new,
                    target_id=tgt_new,
                    type=str(row.get("type", "RELATED_TO")),
                    properties=props,
                )
                await self.repository.create_relationship(rel, namespace=namespace, graph_name=graph_name)
                rels_created += 1
            except Exception as e:
                logger.warning(f"Skipping relationship during reimport: {e}")

        logger.info(f"Reimported {rels_created} relationships")

        # ---- Restore community reports ----
        for report in community_reports:
            try:
                await self.repository.save_community_report(
                    community_id=report.get("community_id", ""),
                    report=report,
                    level=int(report.get("level", 0)),
                    namespace=namespace,
                    graph_name=graph_name,
                    index_name=None,
                    engine_name=None,
                    engine_type=None,
                    metadata_id=None,
                )
            except Exception as e:
                logger.warning(f"Failed to restore community report {report.get('community_id')}: {e}")

        logger.info(f"Restored {len(community_reports)} community reports")
