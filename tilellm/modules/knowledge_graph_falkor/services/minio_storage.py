"""
MinIO Storage Service for GraphRAG artifacts — graph-specific extension.

The base upload/download/list/delete operations (used identically by every
other MinIO caller in the codebase — pdf_ocr, ingestion/docx_processor, ...)
live in tilellm/shared/minio_storage.py and are inherited unchanged here.
Verified byte-for-byte identical before this refactor (2026-08-02): 17 methods
in common, none diverged except this module's delete_artifacts wording, now
unified on the base class's generic message (delete_artifacts has always had
zero callers — see the policy below). This module adds only the
falkor-specific artifact types built on top: extraction checkpoints,
community reports, and full graph snapshots.

Graceful degradation is unchanged by this refactor: it lives in the callers
(e.g. community_graph_service.__init__ wraps get_minio_storage_service() in
try/except and falls back to minio_storage_service=None), not in this class.
The constructor still raises ImportError/ValueError when minio isn't
installed/configured, and .client still raises RuntimeError when the server
is unreachable — callers that need MinIO to be optional must keep catching
that, exactly as before.

Snapshot durability policy
--------------------------
**The parquet snapshots written here are the system's only restore point, and
nothing may delete them implicitly.**

Why (incident, 2026-07-30): a duplicate TaskIQ delivery re-ran a destructive
`overwrite=True` graph build on namespace 256237 and wiped 10 908 freshly
extracted nodes out of FalkorDB. What made that recoverable at zero cost was
the snapshot the successful build had already written here: POST
/api/kg-falkor/reimport rebuilt the graph straight from parquet, with no LLM
calls. Had those files been cleaned up "because the run failed", the only way
back would have been a ~2 h RunPod extraction. A graph build succeeding is
exactly when its snapshot becomes precious — a later failure must never take it
down with it.

Consequences, deliberately accepted:

* Snapshots are **append-only**. Each run writes a fresh timestamped folder
  (`{index_name}/{index_type}/{namespace}/{timestamp}/`) and never overwrites a
  previous one, so any past run stays restorable.
* `delete_artifacts` (base class) and `delete_community_reports` refuse to run
  unless the caller passes `confirm_destroy=True`. Both had **zero callers**
  when this was written: the guard exists so no future cleanup path can
  quietly turn into data loss. Deleting must be a decision, never a side effect.
* Storage grows over time. That is the intended trade-off — disk is cheaper than
  a lost extraction. If retention is ever needed it must be an explicit,
  operator-driven job, never an automatic cleanup inside a build.
* **Checkpoints are NOT covered by this policy.** `checkpoints/` holds transient
  per-window resume state that the extraction loop legitimately clears on
  `overwrite=True` and after completion; `delete_checkpoints` therefore needs no
  confirmation.
"""

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from minio.error import S3Error

from tilellm.shared.minio_storage import MinIOStorageService as _BaseMinIOStorageService

logger = logging.getLogger(__name__)

CHECKPOINTS_PREFIX = "checkpoints/"
COMMUNITY_REPORTS_PREFIX = "community_reports/"
GRAPH_SNAPSHOTS_PREFIX = "graph_snapshots/"


class MinIOStorageService(_BaseMinIOStorageService):
    """MinIOStorageService (tilellm/shared/minio_storage.py) plus the
    checkpoint/community-report/graph-snapshot operations the falkor
    GraphRAG pipeline needs on top of the generic parquet artifact storage."""

    # ==================== CHECKPOINT OPERATIONS ====================

    def save_checkpoint(
        self,
        graph_name: str,
        window_idx: int,
        entity_node_map_delta: Dict[str, str],
        chunk_ids: List[str],
    ) -> str:
        """
        Persist a per-window extraction checkpoint to MinIO.

        Stores entity_name→falkor_node_id mapping and the IDs of all chunks
        processed in this window so that a failed run can be resumed later.
        Files are named window_{n:06d}.parquet and never overwrite previous
        windows (window_idx is global, incremented across resume attempts).
        """
        import pandas as pd

        rows = [{"record_type": "chunk",  "key": cid, "value": "",    "window_idx": window_idx} for cid in chunk_ids]
        rows += [{"record_type": "entity", "key": k,   "value": v,    "window_idx": window_idx} for k, v in entity_node_map_delta.items()]

        buf = io.BytesIO()
        pd.DataFrame(rows, columns=["record_type", "key", "value", "window_idx"]).to_parquet(buf, index=False)
        data = buf.getvalue()

        object_key = f"{CHECKPOINTS_PREFIX}{graph_name}/window_{window_idx:06d}.parquet"
        stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_key,
            data=stream,
            length=len(data),
            content_type="application/octet-stream",
        )
        logger.info(
            f"Checkpoint saved: graph='{graph_name}' window={window_idx} "
            f"({len(entity_node_map_delta)} entities, {len(chunk_ids)} chunks)"
        )
        return object_key

    def load_checkpoints(
        self,
        graph_name: str,
    ):
        """
        Load all checkpoints for a graph.

        Returns a tuple (entity_node_map, processed_chunk_ids, last_window_idx):
          - entity_node_map: Dict[str, str]  entity_name → falkor_node_id (cumulative)
          - processed_chunk_ids: set[str]    chunk IDs already written to FalkorDB
          - last_window_idx: int             highest window number found (-1 if none)
        """
        import pandas as pd

        prefix = f"{CHECKPOINTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
        except S3Error as e:
            logger.warning(f"Could not list checkpoints for '{graph_name}': {e}")
            return {}, set(), -1

        if not objects:
            return {}, set(), -1

        names = sorted(obj.object_name for obj in objects if obj.object_name)
        entity_node_map: Dict[str, str] = {}
        processed_chunk_ids = set()
        last_window_idx = -1

        for name in names:
            try:
                resp = self.client.get_object(self.bucket_name, name)
                data = resp.read()
                resp.close()
                resp.release_conn()

                df = pd.read_parquet(io.BytesIO(data))
                for _, row in df.iterrows():
                    if row["record_type"] == "entity" and row["key"]:
                        entity_node_map[row["key"]] = row["value"]
                    elif row["record_type"] == "chunk" and row["key"]:
                        processed_chunk_ids.add(row["key"])
                if not df.empty:
                    last_window_idx = max(last_window_idx, int(df["window_idx"].iloc[0]))
            except Exception as e:
                logger.warning(f"Skipping corrupted checkpoint '{name}': {e}")

        logger.info(
            f"Loaded checkpoints for '{graph_name}': {len(names)} files, "
            f"{len(entity_node_map)} entities, {len(processed_chunk_ids)} chunks, "
            f"last_window={last_window_idx}"
        )
        return entity_node_map, processed_chunk_ids, last_window_idx

    def delete_checkpoints(self, graph_name: str) -> int:
        """Delete all checkpoint files for a graph (called on overwrite=True or after success)."""
        prefix = f"{CHECKPOINTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
            count = 0
            for obj in objects:
                if obj.object_name:
                    self.client.remove_object(self.bucket_name, obj.object_name)
                    count += 1
            if count:
                logger.info(f"Deleted {count} checkpoint files for graph '{graph_name}'")
            return count
        except S3Error as e:
            logger.warning(f"Failed to delete checkpoints for '{graph_name}': {e}")
            return 0

    # ==================== COMMUNITY REPORTS ====================

    def save_community_reports(self, graph_name: str, level: int, reports: List[Dict[str, Any]]) -> str:
        """
        Persist community reports for a Leiden level to Parquet on MinIO.
        Path: community_reports/{graph_name}/level_{level:02d}.parquet
        Overwrites any previously saved reports for the same level.
        """
        import pandas as pd

        df = pd.DataFrame(reports)
        # Serialize list fields to JSON strings for Parquet compatibility
        for col in ("findings", "entities"):
            if col in df.columns:
                import json
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()

        object_key = f"{COMMUNITY_REPORTS_PREFIX}{graph_name}/level_{level:02d}.parquet"
        stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_key,
            data=stream,
            length=len(data),
            content_type="application/octet-stream",
        )
        logger.info(f"Saved {len(reports)} community reports (level={level}) to '{object_key}'")
        return object_key

    def load_community_reports(self, graph_name: str) -> List[Dict[str, Any]]:
        """
        Load all community reports for a graph across all levels.
        Returns a flat list of report dicts with list fields deserialized.
        """
        import pandas as pd
        import json

        prefix = f"{COMMUNITY_REPORTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
        except S3Error as e:
            logger.warning(f"Could not list community reports for '{graph_name}': {e}")
            return []

        all_reports: List[Dict[str, Any]] = []
        for obj in sorted(objects, key=lambda o: o.object_name or ""):
            if not obj.object_name:
                continue
            try:
                resp = self.client.get_object(self.bucket_name, obj.object_name)
                data = resp.read(); resp.close(); resp.release_conn()
                df = pd.read_parquet(io.BytesIO(data))
                for col in ("findings", "entities"):
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                all_reports.extend(df.to_dict(orient="records"))
            except Exception as e:
                logger.warning(f"Skipping corrupted community report file '{obj.object_name}': {e}")

        logger.info(f"Loaded {len(all_reports)} community reports for graph '{graph_name}'")
        return all_reports

    def delete_community_reports(self, graph_name: str, confirm_destroy: bool = False) -> int:
        """
        Delete all community report files for a graph.

        PROTECTED — see the "snapshot durability" note at the top of this module.
        Community reports cost LLM calls to produce and are restored by
        /reimport, so deletion must be deliberate: pass confirm_destroy=True.
        """
        if not confirm_destroy:
            raise PermissionError(
                f"Refusing to delete community reports for graph '{graph_name}': they are part of "
                f"the restore point and cost LLM calls to regenerate. "
                f"Pass confirm_destroy=True if you really mean it."
            )

        prefix = f"{COMMUNITY_REPORTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
            count = 0
            for obj in objects:
                if obj.object_name:
                    self.client.remove_object(self.bucket_name, obj.object_name)
                    count += 1
            if count:
                logger.info(f"Deleted {count} community report files for graph '{graph_name}'")
            return count
        except S3Error as e:
            logger.warning(f"Failed to delete community reports for '{graph_name}': {e}")
            return 0

    # ==================== GRAPH SNAPSHOTS ====================

    def save_graph_snapshot(
        self,
        graph_name: str,
        nodes_data: bytes,
        rels_data: bytes,
        timestamp: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save a full graph snapshot (nodes + relationships as Parquet bytes) to MinIO.
        Path: graph_snapshots/{graph_name}/{timestamp}/nodes.parquet
              graph_snapshots/{graph_name}/{timestamp}/relationships.parquet
        Returns dict with 'nodes_key' and 'rels_key'.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        keys = {}
        for name, data in (("nodes.parquet", nodes_data), ("relationships.parquet", rels_data)):
            key = f"{GRAPH_SNAPSHOTS_PREFIX}{graph_name}/{timestamp}/{name}"
            stream = io.BytesIO(data)
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=stream,
                length=len(data),
                content_type="application/octet-stream",
            )
            keys[name.replace(".parquet", "_key")] = key

        logger.info(f"Graph snapshot saved for '{graph_name}' at timestamp={timestamp}")
        return {**keys, "timestamp": timestamp}

    def load_graph_snapshot(
        self,
        graph_name: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, bytes]:
        """
        Load nodes and relationships Parquet bytes from a graph snapshot.
        If timestamp is None, loads the most recent snapshot.
        Returns {'nodes': bytes, 'relationships': bytes, 'timestamp': str}.
        """
        if timestamp is None:
            timestamp = self._get_latest_graph_snapshot_timestamp(graph_name)
            if timestamp is None:
                raise FileNotFoundError(f"No graph snapshot found for '{graph_name}'")

        result: Dict[str, Any] = {"timestamp": timestamp}
        for key_name, file_name in (("nodes", "nodes.parquet"), ("relationships", "relationships.parquet")):
            object_key = f"{GRAPH_SNAPSHOTS_PREFIX}{graph_name}/{timestamp}/{file_name}"
            resp = self.client.get_object(self.bucket_name, object_key)
            data = resp.read(); resp.close(); resp.release_conn()
            result[key_name] = data

        logger.info(f"Graph snapshot loaded for '{graph_name}' (timestamp={timestamp})")
        return result

    def _get_latest_graph_snapshot_timestamp(self, graph_name: str) -> Optional[str]:
        prefix = f"{GRAPH_SNAPSHOTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=False))
            timestamps = []
            for obj in objects:
                if obj.object_name:
                    parts = obj.object_name.replace(prefix, "").split("/")
                    if parts[0]:
                        timestamps.append(parts[0])
            timestamps = sorted(set(timestamps), reverse=True)
            return timestamps[0] if timestamps else None
        except S3Error:
            return None

    def load_stats_snapshot(
        self,
        namespace: str,
        index_name: str,
        index_type: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a snapshot saved by community_graph_service._save_stats (every
        /create run, path {index_name}/{index_type}/{namespace}/{timestamp}/)
        — a different, independently-existing snapshot format from
        save_graph_snapshot's graph_snapshots/ (GraphOptimizer's own).
        If timestamp is None, loads the most recent one.
        Returns {'entities': bytes, 'relationships': bytes,
                 'community_reports': List[dict], 'timestamp': str}.
        """
        import pandas as pd

        prefix = f"{index_name}/{index_type}/{namespace}/"
        if timestamp is None:
            try:
                objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=False))
            except S3Error:
                objects = []
            timestamps = sorted(
                {obj.object_name.replace(prefix, "").split("/")[0]
                 for obj in objects if obj.object_name},
                reverse=True,
            )
            if not timestamps:
                raise FileNotFoundError(f"No stats snapshot found under '{prefix}'")
            timestamp = timestamps[0]

        base = f"{prefix}{timestamp}/"
        result: Dict[str, Any] = {"timestamp": timestamp}
        for key_name, file_name in (("entities", "entities.parquet"), ("relationships", "relationships.parquet")):
            resp = self.client.get_object(self.bucket_name, f"{base}{file_name}")
            result[key_name] = resp.read()
            resp.close(); resp.release_conn()

        try:
            resp = self.client.get_object(self.bucket_name, f"{base}community_reports.parquet")
            data = resp.read(); resp.close(); resp.release_conn()
            result["community_reports"] = pd.read_parquet(io.BytesIO(data)).to_dict(orient="records")
        except S3Error:
            result["community_reports"] = []

        logger.info(f"Stats snapshot loaded for namespace='{namespace}' (timestamp={timestamp})")
        return result

    def list_graph_snapshots(self, graph_name: str) -> List[str]:
        """Return list of available snapshot timestamps for a graph, newest first."""
        prefix = f"{GRAPH_SNAPSHOTS_PREFIX}{graph_name}/"
        try:
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=False))
            timestamps = sorted(
                {obj.object_name.replace(prefix, "").split("/")[0]
                 for obj in objects if obj.object_name},
                reverse=True
            )
            return timestamps
        except S3Error:
            return []


# Singleton instance (separate from the shared base-class singleton: callers
# importing from this module need the graph-specific subclass).
_minio_storage_service: Optional[MinIOStorageService] = None

def get_minio_storage_service() -> MinIOStorageService:
    """Get or create a singleton instance of the graph-specific MinIOStorageService."""
    global _minio_storage_service
    if _minio_storage_service is None:
        _minio_storage_service = MinIOStorageService()
    return _minio_storage_service
