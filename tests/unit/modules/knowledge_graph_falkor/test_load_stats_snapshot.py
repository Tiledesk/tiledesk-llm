#!/usr/bin/env python3
"""load_stats_snapshot reads community_graph_service._save_stats's MinIO
layout ({index_name}/{index_type}/{namespace}/{timestamp}/*.parquet) — a
different path/schema from save_graph_snapshot's own graph_snapshots/."""
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from minio.error import S3Error


def _service_with_mock_client():
    # Minio is only referenced in the shared base class (_initialize, inherited
    # unchanged) since the 2026-08-02 dedup refactor — this subclass no longer
    # imports it itself.
    with patch("tilellm.shared.minio_storage.Minio"):
        from tilellm.modules.knowledge_graph_falkor.services.minio_storage import MinIOStorageService
        svc = MinIOStorageService.__new__(MinIOStorageService)
        svc._client = MagicMock()
        svc.bucket_name = "graphrag"
        return svc


def _obj(name):
    o = MagicMock()
    o.object_name = name
    return o


def _parquet_bytes(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


class TestLoadStatsSnapshot:
    def test_loads_explicit_timestamp(self):
        svc = _service_with_mock_client()

        def get_object(bucket, key):
            resp = MagicMock()
            if key.endswith("entities.parquet"):
                resp.read.return_value = _parquet_bytes([{"node_id": "1", "name": "x"}])
            elif key.endswith("relationships.parquet"):
                resp.read.return_value = _parquet_bytes([{"relationship_id": "1"}])
            elif key.endswith("community_reports.parquet"):
                resp.read.return_value = _parquet_bytes([{"community_id": "L0_C1", "summary": "s"}])
            return resp

        svc._client.get_object.side_effect = get_object

        result = svc.load_stats_snapshot(namespace="256237", index_name="debt-recovery-hybrid", index_type="local", timestamp="20260729_151509")

        assert result["timestamp"] == "20260729_151509"
        assert pd.read_parquet(io.BytesIO(result["entities"])).iloc[0]["name"] == "x"
        assert result["community_reports"][0]["community_id"] == "L0_C1"

    def test_resolves_latest_timestamp_when_none_given(self):
        svc = _service_with_mock_client()
        prefix = "debt-recovery-hybrid/local/256237/"
        svc._client.list_objects.return_value = [
            _obj(f"{prefix}20260729_100000/entities.parquet"),
            _obj(f"{prefix}20260729_151509/entities.parquet"),
        ]

        def get_object(bucket, key):
            resp = MagicMock()
            resp.read.return_value = _parquet_bytes([{"node_id": "1"}])
            return resp
        svc._client.get_object.side_effect = get_object

        result = svc.load_stats_snapshot(namespace="256237", index_name="debt-recovery-hybrid", index_type="local")

        assert result["timestamp"] == "20260729_151509"

    def test_missing_snapshot_raises_file_not_found(self):
        svc = _service_with_mock_client()
        svc._client.list_objects.return_value = []

        with pytest.raises(FileNotFoundError):
            svc.load_stats_snapshot(namespace="ghost", index_name="idx", index_type="local")

    def test_missing_community_reports_defaults_to_empty_list(self):
        svc = _service_with_mock_client()

        def get_object(bucket, key):
            if key.endswith("community_reports.parquet"):
                raise S3Error(response=MagicMock(), code="NoSuchKey", message="not found",
                               resource=key, request_id="req", host_id="host")
            resp = MagicMock()
            resp.read.return_value = _parquet_bytes([{"node_id": "1"}])
            return resp
        svc._client.get_object.side_effect = get_object

        result = svc.load_stats_snapshot(namespace="256237", index_name="debt-recovery-hybrid", index_type="local", timestamp="20260729_151509")

        assert result["community_reports"] == []
