#!/usr/bin/env python3
"""
The parquet snapshots on MinIO are the last-resort restore point and must not be
deletable by accident.

Rationale (2026-07-30): a duplicate TaskIQ delivery re-ran a destructive
`overwrite=True` graph build, wiping 10 908 freshly extracted nodes from
FalkorDB. The run was only recoverable because the snapshot written by the
successful build was still sitting in MinIO — POST /api/kg-falkor/reimport
rebuilt the graph from it without spending a single LLM call. Losing those
files would have meant re-running a ~2 h RunPod extraction.

delete_artifacts wipes {index_name}/{index_type}/{namespace}/ — every timestamp
when none is given — and delete_community_reports wipes a graph's reports. Both
had zero callers, i.e. loaded guns nobody was watching. They now refuse unless
the caller explicitly opts in, so deletion can only ever be deliberate.
"""
from unittest.mock import MagicMock, patch

import pytest


def _service():
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


class TestDeleteArtifactsIsProtected:
    def test_refuses_without_explicit_confirmation(self):
        svc = _service()
        svc._client.list_objects.return_value = [_obj("idx/local/256237/20260729_151509/entities.parquet")]

        with pytest.raises(PermissionError, match="confirm_destroy"):
            svc.delete_artifacts(namespace="256237", index_name="idx", index_type="local")

        svc._client.remove_object.assert_not_called()

    def test_deletes_when_explicitly_confirmed(self):
        svc = _service()
        svc._client.list_objects.return_value = [
            _obj("idx/local/256237/20260729_151509/entities.parquet"),
            _obj("idx/local/256237/20260729_151509/relationships.parquet"),
        ]

        deleted = svc.delete_artifacts(
            namespace="256237", index_name="idx", index_type="local", confirm_destroy=True
        )

        assert deleted == 2
        assert svc._client.remove_object.call_count == 2


class TestDeleteCommunityReportsIsProtected:
    def test_refuses_without_explicit_confirmation(self):
        svc = _service()
        svc._client.list_objects.return_value = [_obj("community_reports/g/level_0.parquet")]

        with pytest.raises(PermissionError, match="confirm_destroy"):
            svc.delete_community_reports("256237-debt_recovery")

        svc._client.remove_object.assert_not_called()

    def test_deletes_when_explicitly_confirmed(self):
        svc = _service()
        svc._client.list_objects.return_value = [_obj("community_reports/g/level_0.parquet")]

        assert svc.delete_community_reports("256237-debt_recovery", confirm_destroy=True) == 1
        svc._client.remove_object.assert_called_once()


class TestTwinSharedServiceHasTheSameGuard:
    """tilellm.shared.minio_storage.MinIOStorageService is the base class the
    falkor subclass above inherits delete_artifacts from unchanged (dedup
    refactor, 2026-08-02 — previously these were two diverged copies, which is
    exactly the trap this test now guards against regressing into: instantiate
    the base class directly and confirm the guard is still there, not just
    reachable through the subclass."""

    def _shared_service(self):
        with patch("tilellm.shared.minio_storage.Minio"):
            from tilellm.shared.minio_storage import MinIOStorageService
            svc = MinIOStorageService.__new__(MinIOStorageService)
            svc._client = MagicMock()
            svc.bucket_name = "graphrag"
            return svc

    def test_refuses_without_explicit_confirmation(self):
        svc = self._shared_service()
        svc._client.list_objects.return_value = [_obj("idx/local/ns/ts/entities.parquet")]

        with pytest.raises(PermissionError, match="confirm_destroy"):
            svc.delete_artifacts(namespace="ns", index_name="idx", index_type="local")

        svc._client.remove_object.assert_not_called()

    def test_deletes_when_explicitly_confirmed(self):
        svc = self._shared_service()
        svc._client.list_objects.return_value = [_obj("idx/local/ns/ts/entities.parquet")]

        assert svc.delete_artifacts(
            namespace="ns", index_name="idx", index_type="local", confirm_destroy=True
        ) == 1


class TestCheckpointsStayDeletable:
    """Checkpoints are transient resume state, not the restore point: the
    extraction loop legitimately clears them on overwrite and on completion."""

    def test_delete_checkpoints_needs_no_confirmation(self):
        svc = _service()
        svc._client.list_objects.return_value = [_obj("checkpoints/g/window_000000.parquet")]

        assert svc.delete_checkpoints("256237-debt_recovery") == 1
        svc._client.remove_object.assert_called_once()
