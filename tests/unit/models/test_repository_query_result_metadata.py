#!/usr/bin/env python3
"""
RepositoryQueryResult (id/metadata_id/metadata_source/metadata_type/date/text) is a
FIXED schema used by get_all_obj_namespace across all backend repos — structurally
unable to carry custom fields like page_number/doc_type. A generic `metadata` dict
passthrough (mirroring RetrievalChunksResult.metadata: List[dict], already used by
/api/qa and compliance_checker) unblocks callers like lgraph's build_lgraph.
"""
from tilellm.models.schemas import RepositoryQueryResult


class TestRepositoryQueryResultMetadata:
    def test_metadata_defaults_none(self):
        r = RepositoryQueryResult(id="c1")
        assert r.metadata is None

    def test_metadata_accepts_arbitrary_dict(self):
        raw = {"page_number": 7, "doc_type": "delibera", "custom_key": "x"}
        r = RepositoryQueryResult(id="c1", metadata=raw)
        assert r.metadata == raw
