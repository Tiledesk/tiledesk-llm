#!/usr/bin/env python3
"""
RetrievalChunksResult (get_chunks_from_repo's return shape) exposed `chunks`
(text) and `metadata` (dict) but never the vector store's own per-chunk id —
even though every backend already reads it onto `Document.id` before discarding
it. Needed for /api/lgraph/hybrid: a prior dense/sparse search's chunk ids become
`seed_chunk_ids` for ppr_search (which matches against LChunk.chunk_id, the same
id space populated at lgraph build time from RepositoryQueryResult.id).
"""
from tilellm.models.schemas import RetrievalChunksResult


class TestRetrievalChunksResultChunkIds:
    def test_chunk_ids_defaults_none(self):
        r = RetrievalChunksResult(success=True, namespace="ns")
        assert r.chunk_ids is None

    def test_chunk_ids_accepts_list(self):
        r = RetrievalChunksResult(success=True, namespace="ns", chunk_ids=["c1", "c2"])
        assert r.chunk_ids == ["c1", "c2"]
