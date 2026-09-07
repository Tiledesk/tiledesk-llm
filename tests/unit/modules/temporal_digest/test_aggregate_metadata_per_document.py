"""
_aggregate_metadata counted act_type/amount per *chunk*, not per document.
A document ingested at ~30 chunks/doc average (this corpus) with the same
act_type on every chunk (situated_context runs per chunk but usually agrees
with itself within one document) got counted ~30x in the act_type
distribution, and a repeated amount across overlapping chunks got summed
~30x too — directly caused the wrong "distribuzione act_type" answer in
docs/ASL_BARI_DEMO_RESULTS.md (#8).
"""
from tilellm.modules.temporal_digest.services.digest_service import (
    _aggregate_metadata,
    _dedupe_by_document,
)


def _chunk(doc_id, act_type="LIQUIDAZIONE", amount=None):
    m = {"metadata_id": doc_id}
    if act_type:
        m["act_type"] = act_type
    if amount is not None:
        m["amount"] = amount
    return m


class TestDedupeByDocument:
    def test_collapses_multiple_chunks_to_one_entry_per_document(self):
        metadatas = [_chunk("deter-1"), _chunk("deter-1"), _chunk("deter-1")]
        result = _dedupe_by_document(metadatas)
        assert len(result) == 1

    def test_keeps_distinct_documents_separate(self):
        metadatas = [_chunk("deter-1"), _chunk("deter-2")]
        result = _dedupe_by_document(metadatas)
        assert len(result) == 2


class TestAggregateMetadataPerDocument:
    def test_act_type_counted_once_per_document_not_per_chunk(self):
        """30 chunks of the same LIQUIDAZIONE document -> count 1, not 30."""
        metadatas = [_chunk("deter-1", act_type="LIQUIDAZIONE") for _ in range(30)]
        act_types, _ = _aggregate_metadata(metadatas)
        assert act_types == {"LIQUIDAZIONE": 1}

    def test_distribution_across_distinct_documents(self):
        metadatas = (
            [_chunk("deter-1", act_type="LIQUIDAZIONE") for _ in range(10)]
            + [_chunk("deter-2", act_type="ALTRO") for _ in range(20)]
        )
        act_types, _ = _aggregate_metadata(metadatas)
        assert act_types == {"LIQUIDAZIONE": 1, "ALTRO": 1}

    def test_amount_not_double_counted_across_overlapping_chunks(self):
        """Same total mentioned in 2 overlapping chunks of the same document
        (chunk_overlap=400 in the real splitter) must count once, not twice."""
        metadatas = [
            _chunk("deter-1", amount=1000.0),
            _chunk("deter-1", amount=1000.0),
        ]
        _, total = _aggregate_metadata(metadatas)
        assert total == 1000.0

    def test_amounts_summed_across_distinct_documents(self):
        metadatas = [_chunk("deter-1", amount=1000.0), _chunk("deter-2", amount=500.0)]
        _, total = _aggregate_metadata(metadatas)
        assert total == 1500.0
