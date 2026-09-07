"""
Real quality gap found on the ASL Bari corpus (2026-08-08, see
docs/ASL_BARI_DEMO_RESULTS.md): a busy day (54 real determine) produces
~1.760 chunks (54 × ~32.7 chunks/document measured average); the digest's
evidence block dumped raw chunk text (900 chars/chunk) and truncated at
_MAX_EVIDENCE_CHARS after ~155 chunks, mid-document, in arbitrary order —
only 2 of 54 acts ended up "clearly readable" in the resulting digest.

The fix: one evidence line per *document* (grouped by metadata_id), built
from metadata already extracted once at ingestion (numero_determina, data,
oggetto, act_type, amount, CIG, CUP) — not from raw chunk text. A document's
full picture doesn't need 30 chunks' worth of text; the facts a digest needs
are already sitting in additional_metadata + situated_context's direct
fields. Same char budget now covers hundreds of documents instead of ~5.
"""
from tilellm.modules.temporal_digest.services.digest_service import _build_document_evidence_block


def _meta(doc_id, **over):
    base = {
        "metadata_id": doc_id, "numero_determina": doc_id.split("-")[-1],
        "data_determina": "27/07/2026", "oggetto": "Liquidazione fattura fornitore X",
        "act_type": "LIQUIDAZIONE", "amount": 1000.0, "cig": None, "cup": None,
    }
    base.update(over)
    return base


class TestBuildDocumentEvidenceBlock:
    def test_one_line_per_document_not_per_chunk(self):
        """3 chunks from the same document -> 1 evidence line, not 3."""
        chunks = ["testo chunk 1", "testo chunk 2", "testo chunk 3"]
        metadatas = [_meta("deter-1"), _meta("deter-1"), _meta("deter-1")]

        result = _build_document_evidence_block(chunks, metadatas)

        assert result.count("deter-1") <= 1 or result.count("[1]") == 1
        assert len(result.strip().split("\n")) == 1

    def test_covers_all_distinct_documents(self):
        """54 documents, 1 chunk each -> 54 lines, none dropped (the real bug:
        the old raw-chunk block dropped 52/54 to a budget truncation)."""
        chunks = [f"testo {i}" for i in range(54)]
        metadatas = [_meta(f"deter-{i}") for i in range(54)]

        result = _build_document_evidence_block(chunks, metadatas)

        assert len(result.strip().split("\n")) == 54

    def test_includes_key_metadata_fields(self):
        chunks = ["testo"]
        metadatas = [_meta("deter-42", oggetto="Acquisto dispositivi medici", amount=5000.0, cig="B1EFBDA301")]

        result = _build_document_evidence_block(chunks, metadatas)

        assert "42" in result
        assert "Acquisto dispositivi medici" in result
        assert "5,000.00" in result  # same €{:,.2f} formatting as _build_evidence_block
        assert "B1EFBDA301" in result

    def test_documents_missing_all_metadata_still_get_a_line(self):
        """No additional_metadata at all -> falls back to doc_id, not dropped."""
        chunks = ["testo grezzo senza metadati strutturati"]
        metadatas = [{"metadata_id": "deter-99"}]

        result = _build_document_evidence_block(chunks, metadatas)

        assert "deter-99" in result

    def test_respects_char_budget_and_reports_omitted_count(self):
        """When even the compact per-document form overflows the budget, the
        omission message must state how many were dropped (same contract as
        the raw-chunk _build_evidence_block)."""
        import tilellm.modules.temporal_digest.services.digest_service as svc
        original = svc._MAX_EVIDENCE_CHARS
        svc._MAX_EVIDENCE_CHARS = 100
        try:
            chunks = [f"testo {i}" for i in range(20)]
            metadatas = [_meta(f"deter-{i}", oggetto="Oggetto piuttosto lungo " * 3) for i in range(20)]
            result = svc._build_document_evidence_block(chunks, metadatas)
        finally:
            svc._MAX_EVIDENCE_CHARS = original

        assert "omessi per limiti di contesto" in result
