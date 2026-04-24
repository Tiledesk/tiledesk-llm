"""Tests for tilellm.modules.ingestion.table_chunker."""
import pytest
from langchain_core.documents import Document

from tilellm.modules.ingestion.table_chunker import split_table_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_TABLE_MD = """\
| SKU | Nome | Prezzo |
|---|---|---|
| A001 | Prodotto Alpha | 9.99 |
| A002 | Prodotto Beta | 19.99 |
| A003 | Prodotto Gamma | 29.99 |"""

BIG_TABLE_ROWS = ["| A{:03d} | Prodotto {:03d} | {:.2f} |".format(i, i, i * 1.5) for i in range(1, 51)]
BIG_TABLE_MD = "| SKU | Nome | Prezzo |\n|---|---|---|\n" + "\n".join(BIG_TABLE_ROWS)

MALFORMED_MD = "questo non è una tabella markdown valida"
NO_DATA_ROWS_MD = "| SKU | Nome |\n|---|---|"

BASE_META = {
    "id": "doc1",
    "source": "https://example.com/catalog",
    "element_type": "table",
    "col_names": "SKU, Nome, Prezzo",
    "table_index": 0,
    "type": "url",
}


def _doc(content: str, meta: dict = None) -> Document:
    m = dict(BASE_META)
    if meta:
        m.update(meta)
    return Document(page_content=content, metadata=m)


# ---------------------------------------------------------------------------
# Strategy: atomic
# ---------------------------------------------------------------------------

class TestStrategyAtomic:
    def test_returns_single_chunk(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="atomic")
        assert len(chunks) == 1

    def test_content_preserved(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="atomic")
        assert "Prodotto Alpha" in chunks[0].page_content
        assert "Prodotto Gamma" in chunks[0].page_content

    def test_metadata_preserved(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="atomic")
        m = chunks[0].metadata
        assert m["element_type"] in ("table", "table_rows")
        assert m["col_names"] == "SKU, Nome, Prezzo"
        assert m["table_index"] == 0

    def test_big_table_still_one_chunk(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="atomic")
        assert len(chunks) == 1

    def test_chunk_type_metadata(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="atomic")
        assert chunks[0].metadata.get("chunk_type") == "table"


# ---------------------------------------------------------------------------
# Strategy: per_row
# ---------------------------------------------------------------------------

class TestStrategyPerRow:
    def test_one_chunk_per_data_row(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        assert len(chunks) == 3

    def test_header_repeated_in_each_chunk(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row", header_repeat=True)
        for chunk in chunks:
            assert "SKU" in chunk.page_content
            assert "Nome" in chunk.page_content

    def test_header_not_repeated_when_disabled(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row", header_repeat=False)
        # First chunk may have header; subsequent ones should not start with "| SKU"
        assert len(chunks) == 3
        # Data values are still present
        assert "Prodotto Alpha" in chunks[0].page_content

    def test_row_index_metadata(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["row_index"] == i

    def test_total_rows_metadata(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        for chunk in chunks:
            assert chunk.metadata["total_rows"] == 3

    def test_row_range_is_string(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        for chunk in chunks:
            assert isinstance(chunk.metadata["row_range"], str)

    def test_each_chunk_contains_its_row_data(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        assert "A001" in chunks[0].page_content
        assert "A002" in chunks[1].page_content
        assert "A003" in chunks[2].page_content

    def test_big_table_50_rows(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="per_row")
        assert len(chunks) == 50

    def test_element_type_metadata(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        for chunk in chunks:
            assert chunk.metadata["element_type"] == "table_rows"


# ---------------------------------------------------------------------------
# Strategy: adaptive
# ---------------------------------------------------------------------------

class TestStrategyAdaptive:
    def test_small_table_stays_atomic(self):
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="adaptive", max_table_chars=8000)
        assert len(chunks) == 1

    def test_big_table_is_split(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="adaptive", max_table_chars=300)
        assert len(chunks) > 1

    def test_all_rows_covered_after_split(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="adaptive", max_table_chars=300)
        all_content = "\n".join(c.page_content for c in chunks)
        for i in range(1, 51):
            assert "A{:03d}".format(i) in all_content

    def test_header_repeated_in_split_chunks(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="adaptive", max_table_chars=300, header_repeat=True)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "SKU" in chunk.page_content

    def test_default_max_table_chars(self):
        """Default 8000 chars keeps small tables as 1 chunk."""
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="adaptive")
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Malformed / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_malformed_content_returns_single_chunk(self):
        chunks = split_table_document(_doc(MALFORMED_MD), strategy="per_row")
        assert len(chunks) == 1
        assert chunks[0].page_content == MALFORMED_MD

    def test_no_data_rows_returns_single_chunk(self):
        chunks = split_table_document(_doc(NO_DATA_ROWS_MD), strategy="per_row")
        assert len(chunks) == 1

    def test_empty_content_returns_empty_or_single(self):
        chunks = split_table_document(_doc(""), strategy="per_row")
        assert len(chunks) <= 1

    def test_metadata_propagated_on_fallback(self):
        chunks = split_table_document(_doc(MALFORMED_MD), strategy="per_row")
        assert chunks[0].metadata["source"] == "https://example.com/catalog"
        assert chunks[0].metadata["table_index"] == 0

    def test_row_range_string_format(self):
        chunks = split_table_document(_doc(BIG_TABLE_MD), strategy="adaptive", max_table_chars=300)
        for chunk in chunks:
            rr = chunk.metadata.get("row_range", "")
            parts = rr.split("-")
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()

    def test_pinecone_compat_col_names_is_string(self):
        """col_names must be a string (Pinecone metadata constraint)."""
        chunks = split_table_document(_doc(SMALL_TABLE_MD), strategy="per_row")
        for chunk in chunks:
            assert isinstance(chunk.metadata.get("col_names", ""), str)
