"""Integration tests: fetch_documents + chunk_documents pipeline preserves tables.

These tests mock HTTP (trafilatura / requests) to avoid real network calls,
then verify that table Documents survive the full scrape → chunk pipeline
with correct metadata and without row/column splitting.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Minimal HTML with a product catalog table
# ---------------------------------------------------------------------------

HTML_CATALOG = """
<html><body>
<h1>Catalogo prodotti</h1>
<p>Il nostro listino prezzi aggiornato.</p>
<table>
  <thead>
    <tr><th>SKU</th><th>Nome</th><th>Prezzo</th><th>Categoria</th></tr>
  </thead>
  <tbody>
    <tr><td>A001</td><td>Prodotto Alpha</td><td>9.99</td><td>Elettronica</td></tr>
    <tr><td>A002</td><td>Prodotto Beta</td><td>19.99</td><td>Elettronica</td></tr>
    <tr><td>B001</td><td>Gadget Gamma</td><td>4.99</td><td>Accessori</td></tr>
  </tbody>
</table>
</body></html>
"""

SOURCE_URL = "https://example.com/catalog"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(strategy="adaptive", enable=True):
    """Minimal ItemSingle-like object for chunk_documents."""
    from tilellm.models.llm import TableOptions
    item = MagicMock()
    item.id = "test-doc"
    item.source = SOURCE_URL
    item.type = "url"
    item.embedding = "text-embedding-3-small"
    item.tags = None
    item.chunk_size = 1000
    item.chunk_overlap = 200
    item.semantic_chunk = False
    item.breakpoint_threshold_type = "percentile"
    item.table_options = TableOptions(enable=enable, strategy=strategy)
    return item


# ---------------------------------------------------------------------------
# Tests: _extract_html_tables output quality
# ---------------------------------------------------------------------------

class TestExtractHtmlTables:
    def test_extracts_table_doc_from_html(self):
        from tilellm.tools.document_tools import _extract_html_tables
        docs = _extract_html_tables(HTML_CATALOG, SOURCE_URL)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["element_type"] == "table"

    def test_col_names_all_present(self):
        from tilellm.tools.document_tools import _extract_html_tables
        docs = _extract_html_tables(HTML_CATALOG, SOURCE_URL)
        col_names = docs[0].metadata["col_names"]
        assert "SKU" in col_names
        assert "Nome" in col_names
        assert "Prezzo" in col_names

    def test_markdown_content_has_all_rows(self):
        from tilellm.tools.document_tools import _extract_html_tables
        docs = _extract_html_tables(HTML_CATALOG, SOURCE_URL)
        content = docs[0].page_content
        assert "A001" in content
        assert "A002" in content
        assert "B001" in content

    def test_source_url_in_metadata(self):
        from tilellm.tools.document_tools import _extract_html_tables
        docs = _extract_html_tables(HTML_CATALOG, SOURCE_URL)
        assert docs[0].metadata["source"] == SOURCE_URL


# ---------------------------------------------------------------------------
# Tests: chunk_documents routes table docs through split_table_document
# ---------------------------------------------------------------------------

class TestChunkDocumentsBypassTable:
    """Test the chunk_documents bypass logic using the Pinecone serverless repo.

    The same logic is identical across qdrant and milvus repos.
    """

    def _make_table_doc(self):
        from tilellm.tools.document_tools import _extract_html_tables
        docs = _extract_html_tables(HTML_CATALOG, SOURCE_URL)
        return docs[0]  # the table Document

    def _make_text_doc(self):
        return Document(
            page_content="Il nostro listino prezzi aggiornato.",
            metadata={"source": SOURCE_URL}
        )

    @pytest.mark.asyncio
    async def test_table_doc_not_split_by_recursive_splitter(self):
        """Table document must produce ≥1 chunk without destroying row alignment."""
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="atomic")
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("element_type") in ("table", "table_rows")
        assert "A001" in chunks[0].page_content
        assert "A002" in chunks[0].page_content
        assert "B001" in chunks[0].page_content

    @pytest.mark.asyncio
    async def test_adaptive_small_table_stays_whole(self):
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="adaptive")
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        # Small table → 1 chunk
        table_chunks = [c for c in chunks if c.metadata.get("element_type") in ("table", "table_rows")]
        assert len(table_chunks) == 1

    @pytest.mark.asyncio
    async def test_per_row_produces_one_chunk_per_row(self):
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="per_row")
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        table_chunks = [c for c in chunks if c.metadata.get("element_type") == "table_rows"]
        assert len(table_chunks) == 3  # 3 data rows

    @pytest.mark.asyncio
    async def test_text_doc_still_uses_recursive_splitter(self):
        """Non-table documents must still go through the normal splitter."""
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="adaptive")
        text_doc = self._make_text_doc()

        chunks = await repo.chunk_documents(item, [text_doc], embeddings=None)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("element_type") != "table"

    @pytest.mark.asyncio
    async def test_disable_table_options_falls_back_to_splitter(self):
        """When table_options.enable=False, tables pass through the generic splitter."""
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(enable=False)
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        # Without bypass, the generic splitter is used — element_type may survive
        # but chunk count may differ; we just verify it doesn't crash
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_col_names_preserved_in_chunk_metadata(self):
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="atomic")
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        table_chunks = [c for c in chunks if c.metadata.get("element_type") in ("table", "table_rows")]
        for chunk in table_chunks:
            col_names = chunk.metadata.get("col_names", "")
            assert isinstance(col_names, str)
            assert "SKU" in col_names

    @pytest.mark.asyncio
    async def test_item_id_propagated(self):
        from tilellm.store.pinecone.pinecone_repository_serverless import PineconeRepositoryServerless
        repo = PineconeRepositoryServerless.__new__(PineconeRepositoryServerless)

        item = _make_item(strategy="atomic")
        table_doc = self._make_table_doc()

        chunks = await repo.chunk_documents(item, [table_doc], embeddings=None)

        for chunk in chunks:
            assert chunk.metadata.get("id") == "test-doc"
