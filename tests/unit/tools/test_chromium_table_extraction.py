"""Tests: handle_chromium_loader extracts HTML tables for scrape_type 3/4."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from langchain_core.documents import Document

# Minimal HTML page with a product table
HTML_WITH_TABLE = """
<html><body>
<p>Catalogo prodotti</p>
<table>
  <thead><tr><th>SKU</th><th>Nome</th><th>Prezzo</th></tr></thead>
  <tbody>
    <tr><td>A001</td><td>Prodotto Alpha</td><td>9.99</td></tr>
    <tr><td>A002</td><td>Prodotto Beta</td><td>19.99</td></tr>
  </tbody>
</table>
</body></html>
"""

HTML_NO_TABLE = "<html><body><p>Solo testo, nessuna tabella.</p></body></html>"


def _make_mock_page(html_content: str):
    page = AsyncMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value=html_content)
    return page


def _make_mock_browser(page):
    browser = AsyncMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()
    return browser


def _make_playwright_ctx(browser):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=ctx)
    ctx.__aexit__ = AsyncMock(return_value=False)
    ctx.chromium = MagicMock()
    ctx.chromium.launch = AsyncMock(return_value=browser)
    return ctx


@pytest.fixture
def mock_playwright_with_table(monkeypatch):
    page = _make_mock_page(HTML_WITH_TABLE)
    browser = _make_mock_browser(page)
    ctx = _make_playwright_ctx(browser)

    monkeypatch.setattr(
        "tilellm.tools.document_tools.async_playwright",
        MagicMock(return_value=ctx),
    )
    return ctx


@pytest.fixture
def mock_playwright_no_table(monkeypatch):
    page = _make_mock_page(HTML_NO_TABLE)
    browser = _make_mock_browser(page)
    ctx = _make_playwright_ctx(browser)

    monkeypatch.setattr(
        "tilellm.tools.document_tools.async_playwright",
        MagicMock(return_value=ctx),
    )
    return ctx


class TestChromiumLoaderTableExtraction:
    @pytest.mark.asyncio
    async def test_table_docs_appended_after_transformer(self, mock_playwright_with_table):
        from langchain_community.document_transformers import Html2TextTransformer
        from tilellm.tools.document_tools import handle_chromium_loader

        docs = await handle_chromium_loader(
            urls=["https://example.com/catalog"],
            transformer=Html2TextTransformer(),
        )

        element_types = [d.metadata.get("element_type") for d in docs]
        assert "table" in element_types, f"No table doc found; element_types={element_types}"

    @pytest.mark.asyncio
    async def test_table_doc_has_col_names(self, mock_playwright_with_table):
        from langchain_community.document_transformers import Html2TextTransformer
        from tilellm.tools.document_tools import handle_chromium_loader

        docs = await handle_chromium_loader(
            urls=["https://example.com/catalog"],
            transformer=Html2TextTransformer(),
        )

        table_docs = [d for d in docs if d.metadata.get("element_type") == "table"]
        assert table_docs, "No table documents returned"
        assert "SKU" in table_docs[0].metadata.get("col_names", "")

    @pytest.mark.asyncio
    async def test_table_doc_contains_markdown(self, mock_playwright_with_table):
        from langchain_community.document_transformers import Html2TextTransformer
        from tilellm.tools.document_tools import handle_chromium_loader

        docs = await handle_chromium_loader(
            urls=["https://example.com/catalog"],
            transformer=Html2TextTransformer(),
        )

        table_docs = [d for d in docs if d.metadata.get("element_type") == "table"]
        assert table_docs
        content = table_docs[0].page_content
        assert "Prodotto Alpha" in content or "A001" in content

    @pytest.mark.asyncio
    async def test_no_table_in_html_returns_no_table_docs(self, mock_playwright_no_table):
        from langchain_community.document_transformers import Html2TextTransformer
        from tilellm.tools.document_tools import handle_chromium_loader

        docs = await handle_chromium_loader(
            urls=["https://example.com/no-table"],
            transformer=Html2TextTransformer(),
        )

        table_docs = [d for d in docs if d.metadata.get("element_type") == "table"]
        assert table_docs == []

    @pytest.mark.asyncio
    async def test_text_docs_still_returned_alongside_table(self, mock_playwright_with_table):
        from langchain_community.document_transformers import Html2TextTransformer
        from tilellm.tools.document_tools import handle_chromium_loader

        docs = await handle_chromium_loader(
            urls=["https://example.com/catalog"],
            transformer=Html2TextTransformer(),
        )

        text_docs = [d for d in docs if d.metadata.get("element_type") != "table"]
        assert text_docs, "Text content docs should still be present"
