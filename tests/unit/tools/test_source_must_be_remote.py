"""
`source` / `file_content` come from the request body. A value that is not an
http(s) URL used to be opened from the container filesystem by the loaders
(PyPDFLoader, CSVLoader, StructuredDocxLoader...) or by Playwright (`file://`):
the content was then indexed and readable back through /api/qa.
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest

from tilellm.tools.document_tools import get_content_by_url, load_document, require_remote_url

LOCAL_SOURCES = ["/etc/hosts", "file:///etc/hosts", "../.env", "C:/secrets.csv", ""]


@pytest.mark.parametrize("source", LOCAL_SOURCES)
@pytest.mark.parametrize("type_source", ["pdf", "docx", "csv", "xlsx", "txt", "md"])
def test_load_document_rejects_non_http_sources_before_opening_anything(source, type_source):
    with patch("tilellm.tools.document_tools.CSVLoader") as csv, \
         patch("tilellm.tools.document_tools.PyPDFLoader") as pdf:
        with pytest.raises(ValueError, match="http"):
            load_document(source, type_source)
    csv.assert_not_called()
    pdf.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("scrape_type", [0, 1, 3])
async def test_scraper_rejects_file_urls(scrape_type):
    with pytest.raises(ValueError, match="http"):
        await get_content_by_url("file:///etc/hosts", scrape_type)


@pytest.mark.asyncio
async def test_docx_ingestion_rejects_local_paths():
    from tilellm.modules.ingestion import docx_processor

    raw = inspect.unwrap(docx_processor.process_docx_with_images)
    question = MagicMock(engine=MagicMock(), id="d1", namespace="ns", file_name=None,
                         file_content="/etc/hosts", source=None)
    with patch.object(docx_processor, "StructuredDocxLoader") as loader:
        with pytest.raises(ValueError, match="http"):
            await raw(question, repo=MagicMock())
    loader.assert_not_called()


@pytest.mark.parametrize("url", ["http://files.example.com/a.pdf", "HTTPS://x.example.com/b.docx?sig=1"])
def test_http_urls_are_accepted(url):
    assert require_remote_url(url) == url
