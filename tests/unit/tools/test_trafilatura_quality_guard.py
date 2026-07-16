"""Tests: _handle_trafilatura_scrape retries without favor_precision when the
precision extraction is suspiciously short (e.g. only a cookie banner survives
on chrome-heavy PA portal pages)."""
import pytest
from unittest.mock import patch

from tilellm.tools.document_tools import _handle_trafilatura_scrape

URL = "https://example.org/dettagliostaff"

BANNER = (
    "Per offrire informazioni e servizi nel miglior modo possibile, questo sito "
    "utilizza cookie tecnici e cookie di terze parti. Per maggiori informazioni "
    "sui cookie utilizzati e su come eventualmente disabilitarli leggi la nostra "
    "privacy policy"
)
CONTENT = "MARIO VERDELLI Istruttore Struttura: COMUNICAZIONE ISTITUZIONALE " * 20


@pytest.mark.asyncio
async def test_short_precision_falls_back_to_default_extraction():
    def fake_extract(downloaded, **kwargs):
        return BANNER if kwargs.get("favor_precision") else CONTENT

    with patch("trafilatura.fetch_url", return_value="<html>page</html>"), \
         patch("trafilatura.extract", side_effect=fake_extract) as mock_extract:
        docs = await _handle_trafilatura_scrape(URL)

    assert mock_extract.call_count == 2
    assert docs and CONTENT.strip() == docs[0].page_content


@pytest.mark.asyncio
async def test_long_precision_result_is_kept_without_retry():
    long_precision = "Contenuto principale ricco. " * 30  # > 500 chars

    with patch("trafilatura.fetch_url", return_value="<html>page</html>"), \
         patch("trafilatura.extract", return_value=long_precision) as mock_extract:
        docs = await _handle_trafilatura_scrape(URL)

    assert mock_extract.call_count == 1
    assert docs and docs[0].page_content == long_precision.strip()


@pytest.mark.asyncio
async def test_both_extractions_short_returns_empty():
    with patch("trafilatura.fetch_url", return_value="<html>page</html>"), \
         patch("trafilatura.extract", return_value="troppo corto"):
        docs = await _handle_trafilatura_scrape(URL)

    assert docs == []
