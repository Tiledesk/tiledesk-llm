"""
type=crawl also indexes linked files of the accepted types (`crawl_file_types`,
default pdf and docx). The type comes from Content-Type, not from the extension:
some sites serve PDFs as ".aspx". A file counts as a page in max_pages and cites
its own URL.
"""
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from tilellm.models import Engine
from tilellm.models.llm import ItemSingle
from tilellm.tools.web_crawler import crawl_site

PDF_CT = "application/pdf"
DOCX_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FILLER = "Questa pagina descrive le offerte di connessione in fibra per la casa e per le aziende. " * 5


def _pdf(text):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _docx(text):
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph(text)
    d.save(buf)
    return buf.getvalue()


@pytest.fixture
def site(monkeypatch):
    hits = []
    home = ("<html><head><title>Fibra</title></head><body><nav>"
            '<a href="/tariffe.aspx">tariffe</a> <a href="/guida.docx">guida</a> '
            '<a href="/foto.aspx">foto</a> <a href="/listino.xlsx">listino</a>'
            f"</nav><article><p>{FILLER}</p></article></body></html>")
    files = {
        "/": ("text/html; charset=utf-8", home.encode()),
        "/tariffe.aspx": (PDF_CT, _pdf("Tariffa base mensile ventinove euro")),
        "/guida.docx": (DOCX_CT, _docx("Guida all'attivazione del modem in cinque passi")),
        "/foto.aspx": ("image/jpeg", b"\xff\xd8\xff\xe0" + b"0" * 200),
        "/listino.xlsx": (XLSX_CT, b"PK\x03\x04" + b"0" * 200),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            ctype, body = files.get(self.path, ("text/plain", b""))
            self.send_response(200 if self.path in files else 404)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1")
    yield f"http://127.0.0.1:{server.server_port}", hits
    server.shutdown()


def _by_source(docs, base):
    out = {}
    for d in docs:
        out.setdefault(d.metadata["source"].removeprefix(base), []).append(d)
    return out


@pytest.mark.asyncio
async def test_pdf_served_as_aspx_and_docx_are_indexed_by_default(site):
    base, hits = site
    docs = _by_source(await crawl_site(base + "/", max_depth=1, max_pages=10, ignore_robots=True), base)
    assert "Tariffa base mensile" in " ".join(d.page_content for d in docs["/tariffe.aspx"])
    assert "Guida all'attivazione" in " ".join(d.page_content for d in docs["/guida.docx"])
    assert docs["/tariffe.aspx"][0].metadata["file_name"] == "tariffe.aspx"
    assert "/foto.aspx" not in docs  # an image is never indexed
    assert "/listino.xlsx" not in hits  # not an accepted type: the link is not even fetched


@pytest.mark.asyncio
async def test_no_file_types_means_html_only(site):
    base, hits = site
    docs = _by_source(await crawl_site(base + "/", max_depth=1, max_pages=10, ignore_robots=True,
                                       file_types=[]), base)
    assert list(docs) == ["/"]
    assert "/guida.docx" not in hits


@pytest.mark.asyncio
async def test_a_file_counts_as_a_page(site):
    base, hits = site
    await crawl_site(base + "/", max_depth=1, max_pages=2, ignore_robots=True)
    assert [h for h in hits if h != "/robots.txt"] == ["/", "/tariffe.aspx"]


def _item(**over):
    kw = dict(id="site-1", source="https://www.example.org/", type="crawl", namespace="ns",
              engine=Engine(name="qdrant", type="serverless", apikey="", vector_size=1024,
                            index_name="c", host="localhost", port=6333))
    kw.update(over)
    return ItemSingle(**kw)


def test_crawl_file_types_field():
    assert _item().crawl_file_types == ["pdf", "docx"]
    assert _item(crawl_file_types=["pdf", "xlsx"]).crawl_options()["file_types"] == ["pdf", "xlsx"]
    with pytest.raises(ValidationError):
        _item(crawl_file_types=["exe"])


@pytest.mark.asyncio
async def test_discover_lists_files_by_extension_without_downloading_them(site):
    from tilellm.tools.web_crawler import discover_site
    base, hits = site
    result = await discover_site(base + "/", max_depth=1, max_pages=10, ignore_robots=True)
    listed = {p["url"].removeprefix(base): p for p in result["pages"]}
    assert set(listed) == {"/", "/tariffe.aspx", "/guida.docx"}
    assert listed["/guida.docx"] == {"url": base + "/guida.docx", "title": "guida.docx",
                                     "kind": "file", "file_type": "docx", "depth": 1}
    assert "/guida.docx" not in hits  # listed from the link
    assert listed["/tariffe.aspx"]["file_type"] == "pdf"  # no extension: Content-Type decides
    assert "/listino.xlsx" not in hits


@pytest.mark.asyncio
async def test_discover_without_file_types_lists_pages_only(site):
    from tilellm.tools.web_crawler import discover_site
    base, _ = site
    result = await discover_site(base + "/", max_depth=1, max_pages=10, ignore_robots=True, file_types=[])
    assert [p["url"] for p in result["pages"]] == [base + "/"]
