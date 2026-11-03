"""
type=crawl: start from `source`, follow same-host links breadth first up to
max_depth levels and max_pages fetched pages. Runs against a real local HTTP
server; every hop still goes through the SSRF guard.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tilellm.models.llm import DEFAULT_BROWSER_USER_AGENT
from tilellm.shared.outbound_url import BlockedURLError
from tilellm.tools import web_crawler
from tilellm.tools.web_crawler import crawl_site

FILLER = "Questa pagina descrive i servizi offerti dal comune ai cittadini residenti. " * 6


def _page(title, *links):
    anchors = "".join(f'<a href="{href}">{href}</a> ' for href in links)
    return (f"<html><head><title>{title}</title></head><body><nav>{anchors}</nav>"
            f"<article><h1>{title}</h1><p>{title}: {FILLER}</p></article></body></html>")


@pytest.fixture
def site(monkeypatch):
    hits = []
    pages = {
        "/": _page("Home", "/a", "/b", "/a#top", "http://other.invalid/x", "/archivio.zip",
                   "mailto:info@example.org", "/out", "/meta", "/old.php", "http://localhost:PORT/d", "/moved"),
        "/a": _page("A", "/a1", "/"),
        "/a1": _page("A1", "/a2"),
        "/a2": _page("A2", "/a3"),
        "/a3": _page("A3"),
        "/b": _page("B"),
        "/c": _page("C"),
        "/d": _page("D"),
        "/old.php": _page("Old"),
        "/robots.txt": "User-agent: *\nDisallow: /b\nDisallow: /*.php$\n",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            if self.path == "/out":  # same port, different host name
                return self._redirect(f"http://localhost:{self.server.server_port}/c")
            if self.path == "/meta":
                return self._redirect("http://169.254.169.254/latest/meta-data/")
            if self.path == "/moved":  # same page under its canonical URL
                return self._redirect("/a")
            body = pages.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            ctype = "text/plain" if self.path.endswith(".txt") else "text/html; charset=utf-8"
            data = body.replace("PORT", str(self.server.server_port)).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location):
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "127.0.0.1,localhost")
    yield f"http://127.0.0.1:{server.server_port}", hits
    server.shutdown()


def _paths(docs, base):
    return sorted({d.metadata["source"].removeprefix(base) for d in docs})


@pytest.mark.asyncio
async def test_follows_links_up_to_max_depth(site):
    base, hits = site
    docs = await crawl_site(base + "/", max_depth=2, max_pages=50)
    # /b is excluded by robots.txt, /a2 is three levels down
    assert _paths(docs, base) == ["/", "/a", "/a1"]
    assert "/a2" not in hits


@pytest.mark.asyncio
async def test_stops_at_max_pages(site):
    base, hits = site
    docs = await crawl_site(base + "/", max_depth=5, max_pages=2)
    assert _paths(docs, base) == ["/", "/a"]
    assert [h for h in hits if h != "/robots.txt"] == ["/", "/a"]


@pytest.mark.asyncio
async def test_each_page_fetched_once(site):
    base, hits = site
    await crawl_site(base + "/", max_depth=5, max_pages=50)
    assert hits.count("/a") == 1 and hits.count("/") == 1


@pytest.mark.asyncio
async def test_only_same_host_html_links(site):
    base, hits = site
    docs = await crawl_site(base + "/", max_depth=1, max_pages=50)
    assert "/archivio.zip" not in hits
    assert "/d" not in hits  # linked from home, but on another host name
    assert all(d.metadata["source"].startswith(base) for d in docs)


@pytest.mark.asyncio
async def test_redirect_to_another_host_is_dropped(site):
    base, hits = site
    docs = await crawl_site(base + "/", max_depth=1, max_pages=50)
    assert "/out" in hits
    assert "/c" not in hits  # the redirect target is never requested
    assert not any(d.metadata["source"].endswith("/c") for d in docs)


@pytest.mark.asyncio
async def test_start_page_may_move_to_another_host(site):
    base, hits = site
    # example.org -> www.example.org: the crawl follows the start page and stays there
    docs = await crawl_site(base + "/out", max_depth=0, max_pages=5)
    assert [d.metadata["source"] for d in docs] == [base.replace("127.0.0.1", "localhost") + "/c"]


@pytest.mark.asyncio
async def test_redirect_is_a_link_checked_for_duplicates(site):
    base, hits = site
    await crawl_site(base + "/", max_depth=1, max_pages=50)
    assert "/moved" in hits and hits.count("/a") == 1


@pytest.mark.asyncio
async def test_same_host_on_another_port_is_not_followed(monkeypatch):
    # with a private start host this would be a port scan of that host
    _, fetched = _fake_site(monkeypatch, _page("Home", "https://www.example.org:6379/x", "/y"))
    await crawl_site("https://www.example.org/", max_depth=1, max_pages=10, ignore_robots=True)
    urls = [u for u, _ in fetched]
    assert "https://www.example.org/y" in urls and not any(":6379" in u for u in urls)


@pytest.mark.asyncio
async def test_redirect_to_metadata_is_blocked(site):
    base, hits = site
    docs = await crawl_site(base + "/", max_depth=1, max_pages=50)
    assert "/meta" in hits
    assert not any("169.254" in d.metadata["source"] for d in docs)


@pytest.mark.asyncio
async def test_robots_txt_respected_unless_ignored(site):
    base, hits = site
    await crawl_site(base + "/", max_depth=1, max_pages=50)
    assert "/b" not in hits
    assert "/old.php" not in hits  # wildcard rule: urllib.robotparser would let it through
    docs = await crawl_site(base + "/", max_depth=1, max_pages=50, ignore_robots=True)
    assert "/b" in _paths(docs, base)


@pytest.mark.asyncio
async def test_page_title_becomes_file_name(site):
    base, _ = site
    docs = await crawl_site(base + "/", max_depth=0, max_pages=50)
    assert {d.metadata["file_name"] for d in docs} == {"Home"}


@pytest.mark.asyncio
async def test_start_url_must_pass_the_guard(monkeypatch):
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    with pytest.raises(BlockedURLError):
        await crawl_site("http://169.254.169.254/", max_depth=1, max_pages=5)


@pytest.mark.asyncio
async def test_empty_static_page_is_rendered_with_the_browser(site, monkeypatch):
    base, _ = site
    rendered = []

    async def fake_render(browser, url, headers, origin):
        rendered.append(url)
        return _page("Rendered"), url

    monkeypatch.setattr(web_crawler, "_render_html", fake_render)
    monkeypatch.setattr(web_crawler, "_fetch_html", lambda url, headers: ("<html><body><div id=app></div></body></html>", None, None))
    docs = await crawl_site(base + "/", max_depth=0, max_pages=5, ignore_robots=True)
    assert rendered == [base + "/"]
    assert docs and docs[0].metadata["file_name"] == "Rendered"


def _fake_site(monkeypatch, static_html, rendered_html=None):
    rendered, fetched = [], []
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "www.example.org")

    async def fake_render(browser, url, headers, origin):
        rendered.append(url)
        return (_page("Rendered", "/next") if rendered_html is None else rendered_html), url

    def fake_fetch(url, headers):
        fetched.append((url, headers))
        return static_html, None, None

    monkeypatch.setattr(web_crawler, "_render_html", fake_render)
    monkeypatch.setattr(web_crawler, "_fetch_html", fake_fetch)
    return rendered, fetched


@pytest.mark.asyncio
async def test_page_with_links_but_no_text_is_rendered(monkeypatch):
    # JS app whose static HTML only has the menu
    rendered, _ = _fake_site(monkeypatch, '<html><body><a href="/menu">menu</a><div id=app></div></body></html>')
    docs = await crawl_site("https://www.example.org/", max_depth=0, max_pages=1, ignore_robots=True)
    assert rendered == ["https://www.example.org/"]
    assert docs[0].metadata["file_name"] == "Rendered"


@pytest.mark.asyncio
async def test_page_with_text_but_no_links_is_rendered_when_links_are_needed(monkeypatch):
    # JS app whose static HTML has the text but builds the navigation in the browser
    rendered, fetched = _fake_site(monkeypatch, _page("Static"))
    await crawl_site("https://www.example.org/", max_depth=1, max_pages=5, ignore_robots=True)
    assert rendered[0] == "https://www.example.org/"
    assert "https://www.example.org/next" in [u for u, _ in fetched]


@pytest.mark.asyncio
async def test_page_with_text_at_the_last_level_is_not_rendered(monkeypatch):
    rendered, _ = _fake_site(monkeypatch, _page("Static"))
    docs = await crawl_site("https://www.example.org/", max_depth=0, max_pages=5, ignore_robots=True)
    assert rendered == [] and docs[0].metadata["file_name"] == "Static"


@pytest.mark.asyncio
async def test_http_and_https_of_the_same_page_are_fetched_once(monkeypatch):
    _, fetched = _fake_site(monkeypatch, _page("Home", "http://www.example.org/a", "https://www.example.org/a"))
    await crawl_site("https://www.example.org/", max_depth=1, max_pages=10, ignore_robots=True)
    assert [u for u, _ in fetched if u.endswith("/a")] == ["http://www.example.org/a"]


@pytest.mark.asyncio
async def test_static_text_seen_on_another_page_is_an_app_shell(monkeypatch):
    # every URL of a JS app serves the same static HTML: render instead of indexing it again
    shell = _page("Shell", "/a", "/b")
    rendered, _ = _fake_site(monkeypatch, shell)
    await crawl_site("https://www.example.org/", max_depth=1, max_pages=3, ignore_robots=True)
    assert rendered == ["https://www.example.org/a", "https://www.example.org/b"]


@pytest.mark.asyncio
async def test_same_text_is_indexed_once(monkeypatch):
    rendered, _ = _fake_site(monkeypatch, _page("Shell", "/a", "/b"))
    docs = await crawl_site("https://www.example.org/", max_depth=1, max_pages=3, ignore_robots=True)
    # home (Shell) + /a rendered; /b renders the same text as /a
    assert sorted(d.metadata["file_name"] for d in docs if not d.metadata.get("element_type")) == ["Rendered", "Shell"]


@pytest.mark.asyncio
async def test_browser_failure_keeps_the_static_page(monkeypatch):
    async def broken_render(browser, url, headers, origin):
        raise RuntimeError("chromium not installed")

    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "www.example.org")
    monkeypatch.setattr(web_crawler, "_render_html", broken_render)
    monkeypatch.setattr(web_crawler, "_fetch_html", lambda url, headers: (_page("Static"), None, None))
    docs = await crawl_site("https://www.example.org/", max_depth=1, max_pages=1, ignore_robots=True)
    assert docs and docs[0].metadata["file_name"] == "Static"


@pytest.mark.asyncio
async def test_empty_render_keeps_the_static_text(monkeypatch):
    _fake_site(monkeypatch, _page("Static"), rendered_html="<html><body></body></html>")
    docs = await crawl_site("https://www.example.org/", max_depth=1, max_pages=1, ignore_robots=True)
    assert docs[0].metadata["file_name"] == "Static"


CRAWLER_UA = "TiledeskBot/1.0 (+https://tiledesk.com)"


@pytest.mark.asyncio
@pytest.mark.parametrize("given,expected", [
    # CDNs answer 429 to a client that claims to be a browser and is not one:
    # the crawler says what it is, unless the caller chose a user agent
    (None, CRAWLER_UA),
    ({"user-agent": DEFAULT_BROWSER_USER_AGENT}, CRAWLER_UA),  # ItemSingle's default, not a choice
    ({"Authorization": "Bearer FINTO"}, CRAWLER_UA),
    ({"User-Agent": "Bot-Finto/1.0"}, "Bot-Finto/1.0"),
])
async def test_user_agent_default_and_any_case(monkeypatch, given, expected):
    _, fetched = _fake_site(monkeypatch, _page("Static"))
    await crawl_site("https://www.example.org/", max_depth=0, max_pages=1, ignore_robots=True,
                     browser_headers=given)
    sent = fetched[0][1]
    assert [v for k, v in sent.items() if k.lower() == "user-agent"] == [expected]


@pytest.mark.asyncio
async def test_mostly_failed_crawl_is_an_error(monkeypatch):
    # the site is stored under one id: a crawl that lost most pages must not replace it silently
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "www.example.org")

    def flaky(url, headers):
        if url.endswith("/"):
            return _page("Home", "/a", "/b", "/c"), None, None
        raise ConnectionError("timeout")

    monkeypatch.setattr(web_crawler, "_fetch_html", flaky)
    with pytest.raises(ValueError, match="3 of 4 pages failed"):
        await crawl_site("https://www.example.org/", max_depth=1, max_pages=10, ignore_robots=True)


def test_robots_server_error_disallows_everything(monkeypatch):
    class _Resp:
        status_code = 503
        text = ""

    monkeypatch.setattr(web_crawler, "outbound_requests_get", lambda *a, **k: _Resp())
    rules = web_crawler._robots("https://www.example.org/", {})
    assert not rules.can_fetch("https://www.example.org/a", "bot")


# --- browser: guard, no service workers, headers only to the start origin ---

class _FakeRoute:
    def __init__(self, url):
        self.request = type("R", (), {"url": url, "headers": {"accept": "*/*"}})()
        self.fallback_headers = "not called"

    async def fallback(self, headers=None):
        self.fallback_headers = headers


@pytest.mark.asyncio
async def test_headers_go_only_to_the_start_origin():
    handler = web_crawler._header_route(("www.example.org", None), {"Authorization": "Bearer FINTO"})
    same, other = _FakeRoute("https://www.example.org/x.js"), _FakeRoute("https://tracker.example.net/p.gif")
    await handler(same)
    await handler(other)
    assert same.fallback_headers["Authorization"] == "Bearer FINTO"
    assert other.fallback_headers is None


@pytest.mark.asyncio
async def test_browser_page_is_guarded_and_blocks_service_workers():
    from unittest.mock import AsyncMock, MagicMock

    page = MagicMock()
    page.route, page.goto, page.close, page.wait_for_timeout = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
    page.content = AsyncMock(return_value="<html></html>")
    page.url = "https://www.example.org/"
    browser = web_crawler._Browser()
    browser._browser = MagicMock(new_page=AsyncMock(return_value=page))

    await browser.render("https://www.example.org/", {"user-agent": "UA"}, ("www.example.org", None))

    assert browser._browser.new_page.call_args.kwargs["service_workers"] == "block"
    assert page.route.await_count == 2  # guard_page + headers
    assert page.close.await_count == 1


@pytest.mark.asyncio
async def test_page_that_navigates_by_itself_is_read_after_it_settles():
    # JS apps redirect client-side right after load: page.content() raises meanwhile
    from unittest.mock import AsyncMock, MagicMock

    page = MagicMock()
    page.route, page.goto, page.close, page.wait_for_load_state = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.content = AsyncMock(side_effect=[RuntimeError("page is navigating"), "<html>ok</html>"])
    page.url = "https://www.example.org/home"
    browser = web_crawler._Browser()
    browser._browser = MagicMock(new_page=AsyncMock(return_value=page))

    html, url = await browser.render("https://www.example.org/", {}, ("www.example.org", None))

    assert (html, url) == ("<html>ok</html>", "https://www.example.org/home")
    page.wait_for_load_state.assert_awaited()


@pytest.mark.asyncio
async def test_summary_says_why_pages_were_not_indexed(site, caplog):
    import logging
    crawler_log = logging.getLogger("tilellm.tools.web_crawler")  # may not propagate to root
    crawler_log.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="tilellm.tools.web_crawler")
    base, _ = site
    try:
        await crawl_site(base + "/", max_depth=1, max_pages=50)
    finally:
        crawler_log.removeHandler(caplog.handler)
    summary = [r.getMessage() for r in caplog.records if "indexed" in r.getMessage()][-1]
    # / and /a indexed; /b and /old.php excluded by robots; /moved redirects to /a (same host);
    # /out redirects to another host; /meta redirects to the metadata address: blocked = failed
    for part in ("indexed=2", "robots=2", "redirect=1", "other_host=1", "failed=1"):
        assert part in summary, summary


@pytest.mark.asyncio
@pytest.mark.parametrize("ua,expected", [
    (CRAWLER_UA, None),  # Chromium keeps its own user agent: JS apps check it against the real browser
    ("Bot-Finto/1.0", "Bot-Finto/1.0"),  # a caller's choice applies to the browser too
])
async def test_browser_user_agent(ua, expected):
    from unittest.mock import AsyncMock, MagicMock

    page = MagicMock()
    page.route, page.goto, page.close, page.wait_for_timeout = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()
    page.content = AsyncMock(return_value="<html></html>")
    page.url = "https://www.example.org/"
    browser = web_crawler._Browser()
    browser._browser = MagicMock(new_page=AsyncMock(return_value=page))

    await browser.render("https://www.example.org/", {"user-agent": ua}, ("www.example.org", None))

    assert browser._browser.new_page.call_args.kwargs.get("user_agent") == expected


# --- discover_site: the same walk, a list of URLs instead of documents ---

def _listed(result, base):
    return [(p["url"].removeprefix(base), p["kind"], p["depth"]) for p in result["pages"]]


@pytest.mark.asyncio
async def test_discover_lists_what_crawl_would_index(site):
    from tilellm.tools.web_crawler import discover_site
    base, hits = site
    result = await discover_site(base + "/", max_depth=2, max_pages=50)
    assert _listed(result, base) == [("/", "page", 0), ("/a", "page", 1), ("/a1", "page", 2)]
    assert [p["title"] for p in result["pages"]] == ["Home", "A", "A1"]
    assert result["truncated"] is False
    assert result["skipped"]["robots"] == 2 and result["skipped"]["other_host"] == 1
    assert "/a2" not in hits


@pytest.mark.asyncio
async def test_discover_says_when_max_pages_cut_the_list(site):
    from tilellm.tools.web_crawler import discover_site
    base, _ = site
    result = await discover_site(base + "/", max_depth=5, max_pages=2)
    assert _listed(result, base) == [("/", "page", 0), ("/a", "page", 1)]
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_discover_start_url_must_pass_the_guard(monkeypatch):
    from tilellm.tools.web_crawler import discover_site
    monkeypatch.delenv("OUTBOUND_URL_ALLOWED_HOSTS", raising=False)
    with pytest.raises(BlockedURLError):
        await discover_site("http://169.254.169.254/", max_depth=1, max_pages=5)


@pytest.mark.asyncio
async def test_discover_renders_js_apps_like_crawl(monkeypatch):
    from tilellm.tools.web_crawler import discover_site
    rendered, _ = _fake_site(monkeypatch, '<html><body><div id=app></div></body></html>')
    result = await discover_site("https://www.example.org/", max_depth=0, max_pages=1, ignore_robots=True)
    assert rendered == ["https://www.example.org/"]
    assert result["pages"][0]["title"] == "Rendered"


@pytest.mark.asyncio
async def test_discover_reports_failures_instead_of_raising(monkeypatch):
    # nothing is stored: a mostly failed walk is still a useful list
    from tilellm.tools.web_crawler import discover_site
    monkeypatch.setenv("OUTBOUND_URL_ALLOWED_HOSTS", "www.example.org")

    def flaky(url, headers):
        if url.endswith("/"):
            return _page("Home", "/a", "/b", "/c"), None, None
        raise ConnectionError("timeout")

    monkeypatch.setattr(web_crawler, "_fetch_html", flaky)
    result = await discover_site("https://www.example.org/", max_depth=1, max_pages=10, ignore_robots=True)
    assert [p["url"] for p in result["pages"]] == ["https://www.example.org/"]
    assert result["skipped"]["failed"] == 3


@pytest.mark.asyncio
async def test_discover_stops_at_max_seconds(site, monkeypatch):
    from tilellm.tools import web_crawler
    from tilellm.tools.web_crawler import discover_site
    base, _ = site
    clock = iter(range(0, 10_000, 20))  # every clock call advances 20 s
    monkeypatch.setattr(web_crawler, "_clock", lambda: next(clock))
    result = await discover_site(base + "/", max_depth=5, max_pages=50, max_seconds=30)
    assert result["truncated"] is True
    assert len(result["pages"]) < 4


@pytest.mark.asyncio
async def test_no_browser_render_once_the_time_limit_has_passed(monkeypatch):
    # the static page needs a render (text, no links, links wanted) but its fetch ended past the limit
    from tilellm.tools.web_crawler import discover_site
    now = [0.0]
    monkeypatch.setattr(web_crawler, "_clock", lambda: now[0])
    rendered, _ = _fake_site(monkeypatch, _page("Static"))

    def late_fetch(url, headers):
        now[0] = 31.0
        return _page("Static"), None, None

    monkeypatch.setattr(web_crawler, "_fetch_html", late_fetch)
    result = await discover_site("https://www.example.org/", max_depth=1, max_pages=5, ignore_robots=True,
                                 max_seconds=30)
    assert rendered == []
    assert [p["title"] for p in result["pages"]] == ["Static"]


@pytest.mark.asyncio
async def test_robots_txt_counts_against_the_time_limit(monkeypatch):
    from tilellm.tools.web_crawler import discover_site
    now = [0.0]
    monkeypatch.setattr(web_crawler, "_clock", lambda: now[0])
    _, fetched = _fake_site(monkeypatch, _page("Static", "/a"))

    def slow_robots(start, headers):
        now[0] = 31.0
        return None

    monkeypatch.setattr(web_crawler, "_robots", slow_robots)
    result = await discover_site("https://www.example.org/", max_depth=1, max_pages=5, max_seconds=30)
    assert fetched == [] and result["pages"] == [] and result["truncated"] is True
