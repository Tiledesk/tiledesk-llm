"""
type=crawl: start from a URL and follow same-origin links breadth first, up to
max_depth link levels (0 = the start page only) and max_pages fetched pages.

Every fetch goes through the SSRF guard. Redirects are not followed by the HTTP
client: a redirect target is queued like a link, so it is checked for origin,
robots.txt and duplicates, and request headers never leave the start origin.
Only the start page may move to another host (example.org -> www.example.org).
robots.txt is honoured with Protego (Google semantics: * and $ wildcards) unless
ignore_robots is set.

Pages whose static HTML has no text, the same text as another page (one shell
for every URL of a JS app) or no links while more levels are needed are rendered
with one Chromium per crawl (guard_page, service workers blocked, headers only
to the start origin).

ponytail: one page at a time, no politeness delay and no Crawl-delay; add a
small semaphore when crawls get slow, a delay if a site complains.
"""
import asyncio
import hashlib
import logging
import os
import tempfile
import time
from collections import Counter, deque
from typing import Optional, Sequence
from urllib.parse import unquote, urldefrag, urljoin, urlparse

import lxml.html
from langchain_core.documents import Document
from protego import Protego
from requests.structures import CaseInsensitiveDict

from tilellm.models.llm import DEFAULT_BROWSER_USER_AGENT
from tilellm.shared.outbound_url import avalidate_outbound_url, guard_page, outbound_requests_get

logger = logging.getLogger(__name__)

# A crawler says what it is: CDNs answer 429 to a client that claims to be a browser and
# is not one, and robots.txt rules are matched against this name.
CRAWLER_USER_AGENT = "TiledeskBot/1.0 (+https://tiledesk.com)"
_PAGE_MAX_BYTES = 10 * 1024 * 1024
_ROBOTS_MAX_BYTES = 512 * 1024
_MAX_SECONDS = 30 * 60  # whole crawl: what is fetched by then is indexed
_clock = time.monotonic  # the deadline clock; tests replace it (patching time.monotonic breaks asyncio)
_RENDER_SETTLE_MS = 1500
_DISALLOW_ALL = "User-agent: *\nDisallow: /\n"
_NOT_HTML = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".7z",
             ".gz", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".mp3", ".mp4",
             ".avi", ".mov", ".css", ".js", ".json", ".xml", ".csv", ".txt")
# Files a crawl can index (crawl_file_types), recognised by Content-Type: some sites serve a
# PDF as ".aspx". application/octet-stream falls back to the extension.
_FILE_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
}
_EXTENSIONS = {t: "." + t for t in ("pdf", "docx", "xlsx", "xls", "csv", "txt", "md")}


def _origin(url: str) -> tuple:
    """(host, explicit port): http and https of the same site match, another port does not."""
    parsed = urlparse(url)
    return (parsed.hostname or "").lower(), parsed.port


def _page_key(url: str) -> str:
    """Same page whatever the scheme: sites often link both http:// and https://."""
    return url.split("://", 1)[-1]


def _text_key(page_docs: list[Document]) -> Optional[str]:
    """Fingerprint of a page's main text (first document; the rest are its tables)."""
    if not page_docs:
        return None
    return hashlib.sha1(page_docs[0].page_content.encode()).hexdigest()


def _fetch_html(url: str, headers: dict) -> tuple[Optional[str], Optional[str], Optional[tuple]]:
    """(html, None, None) for an HTML page, (None, target, None) for a redirect,
    (None, None, (file_type, bytes)) for a file a crawl can index, (None, None, None) otherwise."""
    response = outbound_requests_get(url, headers=headers, timeout=30, allow_redirects=False,
                                     max_bytes=_PAGE_MAX_BYTES)
    if response.is_redirect:
        return None, urljoin(url, response.headers["location"]), None
    response.raise_for_status()
    ctype = response.headers.get("content-type", "text/html").split(";")[0].strip().lower()
    if "html" in ctype:
        return response.text, None, None
    file_type = _FILE_TYPES.get(ctype)
    if file_type is None and ctype == "application/octet-stream":
        path = urlparse(url).path.lower()
        file_type = next((t for t, ext in _EXTENSIONS.items() if path.endswith(ext)), None)
    return None, None, (file_type, response.content) if file_type else None


def _load_bytes(data: bytes, file_type: str, url: str) -> list[Document]:
    """Documents of a downloaded file, read with the loaders of type=pdf/docx/... (no second download)."""
    from tilellm.tools.document_tools import _load_file

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "file" + _EXTENSIONS[file_type])
        with open(path, "wb") as f:
            f.write(data)
        docs = _load_file(path, file_type, url) or []
    docs = [d for d in docs if d.page_content.strip()]
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def _robots(start_url: str, headers: dict) -> Optional[Protego]:
    """robots.txt rules of the start origin: missing = everything allowed, server
    error = everything disallowed (RFC 9309), unreachable = everything allowed."""
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = outbound_requests_get(robots_url, headers=headers, timeout=10,
                                         max_bytes=_ROBOTS_MAX_BYTES)
    except Exception as exc:
        logger.info("crawl: robots.txt not readable at %s: %s", robots_url, exc)
        return None
    if response.status_code >= 500:
        return Protego.parse(_DISALLOW_ALL)
    if response.status_code != 200:
        return None
    return Protego.parse(response.text)


def _links(tree, page_url: str, origin: tuple, skip_ext: tuple = _NOT_HTML) -> list[str]:
    links = []
    for href in tree.xpath("//a/@href"):
        link, _ = urldefrag(urljoin(page_url, href.strip()))
        parsed = urlparse(link)
        if (parsed.scheme in ("http", "https") and _origin(link) == origin
                and not parsed.path.lower().endswith(skip_ext)):
            links.append(link)
    return links


def _parse(html: str, page_url: str, origin: tuple,
           skip_ext: tuple = _NOT_HTML) -> tuple[list[Document], list[str]]:
    """Text documents and same-origin links of a page (CPU work: run it in a thread)."""
    from tilellm.tools.document_tools import _trafilatura_documents

    docs = _trafilatura_documents(html, page_url)
    try:
        links = _links(lxml.html.fromstring(html), page_url, origin, skip_ext)
    except (ValueError, lxml.etree.ParserError):
        links = []
    return docs, links


def _header_route(origin: tuple, extra: dict):
    """Route handler: request headers (tokens, cookies) only to the start origin."""
    async def handler(route) -> None:
        if extra and _origin(route.request.url) == origin:
            await route.fallback(headers={**route.request.headers, **extra})
        else:
            await route.fallback()
    return handler


class _Browser:
    """One Chromium for the whole crawl, started by the first page that needs it."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    async def render(self, url: str, headers: dict, origin: tuple) -> tuple[str, str]:
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True, args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
        extra = {k: v for k, v in headers.items() if k.lower() != "user-agent"}
        # Chromium keeps its own user agent unless the caller chose one: JS apps compare the
        # declared user agent with the real browser and serve an empty shell on a mismatch.
        agent = CaseInsensitiveDict(headers).get("user-agent")
        page = await self._browser.new_page(
            **({} if agent in (None, CRAWLER_USER_AGENT) else {"user_agent": agent}),
            service_workers="block",  # page.route() does not see service-worker requests
            java_script_enabled=True,
        )
        try:
            await guard_page(page)
            await page.route("**/*", _header_route(origin, extra))  # runs first, falls back to the guard
            await page.goto(url=url, wait_until="networkidle", timeout=60000)
            # ponytail: fixed settle time, JS apps build the page after networkidle;
            # wait for a selector if some site needs more
            await page.wait_for_timeout(_RENDER_SETTLE_MS)
            for attempt in range(3):
                try:
                    return await page.content(), page.url
                except Exception:  # JS apps navigate client-side right after load
                    if attempt == 2:
                        raise
                    await page.wait_for_load_state("networkidle", timeout=30000)
        finally:
            await page.close()

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


async def _render_html(browser: _Browser, url: str, headers: dict, origin: tuple) -> tuple[str, str]:
    return await browser.render(url, headers, origin)


def _listed_type(url: str, file_types: Sequence[str]) -> Optional[str]:
    path = urlparse(url).path.lower()
    return next((t for t in file_types if path.endswith(_EXTENSIONS[t])), None)


async def _walk(start_url: str, *, max_depth: int, max_pages: int, ignore_robots: bool,
                file_types: Sequence[str], browser_headers: Optional[dict], stats: Counter,
                load_files: bool = True, max_seconds: float = _MAX_SECONDS):
    """Breadth-first walk shared by crawl_site and discover_site: yields
    (url, depth, documents, file_type or None) for every page or file with text.

    load_files=False lists files without reading them (documents is []): a link with
    an accepted extension is not even downloaded, other links still are, to read their
    Content-Type. `stats` counts fetched, indexed, files, failed, the skip reasons
    and `left` (links not visited when the walk stopped). The walk stops after
    `max_seconds`; a page being fetched at that moment still finishes."""
    await avalidate_outbound_url(start_url)
    headers = CaseInsensitiveDict(browser_headers or {})
    if headers.get("user-agent") in (None, DEFAULT_BROWSER_USER_AGENT):  # not a caller's choice
        headers["user-agent"] = CRAWLER_USER_AGENT
    headers = dict(headers)
    agent = CaseInsensitiveDict(headers)["user-agent"]

    # links to accepted files (e.g. ".pdf") are followed, other non-HTML extensions are not
    skip_ext = tuple(e for e in _NOT_HTML if e not in {_EXTENSIONS[t] for t in file_types})

    start = urldefrag(start_url)[0]
    origin = _origin(start)
    rules = None if ignore_robots else await asyncio.to_thread(_robots, start, headers)
    queue, seen = deque([(start, 0)]), {_page_key(start)}
    seen_texts: set[str] = set()
    deadline = _clock() + max_seconds
    browser = _Browser()
    try:
        while queue and stats["fetched"] < max_pages:
            if _clock() > deadline:
                logger.warning("crawl %s: time limit reached, %d pages left", start, len(queue))
                break
            url, depth = queue.popleft()
            if rules is not None and not rules.can_fetch(url, agent):
                stats["robots"] += 1
                continue
            stats["fetched"] += 1
            listed = None if load_files else _listed_type(url, file_types)
            if listed:
                stats["files"] += 1
                stats["indexed"] += 1
                yield url, depth, [], listed
                continue
            try:
                html, target, file = await asyncio.to_thread(_fetch_html, url, headers)
            except Exception as exc:
                stats["failed"] += 1
                logger.warning("crawl: page skipped %s: %s", url, exc)
                continue

            if target:
                target = urldefrag(target)[0]
                if depth == 0 and not stats["indexed"] and _origin(target) != origin:
                    # the start page itself moved (example.org -> www.example.org)
                    await avalidate_outbound_url(target)
                    origin = _origin(target)
                    if not ignore_robots:
                        rules = await asyncio.to_thread(_robots, target, headers)
                if _origin(target) != origin:
                    stats["other_host"] += 1
                    logger.info("crawl: page skipped %s: redirected to another host", url)
                else:
                    stats["redirect"] += 1
                    if _page_key(target) not in seen:
                        seen.add(_page_key(target))
                        queue.appendleft((target, depth))
                continue
            file_type = None
            if file is not None and file[0] in file_types:
                file_type = file[0]
                if not load_files:
                    stats["files"] += 1
                    stats["indexed"] += 1
                    yield url, depth, [], file_type
                    continue
                try:
                    page_docs = await asyncio.to_thread(_load_bytes, file[1], file_type, url)
                except Exception as exc:
                    stats["failed"] += 1
                    logger.warning("crawl: file skipped %s: %s", url, exc)
                    continue
                stats["files"] += 1
                links, static_key = [], None
            elif html is None:
                stats["not_html"] += 1  # images, archives, file types not accepted
                continue
            else:
                page_docs, links = await asyncio.to_thread(_parse, html, url, origin, skip_ext)
                static_key = _text_key(page_docs)
                if not page_docs or static_key in seen_texts or (not links and depth < max_depth):
                    try:
                        rendered, rendered_url = await _render_html(browser, url, headers, origin)
                        if _origin(rendered_url) == origin:
                            r_docs, r_links = await asyncio.to_thread(
                                _parse, rendered, rendered_url, origin, skip_ext)
                            page_docs, links = r_docs or page_docs, r_links or links
                    except Exception as exc:
                        logger.warning("crawl: browser failed on %s, static page kept: %s", url, exc)

            text_key = _text_key(page_docs)
            if not page_docs:
                stats["no_text"] += 1
            elif text_key in seen_texts:
                stats["duplicate"] += 1
                page_docs = []  # same text as a page already indexed
            seen_texts.update(k for k in (static_key, text_key) if k)

            if depth < max_depth:
                for link in links:
                    # ponytail: the queue never holds more than max_pages links
                    if _page_key(link) not in seen and len(queue) < max_pages:
                        seen.add(_page_key(link))
                        queue.append((link, depth + 1))
            if page_docs:
                stats["indexed"] += 1
                yield url, depth, page_docs, file_type
        stats["left"] = len(queue)
    finally:
        await browser.close()


_SKIPPED = ("robots", "redirect", "other_host", "duplicate", "no_text", "not_html", "failed")


async def crawl_site(
    start_url: str,
    *,
    max_depth: int = 3,
    max_pages: int = 50,
    ignore_robots: bool = False,
    file_types: Sequence[str] = ("pdf", "docx"),
    browser_headers: Optional[dict] = None,
) -> list[Document]:
    """Documents of every crawled page, each with its own `source` URL.

    Raises when more pages failed than succeeded: the caller replaces the whole
    site under one id, a mostly failed crawl must not do that silently."""
    stats: Counter = Counter()
    docs: list[Document] = []
    async for _, _, page_docs, _ in _walk(start_url, max_depth=max_depth, max_pages=max_pages,
                                          ignore_robots=ignore_robots, file_types=file_types,
                                          browser_headers=browser_headers, stats=stats):
        docs.extend(page_docs)

    fetched, failed = stats["fetched"], stats["failed"]
    if stats["left"] and fetched >= max_pages:
        logger.warning("crawl %s: max_pages=%d reached, %d links not followed", start_url, max_pages, stats["left"])
    logger.info(
        "crawl %s: %d fetched, indexed=%d (files=%d), failed=%d, redirect=%d, other_host=%d, "
        "duplicate=%d, no_text=%d, not_html=%d, robots=%d, documents=%d",
        start_url, fetched, stats["indexed"], stats["files"], failed, stats["redirect"], stats["other_host"],
        stats["duplicate"], stats["no_text"], stats["not_html"], stats["robots"], len(docs),
    )
    if failed and failed * 2 > fetched:
        raise ValueError(f"crawl {start_url}: {failed} of {fetched} pages failed")
    return docs


async def discover_site(
    start_url: str,
    *,
    max_depth: int = 3,
    max_pages: int = 50,
    ignore_robots: bool = False,
    file_types: Sequence[str] = ("pdf", "docx"),
    browser_headers: Optional[dict] = None,
    max_seconds: float = 300,
) -> dict:
    """The pages and files a crawl_site with the same options would index, without
    indexing them: [{url, title, kind, file_type?, depth}], `truncated` and why other
    URLs were skipped. Files with an accepted extension are listed from the link.
    The walk ends after `max_seconds` (truncated=True, partial list)."""
    stats: Counter = Counter()
    pages = []
    async for url, depth, page_docs, file_type in _walk(
            start_url, max_depth=max_depth, max_pages=max_pages, ignore_robots=ignore_robots,
            file_types=file_types, browser_headers=browser_headers, stats=stats, load_files=False,
            max_seconds=max_seconds):
        if file_type:
            title = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]) or None
            pages.append({"url": url, "title": title, "kind": "file", "file_type": file_type, "depth": depth})
        else:
            pages.append({"url": url, "title": page_docs[0].metadata.get("file_name"), "kind": "page",
                          "depth": depth})
    logger.info("crawl discover %s: %d listed, %d fetched", start_url, len(pages), stats["fetched"])
    return {"pages": pages, "truncated": stats["left"] > 0, "fetched": stats["fetched"],
            "skipped": {k: stats[k] for k in _SKIPPED}}
