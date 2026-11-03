"""Crawl limits shared by ItemSingle (type=crawl) and POST /api/crawl/discover."""
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ponytail: fixed caps; move them to env only if a deployment needs bigger crawls
CrawlDepth = Annotated[int, Field(ge=0, le=10)]
CrawlPages = Annotated[int, Field(ge=1, le=500)]
CrawlFileType = Literal["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"]


class CrawlDiscoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an ingestion field here is a mistake, not a no-op

    source: str = Field(description="Start page (http/https).")
    max_depth: CrawlDepth = Field(default=3, description="Link levels followed from the start page (0 = start page only).")
    max_pages: CrawlPages = Field(default=50, description="Maximum number of pages and files listed or fetched.")
    ignore_robots: bool = Field(default=False, description="true skips the site's robots.txt rules.")
    crawl_file_types: List[CrawlFileType] = Field(
        default_factory=lambda: ["pdf", "docx"],
        description="Linked files of these types are listed too. [] = HTML pages only.",
    )
    browser_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Request headers, sent only to the start origin.")
    max_seconds: int = Field(
        default=300, ge=30, le=1800,
        description="Time limit of the walk: when it ends the list is partial (truncated=true).")


class DiscoveredPage(BaseModel):
    url: str
    title: Optional[str] = None
    kind: Literal["page", "file"]
    file_type: Optional[CrawlFileType] = None
    depth: int


class CrawlDiscoverResult(BaseModel):
    pages: List[DiscoveredPage]
    truncated: bool = Field(description="true when max_pages or the time limit left links unvisited.")
    fetched: int = Field(description="URLs processed, max_pages included (listed files are not downloaded).")
    skipped: Dict[str, int] = Field(
        description="Why URLs were not listed: robots, redirect, other_host, duplicate, no_text, not_html, failed.")
