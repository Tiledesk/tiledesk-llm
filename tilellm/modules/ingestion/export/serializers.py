"""
ExtractedDocument <-> Markdown+frontmatter, ExtractedDocument <-> JSON.

Both are projections of the single `ExtractedDocument` model (models.py) — no
separate parsing logic per format. JSON is the lossless contract; Markdown is
the human-curable one, using an HTML-comment marker to carry per-block
structure (page / heading_path / block_type) when it exists. A document with a
single untyped block serializes with no markers, so plain text/passthrough
output stays clean and human-friendly.
"""
import json
import re
from typing import List

import yaml

from tilellm.modules.ingestion.export.models import Block, ExtractedDocument

_FRONTMATTER_FIELDS = ("type", "title", "description", "resource", "tags", "timestamp")

_BLOCK_MARKER_RE = re.compile(
    r'^<!--block(?: type="(?P<type>[^"]*)")?(?: page=(?P<page>\d+))?'
    r'(?: position=(?P<position>\d+))?(?: heading="(?P<heading>[^"]*)")?-->$',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def to_json(doc: ExtractedDocument) -> str:
    return doc.model_dump_json()


def from_json(raw: str) -> ExtractedDocument:
    return ExtractedDocument.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Markdown + frontmatter
# ---------------------------------------------------------------------------

def _is_structural(block: Block, is_only_block: bool) -> bool:
    """A block needs an explicit marker unless it's the sole, untyped block."""
    if (
        is_only_block
        and block.block_type == "text"
        and block.page is None
        and block.position is None
        and block.heading_path is None
    ):
        return False
    return True


def _block_marker(block: Block) -> str:
    parts = []
    if block.block_type != "text":
        parts.append(f'type="{block.block_type}"')
    if block.page is not None:
        parts.append(f"page={block.page}")
    if block.position is not None:
        parts.append(f"position={block.position}")
    if block.heading_path:
        parts.append(f'heading="{block.heading_path}"')
    return f"<!--block {' '.join(parts)}-->" if parts else "<!--block-->"


def to_md(doc: ExtractedDocument) -> str:
    frontmatter = {k: getattr(doc, k) for k in _FRONTMATTER_FIELDS if getattr(doc, k) not in (None, [], "")}
    frontmatter.update(doc.extra)
    fm_yaml = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False)

    ordered = sorted(doc.blocks, key=lambda b: b.order)
    single = len(ordered) == 1
    parts = []
    for block in ordered:
        if single and not _is_structural(block, is_only_block=True):
            parts.append(block.content)
        else:
            parts.append(f"{_block_marker(block)}\n{block.content}")

    body = "\n\n".join(parts)
    return f"---\n{fm_yaml}---\n\n{body}\n"


def from_md(raw: str) -> ExtractedDocument:
    match = re.match(r"^---\n(.*?)\n---\n\n?(.*)$", raw, re.DOTALL)
    if not match:
        raise ValueError("Markdown input has no YAML frontmatter block (--- ... ---).")

    frontmatter = yaml.safe_load(match.group(1)) or {}
    if not frontmatter.get("type"):
        raise ValueError("Frontmatter is missing the required 'type' field.")

    body = match.group(2)
    known = {k: frontmatter.get(k) for k in _FRONTMATTER_FIELDS if k in frontmatter}
    extra = {k: v for k, v in frontmatter.items() if k not in _FRONTMATTER_FIELDS}

    blocks = _parse_blocks(body)
    return ExtractedDocument(**known, extra=extra, blocks=blocks)


def _parse_blocks(body: str) -> List[Block]:
    markers = list(_BLOCK_MARKER_RE.finditer(body))
    if not markers:
        content = body.strip()
        return [Block(content=content, order=0)] if content else []

    blocks: List[Block] = []
    for i, m in enumerate(markers):
        start = m.end() + 1  # skip the newline right after the marker
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        content = body[start:end].strip("\n")
        # trailing blank-line separator between blocks
        content = content.rstrip("\n")
        if content.endswith("\n\n"):
            content = content[:-1]
        blocks.append(Block(
            content=content.rstrip(),
            block_type=m.group("type") or "text",
            page=int(m.group("page")) if m.group("page") else None,
            position=int(m.group("position")) if m.group("position") else None,
            heading_path=m.group("heading"),
            order=i,
        ))
    return blocks
