"""Table-aware chunker for HTML/markdown tables extracted from web pages.

Documents with ``metadata.element_type == "table"`` bypass the generic
RecursiveCharacterTextSplitter and are routed here to preserve row/column
alignment during ingestion.
"""
import copy
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Reuse parser from text_processor to avoid duplication
from tilellm.modules.ingestion.text_processor import _parse_pipe_table


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_markdown_table(content: str):
    """Parse a markdown ``| ... |`` table.

    Returns (headers, align_row, data_rows) or None when parsing fails.
    * headers: list[str]
    * align_row: str (the ``|---|---|`` line, may be empty string if absent)
    * data_rows: list[str] (raw markdown lines for each data row)
    """
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return None

    table_lines = [l for l in lines if '|' in l]
    if len(table_lines) < 2:
        return None

    parsed = _parse_pipe_table(table_lines)
    if not parsed or not parsed.get('rows'):
        return None

    headers = parsed['headers']
    alignment = parsed.get('alignment') or ("|" + "|".join([" --- "] * len(headers)) + "|")

    # Reconstruct raw data-row lines from the parsed row dicts
    data_row_lines = []
    for row_dict in parsed['rows']:
        cells = [str(row_dict.get(h, "")) for h in headers]
        data_row_lines.append("| " + " | ".join(cells) + " |")

    return headers, alignment, data_row_lines


def _header_md(headers: List[str], alignment: str) -> str:
    return "| " + " | ".join(headers) + " |\n" + alignment


def _make_chunk(
    content: str,
    base_meta: Dict[str, Any],
    *,
    element_type: str,
    chunk_type: str = "table",
    row_index: Optional[int] = None,
    row_range: Optional[str] = None,
    total_rows: Optional[int] = None,
) -> Document:
    meta = copy.deepcopy(base_meta)
    meta["element_type"] = element_type
    meta["chunk_type"] = chunk_type
    if row_index is not None:
        meta["row_index"] = row_index
    if row_range is not None:
        meta["row_range"] = row_range
    if total_rows is not None:
        meta["total_rows"] = total_rows
    # Pinecone compat: col_names must be str
    if "col_names" in meta and not isinstance(meta["col_names"], str):
        meta["col_names"] = str(meta["col_names"])
    return Document(page_content=content.strip(), metadata=meta)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback(doc: Document) -> List[Document]:
    """Return the document unchanged when table parsing fails."""
    logger.debug("table_chunker: parsing failed, returning doc as-is")
    return [Document(page_content=doc.page_content, metadata=copy.deepcopy(doc.metadata))]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _atomic(doc: Document) -> List[Document]:
    meta = copy.deepcopy(doc.metadata)
    meta.setdefault("chunk_type", "table")
    if "col_names" in meta and not isinstance(meta["col_names"], str):
        meta["col_names"] = str(meta["col_names"])
    return [Document(page_content=doc.page_content, metadata=meta)]


def _per_row(doc: Document, header_repeat: bool = True) -> List[Document]:
    parsed = _parse_markdown_table(doc.page_content)
    if not parsed:
        return _fallback(doc)

    headers, alignment, data_rows = parsed
    if not data_rows:
        return _fallback(doc)

    total = len(data_rows)
    chunks = []
    for i, row_line in enumerate(data_rows):
        if header_repeat:
            content = _header_md(headers, alignment) + "\n" + row_line
        else:
            content = row_line
        chunk = _make_chunk(
            content,
            doc.metadata,
            element_type="table_rows",
            chunk_type="table",
            row_index=i,
            row_range=f"{i}-{i}",
            total_rows=total,
        )
        chunks.append(chunk)
    return chunks


def _adaptive(
    doc: Document,
    max_table_chars: int = 8000,
    header_repeat: bool = True,
) -> List[Document]:
    if len(doc.page_content) <= max_table_chars:
        return _atomic(doc)

    parsed = _parse_markdown_table(doc.page_content)
    if not parsed:
        return _fallback(doc)

    headers, alignment, data_rows = parsed
    if not data_rows:
        return _fallback(doc)

    total = len(data_rows)
    header_block = _header_md(headers, alignment)
    chunks: List[Document] = []
    group: List[str] = []
    group_start = 0
    current_size = len(header_block) if header_repeat else 0

    def _flush(group: List[str], start: int) -> None:
        if not group:
            return
        content_parts = ([header_block] if header_repeat else []) + group
        content = "\n".join(content_parts)
        end = start + len(group) - 1
        chunk = _make_chunk(
            content,
            doc.metadata,
            element_type="table_rows",
            chunk_type="table",
            row_range=f"{start}-{end}",
            total_rows=total,
        )
        chunks.append(chunk)

    for i, row_line in enumerate(data_rows):
        row_size = len(row_line) + 1  # +1 for newline
        if group and current_size + row_size > max_table_chars:
            _flush(group, group_start)
            group = []
            group_start = i
            current_size = len(header_block) if header_repeat else 0
        group.append(row_line)
        current_size += row_size

    _flush(group, group_start)
    return chunks if chunks else _fallback(doc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_table_document(
    doc: Document,
    strategy: str = "adaptive",
    max_table_chars: int = 8000,
    header_repeat: bool = True,
) -> List[Document]:
    """Split a table Document according to the given strategy.

    Args:
        doc: A Document with ``metadata.element_type == "table"`` (markdown content).
        strategy: One of ``"atomic"`` | ``"per_row"`` | ``"adaptive"``.
        max_table_chars: Size threshold (chars) used by the adaptive strategy.
        header_repeat: Repeat the header row in every sub-chunk.

    Returns:
        List of Document chunks with enriched table metadata.
    """
    if not doc.page_content or not doc.page_content.strip():
        return [Document(page_content=doc.page_content or "", metadata=copy.deepcopy(doc.metadata))]

    if strategy == "atomic":
        return _atomic(doc)
    if strategy == "per_row":
        return _per_row(doc, header_repeat=header_repeat)
    if strategy == "adaptive":
        return _adaptive(doc, max_table_chars=max_table_chars, header_repeat=header_repeat)

    logger.warning("table_chunker: unknown strategy '%s', falling back to atomic", strategy)
    return _atomic(doc)
