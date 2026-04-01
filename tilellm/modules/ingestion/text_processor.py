"""
Auto-detect and process direct text content.

Detects text format (Markdown, tabular, plain) and applies appropriate
chunking strategy to preserve semantic structure.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextFormat:
    """Detected text format."""
    MARKDOWN = "markdown"
    TABULAR = "tabular"
    PLAIN = "plain"


def auto_detect_text_format(content: str) -> str:
    """
    Auto-detect text format from content patterns.

    Returns:
        - 'markdown': Content starts with heading (#) or has heading patterns
        - 'tabular': Content has pipe-delimited table patterns (|...|)
        - 'plain': Standard text
    """
    if not content or not isinstance(content, str):
        return TextFormat.PLAIN

    lines = content.split('\n')

    # Check for Markdown indicators
    for line in lines[:10]:  # Check first 10 lines
        stripped = line.strip()
        # Heading indicators
        if stripped.startswith('#'):
            return TextFormat.MARKDOWN
        # Heading underline (=== or ---)
        if re.match(r'^[=\-]{3,}$', stripped):
            return TextFormat.MARKDOWN

    # Check for table patterns (pipe-delimited)
    # Must have multiple rows with pipes and alignment row (|---|)
    pipe_lines = [l for l in lines if '|' in l]
    if len(pipe_lines) >= 2:
        # Check for markdown table alignment row (|---|---|)
        for i, line in enumerate(pipe_lines):
            if re.search(r'\|\s*[-:]+\s*\|', line):
                return TextFormat.TABULAR
        # Also detect simple pipe tables (2+ rows with consistent pipes)
        if all('|' in l for l in pipe_lines[:min(3, len(pipe_lines))]):
            return TextFormat.TABULAR

    return TextFormat.PLAIN


def extract_tables(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract structured tables from pipe-delimited content.

    Returns:
        Tuple of:
        - Remaining content (with tables removed/marked)
        - List of extracted table structures
    """
    tables = []
    lines = content.split('\n')
    i = 0
    remaining_lines = []

    while i < len(lines):
        line = lines[i]

        # Check if this is a table start
        if '|' in line:
            # Try to parse table starting at this line
            table_lines = [line]
            i += 1

            # Collect alignment row and data rows
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # Parse the table
            if len(table_lines) >= 2:
                try:
                    table_data = _parse_pipe_table(table_lines)
                    if table_data:
                        tables.append(table_data)
                        remaining_lines.append(f"\n[TABLE {len(tables)}]\n")
                        continue
                except Exception as e:
                    logger.debug(f"Failed to parse table: {e}")

        remaining_lines.append(line)
        i += 1

    return '\n'.join(remaining_lines), tables


def _parse_pipe_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a pipe-delimited table into structured format."""
    if len(lines) < 2:
        return None

    # Extract header
    header_line = lines[0].strip()
    if not header_line.startswith('|') or not header_line.endswith('|'):
        return None

    headers = [h.strip() for h in header_line.split('|')[1:-1]]

    # Check for alignment row
    align_idx = 1
    alignment = None
    if len(lines) > 1:
        align_line = lines[1].strip()
        if re.match(r'^\|[\s\-:|\s]+\|$', align_line):
            alignment = align_line
            align_idx = 2
        else:
            align_idx = 1

    # Extract data rows
    rows = []
    for line_idx in range(align_idx, len(lines)):
        line = lines[line_idx].strip()
        if not line.startswith('|') or not line.endswith('|'):
            break

        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))

    if rows:
        return {
            'type': 'table',
            'headers': headers,
            'rows': rows,
            'alignment': alignment,
            'row_count': len(rows)
        }

    return None


async def process_auto_detected_text(
    content: str,
    source: str,
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 400,
    metadata: Optional[Dict[str, Any]] = None,
    semantic_chunk: bool = False,
    embeddings: Optional[Any] = None,
) -> List[Document]:
    """
    Process text with auto-detected format.

    Applies format-specific chunking and metadata enrichment.

    Args:
        content: Raw text content
        source: Source reference
        doc_id: Document ID
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata to include
        semantic_chunk: Enable semantic chunking
        embeddings: Embedding model (for semantic chunking)

    Returns:
        List of chunked Document objects with enriched metadata
    """
    detected_format = auto_detect_text_format(content)
    logger.info(f"Detected text format: {detected_format} for {doc_id}")

    # Prepare base metadata
    base_metadata = {
        'id': doc_id,
        'source': source,
        'type': detected_format,
        'detected_format': detected_format,
    }
    if metadata:
        base_metadata.update(metadata)

    documents = []

    if detected_format == TextFormat.MARKDOWN:
        try:
            from tilellm.modules.pdf_ocr.services.markdown_chunker import MarkdownChunker

            chunker = MarkdownChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                respect_headings=True,
                respect_tables=True,
                include_heading_context=True
            )
            documents = chunker.chunk_markdown(content, doc_id, base_metadata)
            logger.info(f"Markdown chunking produced {len(documents)} chunks")

        except Exception as e:
            logger.warning(f"Markdown chunking failed, falling back to plain: {e}")
            documents = _chunk_plain_text(content, chunk_size, chunk_overlap, base_metadata)

    elif detected_format == TextFormat.TABULAR:
        try:
            # Extract tables and get remaining content
            remaining_content, tables = extract_tables(content)

            # Create documents for each table
            for table_idx, table in enumerate(tables):
                table_content = _format_table_content(table)
                table_meta = {
                    **base_metadata,
                    'chunk_type': 'table',
                    'table_index': table_idx,
                    'table_row_count': table.get('row_count', 0),
                    'has_table': True,
                }
                table_doc = Document(
                    page_content=table_content,
                    metadata=table_meta
                )
                documents.append(table_doc)

            # Chunk remaining text
            if remaining_content.strip():
                remaining_meta = {**base_metadata, 'has_table': False}
                text_docs = _chunk_plain_text(
                    remaining_content, chunk_size, chunk_overlap, remaining_meta
                )
                documents.extend(text_docs)

            logger.info(f"Tabular processing extracted {len(tables)} tables and "
                       f"{len([d for d in documents if d.metadata.get('chunk_type') != 'table'])} text chunks")

        except Exception as e:
            logger.warning(f"Table extraction failed, falling back to plain: {e}")
            documents = _chunk_plain_text(content, chunk_size, chunk_overlap, base_metadata)

    else:  # TextFormat.PLAIN
        documents = _chunk_plain_text(content, chunk_size, chunk_overlap, base_metadata)

    # Add chunk indexing metadata
    for idx, doc in enumerate(documents):
        doc.metadata['chunk_index'] = idx
        doc.metadata['total_chunks'] = len(documents)

    return documents


def _chunk_plain_text(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Dict[str, Any]
) -> List[Document]:
    """Chunk plain text using standard splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    text_chunks = splitter.split_text(content)
    documents = [
        Document(page_content=chunk, metadata=metadata.copy())
        for chunk in text_chunks
    ]

    logger.info(f"Plain text chunking produced {len(documents)} chunks")
    return documents


def _format_table_content(table: Dict[str, Any]) -> str:
    """Format extracted table into readable string."""
    headers = table.get('headers', [])
    rows = table.get('rows', [])

    # Build formatted table string
    lines = []

    # Header
    if headers:
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join([' --- ' for _ in headers]) + '|')

    # Rows
    for row in rows:
        cells = [str(row.get(h, '')) for h in headers]
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)
