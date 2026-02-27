"""
Markdown Chunker Service for PDF OCR Module.

This service provides specialized chunking for Markdown documents,
preserving semantic structure and context.

Features:
- Structure-aware chunking (respects headings, sections)
- Configurable chunk sizes and overlap
- Smart boundary detection (paragraphs, lists, tables)
- Context preservation between chunks
- Metadata preservation for source tracking
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MarkdownElementType(Enum):
    """Types of Markdown elements."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    IMAGE = "image"
    HORIZONTAL_RULE = "horizontal_rule"
    BLOCKQUOTE = "blockquote"


@dataclass
class MarkdownElement:
    """Represents a parsed Markdown element."""
    element_type: MarkdownElementType
    content: str
    level: int = 0  # For headings (1-6)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0


@dataclass
class MarkdownChunk:
    """Represents a chunk of Markdown content."""
    content: str
    elements: List[MarkdownElement]
    metadata: Dict[str, Any]
    chunk_index: int
    start_line: int
    end_line: int
    headings_context: List[str] = field(default_factory=list)  # Hierarchy of headings


class MarkdownChunker:
    """
    Specialized chunker for Markdown documents.
    
    Preserves semantic structure by:
    - Respecting heading boundaries
    - Keeping related content together
    - Maintaining table integrity
    - Preserving image descriptions
    - Including heading context in metadata
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_headings: bool = True,
        respect_tables: bool = True,
        include_heading_context: bool = True
    ):
        """
        Initialize Markdown chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to avoid tiny chunks
            respect_headings: Whether to keep headings with their content
            respect_tables: Whether to keep tables intact
            include_heading_context: Whether to include heading hierarchy in metadata
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_headings = respect_headings
        self.respect_tables = respect_tables
        self.include_heading_context = include_heading_context
        
        logger.info(f"MarkdownChunker initialized: size={chunk_size}, "
                   f"overlap={chunk_overlap}, respect_headings={respect_headings}")
    
    def chunk_markdown(
        self,
        markdown_content: str,
        doc_id: str,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Chunk Markdown content into LangChain Documents.
        
        Args:
            markdown_content: Full Markdown content
            doc_id: Document identifier
            source_metadata: Additional metadata to include
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Chunking Markdown document {doc_id}")
        
        # Parse Markdown into elements
        elements = self._parse_markdown(markdown_content)
        logger.info(f"Parsed {len(elements)} Markdown elements")
        
        # Group elements into chunks
        chunks = self._group_elements_into_chunks(elements, markdown_content)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Convert to LangChain Documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Build metadata
            metadata = {
                'doc_id': doc_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'markdown',
                'source': f"md_{doc_id}",
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'element_types': list(set(e.element_type.value for e in chunk.elements)),
                'headings_context': chunk.headings_context,
                'has_images': any(e.element_type == MarkdownElementType.IMAGE for e in chunk.elements),
                'has_tables': any(e.element_type == MarkdownElementType.TABLE for e in chunk.elements),
            }
            
            # Add source metadata
            if source_metadata:
                metadata.update(source_metadata)
            
            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} LangChain Documents from Markdown")
        return documents
    
    def _parse_markdown(self, content: str) -> List[MarkdownElement]:
        """
        Parse Markdown content into structured elements.
        
        Uses regex-based parsing to identify different Markdown elements.
        """
        elements = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            start_line = i
            
            # Check for heading (ATX style)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.HEADING,
                    content=line,
                    level=level,
                    start_line=start_line,
                    end_line=start_line
                ))
                i += 1
                continue
            
            # Check for code block (fenced)
            if line.startswith('```'):
                code_content = [line]
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    code_content.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_content.append(lines[i])  # closing ```
                    i += 1
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.CODE_BLOCK,
                    content='\n'.join(code_content),
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Check for table
            if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
                table_lines = [line]
                i += 1
                # Header separator
                table_lines.append(lines[i])
                i += 1
                # Table rows
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.TABLE,
                    content='\n'.join(table_lines),
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Check for image
            image_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', line)
            if image_match:
                alt_text = image_match.group(1)
                url = image_match.group(2)
                
                # Check if there's a caption after the image
                caption_lines = [line]
                i += 1
                
                # Capture any following lines that might be part of the image description
                # (e.g., bold text starting with "Image:" or italic caption)
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line and (next_line.startswith('**Image:') or 
                                     next_line.startswith('*Image:') or
                                     next_line.startswith('**Location:**') or
                                     (next_line.startswith('**') and 'Image' in next_line)):
                        caption_lines.append(lines[i])
                        i += 1
                    elif next_line and not next_line.startswith('#') and not next_line.startswith('!['):
                        # Check if it's a description paragraph
                        caption_lines.append(lines[i])
                        i += 1
                        break
                    else:
                        break
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.IMAGE,
                    content='\n'.join(caption_lines),
                    metadata={'alt_text': alt_text, 'url': url},
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Check for list
            list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', line)
            if list_match:
                list_lines = [line]
                i += 1
                
                # Continue capturing list items
                while i < len(lines):
                    next_line = lines[i]
                    # Check if it's another list item
                    if re.match(r'^(\s*)([-*+]|\d+\.)\s+', next_line):
                        list_lines.append(next_line)
                        i += 1
                    # Check if it's a continuation (indented or blank)
                    elif next_line.strip() == '' or next_line.startswith('  ') or next_line.startswith('\t'):
                        list_lines.append(next_line)
                        i += 1
                    else:
                        break
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.LIST,
                    content='\n'.join(list_lines),
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Check for horizontal rule
            if re.match(r'^(---|\*\*\*|___)\s*$', line):
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.HORIZONTAL_RULE,
                    content=line,
                    start_line=start_line,
                    end_line=start_line
                ))
                i += 1
                continue
            
            # Check for blockquote
            if line.startswith('>'):
                quote_lines = [line]
                i += 1
                while i < len(lines) and (lines[i].startswith('>') or lines[i].strip() == ''):
                    quote_lines.append(lines[i])
                    i += 1
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.BLOCKQUOTE,
                    content='\n'.join(quote_lines),
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Treat as paragraph
            if line.strip():
                para_lines = [line]
                i += 1
                # Continue until blank line or new element
                while i < len(lines) and lines[i].strip() and not self._is_element_start(lines[i]):
                    para_lines.append(lines[i])
                    i += 1
                
                elements.append(MarkdownElement(
                    element_type=MarkdownElementType.PARAGRAPH,
                    content='\n'.join(para_lines),
                    start_line=start_line,
                    end_line=i - 1
                ))
                continue
            
            # Skip empty lines
            i += 1
        
        return elements
    
    def _is_element_start(self, line: str) -> bool:
        """Check if a line starts a new Markdown element."""
        stripped = line.strip()
        
        # Heading
        if re.match(r'^#{1,6}\s+', stripped):
            return True
        
        # Code block
        if stripped.startswith('```'):
            return True
        
        # Table
        if '|' in stripped:
            return True
        
        # Image
        if stripped.startswith('!['):
            return True
        
        # List
        if re.match(r'^([-*+]|\d+\.)\s+', stripped):
            return True
        
        # Horizontal rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            return True
        
        # Blockquote
        if stripped.startswith('>'):
            return True
        
        return False
    
    def _group_elements_into_chunks(
        self, 
        elements: List[MarkdownElement],
        original_content: str
    ) -> List[MarkdownChunk]:
        """
        Group Markdown elements into chunks respecting structure.
        """
        chunks = []
        current_chunk_elements = []
        current_chunk_size = 0
        chunk_index = 0
        current_headings = []  # Stack of current headings
        
        for i, element in enumerate(elements):
            element_size = len(element.content)
            
            # Update heading context
            if element.element_type == MarkdownElementType.HEADING:
                # Pop headings that are at same or lower level
                while current_headings and current_headings[-1][0] >= element.level:
                    current_headings.pop()
                current_headings.append((element.level, element.content))
            
            # Check if we need to start a new chunk
            need_new_chunk = False
            
            # If adding this element would exceed chunk size significantly
            if current_chunk_size + element_size > self.chunk_size and current_chunk_elements:
                # But respect certain elements
                if element.element_type == MarkdownElementType.HEADING and self.respect_headings:
                    # New chunk for new section
                    need_new_chunk = True
                elif element.element_type == MarkdownElementType.TABLE and self.respect_tables:
                    # Keep tables intact
                    if element_size > self.chunk_size:
                        # Table is too big, must split
                        need_new_chunk = True
                    elif current_chunk_size + element_size > self.chunk_size * 1.5:
                        # Would make chunk too large
                        need_new_chunk = True
                elif element_size > self.chunk_size:
                    # Element itself is too large, split it
                    need_new_chunk = True
                elif current_chunk_size >= self.chunk_size - self.chunk_overlap:
                # Chunk is getting full
                    need_new_chunk = True
            
            # Start new chunk if needed
            if need_new_chunk:
                # Save current chunk
                if current_chunk_elements:
                    chunk = self._create_chunk(
                        current_chunk_elements, 
                        chunk_index, 
                        current_headings
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                # Include last heading for context
                current_chunk_elements = []
                current_chunk_size = 0
                
                # Add context from previous chunk (heading)
                if self.include_heading_context and current_headings:
                    # Add the most recent heading for context
                    last_heading = current_headings[-1]
                    heading_element = MarkdownElement(
                        element_type=MarkdownElementType.HEADING,
                        content=last_heading[1],
                        level=last_heading[0]
                    )
                    current_chunk_elements.append(heading_element)
                    current_chunk_size += len(heading_element.content)
            
            # Add element to current chunk
            current_chunk_elements.append(element)
            current_chunk_size += element_size
            
            # If element is very large (table, code block), consider finalizing chunk
            if element_size > self.chunk_size and element.element_type in [
                MarkdownElementType.TABLE, 
                MarkdownElementType.CODE_BLOCK
            ]:
                # Finalize this chunk
                chunk = self._create_chunk(
                    current_chunk_elements, 
                    chunk_index, 
                    current_headings
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk_elements = []
                current_chunk_size = 0
        
        # Add final chunk
        if current_chunk_elements:
            chunk = self._create_chunk(
                current_chunk_elements, 
                chunk_index, 
                current_headings
            )
            chunks.append(chunk)
        
        # Merge tiny chunks with neighbors
        chunks = self._merge_tiny_chunks(chunks)
        
        return chunks
    
    def _create_chunk(
        self, 
        elements: List[MarkdownElement], 
        chunk_index: int,
        current_headings: List[Tuple[int, str]]
    ) -> MarkdownChunk:
        """Create a MarkdownChunk from elements."""
        content = '\n\n'.join(e.content for e in elements)
        
        start_line = elements[0].start_line if elements else 0
        end_line = elements[-1].end_line if elements else 0
        
        # Extract heading context
        headings_context = [h[1] for h in current_headings]
        
        return MarkdownChunk(
            content=content,
            elements=elements.copy(),
            metadata={},
            chunk_index=chunk_index,
            start_line=start_line,
            end_line=end_line,
            headings_context=headings_context
        )
    
    def _merge_tiny_chunks(self, chunks: List[MarkdownChunk]) -> List[MarkdownChunk]:
        """Merge chunks that are too small with their neighbors."""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            # If chunk is too small and not the only one
            if len(chunk.content) < self.min_chunk_size and len(merged) > 0:
                # Merge with previous chunk
                prev_chunk = merged[-1]
                prev_chunk.content += '\n\n' + chunk.content
                prev_chunk.elements.extend(chunk.elements)
                prev_chunk.end_line = chunk.end_line
                # Keep the previous chunk's headings context (more complete)
            else:
                merged.append(chunk)
            
            i += 1
        
        return merged
    
    def chunk_with_semantic_splitting(
        self,
        markdown_content: str,
        doc_id: str,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Advanced chunking with semantic section splitting.
        
        Creates chunks based on semantic sections (delimited by H1/H2 headings)
        rather than just character count.
        """
        elements = self._parse_markdown(markdown_content)
        
        # Group by sections (H1/H2 delimited)
        sections = []
        current_section = []
        current_section_size = 0
        
        for element in elements:
            # Start new section on H1 or H2
            if (element.element_type == MarkdownElementType.HEADING and 
                element.level <= 2 and 
                current_section):
                sections.append(current_section)
                current_section = [element]
                current_section_size = len(element.content)
            else:
                current_section.append(element)
                current_section_size += len(element.content)
                
                # Split section if it gets too large
                if current_section_size > self.chunk_size * 2:
                    sections.append(current_section)
                    current_section = []
                    current_section_size = 0
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # Convert sections to chunks
        documents = []
        for i, section in enumerate(sections):
            content = '\n\n'.join(e.content for e in section)
            
            # If section is too large, use regular chunking
            if len(content) > self.chunk_size * 1.5:
                sub_chunks = self._group_elements_into_chunks(section, markdown_content)
                for j, sub_chunk in enumerate(sub_chunks):
                    metadata = {
                        'doc_id': doc_id,
                        'chunk_index': len(documents),
                        'total_chunks': None,  # Will update later
                        'chunk_type': 'markdown',
                        'source': f"md_{doc_id}",
                        'start_line': sub_chunk.start_line,
                        'end_line': sub_chunk.end_line,
                        'section_index': i,
                        'sub_chunk_index': j,
                        'element_types': list(set(e.element_type.value for e in sub_chunk.elements)),
                        'headings_context': sub_chunk.headings_context,
                        'has_images': any(e.element_type == MarkdownElementType.IMAGE for e in sub_chunk.elements),
                        'has_tables': any(e.element_type == MarkdownElementType.TABLE for e in sub_chunk.elements),
                    }
                    
                    if source_metadata:
                        metadata.update(source_metadata)

                    documents.append(Document(
                        page_content=sub_chunk.content,
                        metadata=metadata
                    ))
            else:
                # Use section as single chunk
                start_line = section[0].start_line if section else 0
                end_line = section[-1].end_line if section else 0
                
                # Extract headings
                headings = [e.content for e in section if e.element_type == MarkdownElementType.HEADING]
                
                metadata = {
                    'doc_id': doc_id,
                    'chunk_index': len(documents),
                    'total_chunks': None,
                    'chunk_type': 'markdown_section',
                    'source': f"md_{doc_id}",
                    'start_line': start_line,
                    'end_line': end_line,
                    'section_index': i,
                    'element_types': list(set(e.element_type.value for e in section)),
                    'headings_context': headings,
                    'has_images': any(e.element_type == MarkdownElementType.IMAGE for e in section),
                    'has_tables': any(e.element_type == MarkdownElementType.TABLE for e in section),
                    'is_semantic_section': True
                }
                
                if source_metadata:
                    metadata.update(source_metadata)
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        # Update total chunks
        for doc in documents:
            doc.metadata['total_chunks'] = len(documents)
        
        return documents


# Convenience function
def chunk_markdown_document(
    markdown_content: str,
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    semantic_splitting: bool = True,
    source_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Convenience function to chunk a Markdown document.
    
    Args:
        markdown_content: Full Markdown content
        doc_id: Document identifier
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        semantic_splitting: Use semantic section-based splitting
        source_metadata: Additional metadata
        
    Returns:
        List of LangChain Document objects
    """
    chunker = MarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_headings=True,
        respect_tables=True,
        include_heading_context=True
    )
    
    if semantic_splitting:
        return chunker.chunk_with_semantic_splitting(
            markdown_content, 
            doc_id, 
            source_metadata
        )
    else:
        return chunker.chunk_markdown(
            markdown_content, 
            doc_id, 
            source_metadata
        )
