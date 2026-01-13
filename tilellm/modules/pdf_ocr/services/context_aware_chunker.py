"""
Context Aware Chunker
Chunking strategy that preserves document hierarchy and cross-references.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ContextAwareChunker:
    """
    Chunking strategy that preserves:
    - Document hierarchy (Section > Subsection)
    - Reading order
    - Cross-references to tables, images, and other sections
    - Token limits with overlap
    """

    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_with_structure(
        self,
        doc_id: str,
        text_elements: List[Dict[str, Any]],
        structure: Dict[str, Any]
    ) -> List[Document]:
        """
        Create enriched chunks from text elements and document structure.
        Chunks are created by combining elements within the same section
        until max_tokens limit is reached.
        """
        chunks = []
        
        # Helper to get section path
        sections = structure.get('sections', {})
        
        def get_path(element_id: str) -> str:
            containing_section_id = None
            for sid, sec in sections.items():
                if element_id in sec.get('elements', []) or element_id == sid:
                    containing_section_id = sid
                    break
            
            if not containing_section_id:
                return "Root"
                
            path_parts = []
            curr_id = containing_section_id
            while curr_id:
                sec = sections.get(curr_id)
                if not sec: break
                path_parts.insert(0, sec.get('title', ''))
                curr_id = sec.get('parent_id')
            
            return " > ".join(path_parts)

        # Get cross references
        all_cross_refs = structure.get('cross_refs', {})

        # Group elements by section
        sections_elements = {}
        for element in text_elements:
            element_id = element.get('id')
            section_id = self._find_containing_section(element_id, sections)
            
            if section_id not in sections_elements:
                sections_elements[section_id] = []
            sections_elements[section_id].append(element)

        # Process each section
        chunk_idx = 0
        for section_id, elements in sections_elements.items():
            section_path = get_path(section_id)
            section_text_parts = []
            section_metadata = {
                "doc_id": doc_id,
                "section_id": section_id,
                "section_path": section_path,
                "type": "text",
                "related_tables": [],
                "related_images": [],
                "related_sections": [],
                "chunk_type": "context_aware"
            }
            
            current_tokens = 0
            current_elements = []
            
            for element in elements:
                element_id = element.get('id')
                text = element.get('text', '').strip()
                if not text:
                    continue
                
                # Estimate tokens (rough approximation: ~4 chars per token)
                element_tokens = len(text) // 4
                
                # Check if adding this element would exceed max_tokens
                if current_tokens + element_tokens > self.max_tokens and current_elements:
                    # Create chunk from accumulated elements
                    chunk_text = "\n\n".join(current_elements)
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={
                            **section_metadata,
                            "chunk_index": chunk_idx,
                            "element_ids": [e.get('id') for e in section_text_parts],
                            "page": section_text_parts[0].get('page', 0) if section_text_parts else 0
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    
                    # Start new chunk with overlap
                    current_elements = []
                    current_tokens = 0
                    section_text_parts = []
                
                # Add element to current chunk
                current_elements.append(text)
                current_tokens += element_tokens
                section_text_parts.append(element)
                
                # Collect cross-references
                element_cross_refs = all_cross_refs.get(element_id, [])
                text_refs = self._extract_cross_refs(text)
                combined_refs = list(set(element_cross_refs + text_refs))
                
                section_metadata["related_tables"].extend([r for r in combined_refs if 'table' in r.lower()])
                section_metadata["related_images"].extend([r for r in combined_refs if 'figure' in r.lower() or 'image' in r.lower()])
                section_metadata["related_sections"].extend([r for r in combined_refs if 'section' in r.lower()])
            
            # Create final chunk for this section
            if current_elements:
                chunk_text = "\n\n".join(current_elements)
                # Deduplicate and limit related elements
                section_metadata["related_tables"] = list(set(section_metadata["related_tables"]))[:10]
                section_metadata["related_images"] = list(set(section_metadata["related_images"]))[:10]
                section_metadata["related_sections"] = list(set(section_metadata["related_sections"]))[:10]
                
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **section_metadata,
                        "chunk_index": chunk_idx,
                        "element_ids": [e.get('id') for e in section_text_parts],
                        "page": section_text_parts[0].get('page', 0) if section_text_parts else 0
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1

        logger.info(f"Created {len(chunks)} context-aware chunks for document {doc_id}")
        return chunks

    def _find_containing_section(
        self,
        element_id: str,
        sections: Dict[str, Any]
    ) -> Optional[str]:
        """Find the section that contains this element."""
        for sid, sec in sections.items():
            if element_id in sec.get('elements', []):
                return sid
        return None

    def _extract_cross_refs(self, text: str) -> List[str]:
        """
        Extract references like "Figure 3", "Table 2", "Section 4.2".
        """
        ref_patterns = [
            (r'Figure\s+(\d+)', 'figure'),
            (r'Fig\.\s+(\d+)', 'figure'),
            (r'Table\s+(\d+)', 'table'),
            (r'Section\s+([\d.]+)', 'section'),
            (r'Equation\s+\((\d+)\)', 'equation')
        ]
        
        refs = []
        for pattern, ref_type in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                refs.append(f"{ref_type}_{m}")
        return refs
