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
        """
        chunks = []
        
        # Helper to get section path
        sections = structure.get('sections', {})
        
        def get_path(element_id: str) -> str:
            # Find which section contains this element
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

        for element in text_elements:
            element_id = element.get('id')
            text = element.get('text', '').strip()
            
            if not text:
                continue
                
            # Get structural context
            section_path = get_path(element_id)
            
            # Get explicit cross-references from structure
            element_cross_refs = all_cross_refs.get(element_id, [])
            
            # Detect new cross-references in text (regex)
            text_refs = self._extract_cross_refs(text)
            combined_refs = list(set(element_cross_refs + text_refs))
            
            # Separate refs by type
            tables = [r for r in combined_refs if 'table' in r.lower()]
            images = [r for r in combined_refs if 'figure' in r.lower() or 'image' in r.lower()]
            other_sections = [r for r in combined_refs if 'section' in r.lower()]

            # Create Document object
            # Note: For very long elements, we might still need to split them, 
            # but Docling usually gives us reasonably sized paragraphs.
            
            metadata = {
                "doc_id": doc_id,
                "element_id": element_id,
                "page": element.get('page', 0),
                "type": element.get('type', 'text'),
                "section_path": section_path,
                "related_tables": tables,
                "related_images": images,
                "related_sections": other_sections,
                "bbox": element.get('bbox'),
                "chunk_type": "context_aware"
            }
            
            chunk = Document(
                page_content=text,
                metadata=metadata
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} context-aware chunks for document {doc_id}")
        return chunks

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
