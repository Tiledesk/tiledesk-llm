"""
Document Structure Extractor - Phase 1 Implementation
Extracts hierarchical structure from Docling parsed documents.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DocumentSection:
    """Represents a document section with hierarchy."""

    def __init__(
        self,
        section_id: str,
        title: str,
        level: int,
        page: int,
        parent_id: Optional[str] = None
    ):
        self.section_id = section_id
        self.title = title
        self.level = level
        self.page = page
        self.parent_id = parent_id
        self.children: List[str] = []
        self.elements: List[str] = []  # IDs of paragraphs, tables, images

    def to_dict(self) -> Dict:
        return {
            'section_id': self.section_id,
            'title': self.title,
            'level': self.level,
            'page': self.page,
            'parent_id': self.parent_id,
            'children': self.children,
            'elements': self.elements
        }


class DocumentStructureExtractor:
    """
    Extracts hierarchical structure from Docling parsed documents.

    Capabilities:
    1. Extract document outline (Table of Contents)
    2. Build section hierarchy
    3. Identify reading order
    4. Extract cross-references (Figure X, Table Y, Section Z)
    5. Map elements to sections
    """

    def __init__(self):
        self.sections: Dict[str, DocumentSection] = {}
        self.cross_references: Dict[str, List[str]] = defaultdict(list)
        self.reading_order: List[str] = []

    def extract_hierarchy(
        self,
        doc_id: str,
        structured_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for structure extraction.

        Args:
            doc_id: Document identifier
            structured_content: Output from ProductionDocumentProcessor

        Returns:
            Dict with:
            - outline: List of sections with hierarchy
            - sections: Dict of section_id -> DocumentSection
            - cross_refs: Dict of element_id -> List[referenced_element_ids]
            - reading_order: List of element IDs in reading order
        """
        logger.info(f"Extracting structure for document {doc_id}")

        # 1. Extract sections from text elements
        text_elements = structured_content.get('text_elements', [])
        self._extract_sections(text_elements, doc_id)

        # 2. Assign elements to sections
        self._assign_elements_to_sections(
            text_elements,
            structured_content.get('tables', []),
            structured_content.get('images', []),
            structured_content.get('formulas', [])
        )

        # 3. Extract cross-references
        self._extract_cross_references(text_elements)

        # 4. Build reading order
        self._build_reading_order()

        # 5. Generate outline
        outline = self._generate_outline()

        return {
            'outline': outline,
            'sections': {sid: sec.to_dict() for sid, sec in self.sections.items()},
            'cross_refs': dict(self.cross_references),
            'reading_order': self.reading_order,
            'metadata': {
                'num_sections': len(self.sections),
                'num_cross_refs': sum(len(refs) for refs in self.cross_references.values()),
                'depth': self._calculate_max_depth()
            }
        }

    def _extract_sections(
        self,
        text_elements: List[Dict],
        doc_id: str
    ):
        """
        Extract sections by identifying headings.

        Heuristics:
        1. Element type == 'heading' or 'title'
        2. Numbering pattern (1., 1.1, 1.1.1)
        3. Font size/weight (if available in bbox metadata)
        4. All caps short text
        """
        current_section_stack: List[Tuple[int, str]] = []  # [(level, section_id)]

        for idx, element in enumerate(text_elements):
            text = element.get('text', '').strip()
            element_type = element.get('type', 'text')

            # Check if this is a heading
            is_heading, level = self._is_heading(text, element_type)

            if is_heading:
                section_id = element.get('id', f"{doc_id}_section_{idx}")
                page = element.get('page', 0)

                # Determine parent based on level
                parent_id = None
                if level > 1:
                    # Find parent section (last section with level < current level)
                    for stack_level, stack_section_id in reversed(current_section_stack):
                        if stack_level < level:
                            parent_id = stack_section_id
                            break

                # Create section
                section = DocumentSection(
                    section_id=section_id,
                    title=text,
                    level=level,
                    page=page,
                    parent_id=parent_id
                )

                self.sections[section_id] = section

                # Update parent's children
                if parent_id and parent_id in self.sections:
                    self.sections[parent_id].children.append(section_id)

                # Update stack
                # Remove sections at same or lower level
                current_section_stack = [
                    (l, sid) for l, sid in current_section_stack if l < level
                ]
                current_section_stack.append((level, section_id))

                logger.debug(f"Found section: {text} (level={level}, parent={parent_id})")

    def _is_heading(
        self,
        text: str,
        element_type: str
    ) -> Tuple[bool, int]:
        """
        Determine if text is a heading and its level.

        Returns:
            (is_heading: bool, level: int)
        """
        # Method 1: Explicit type
        if element_type in ['heading', 'title', 'section-header']:
            # Try to infer level from numbering
            level = self._infer_level_from_numbering(text)
            return True, level

        # Method 2: Numbering pattern
        level = self._infer_level_from_numbering(text)
        if level > 0:
            return True, level

        # Method 3: Short ALL CAPS text (likely heading)
        if len(text) < 100 and text.isupper() and not text.endswith('.'):
            return True, 1

        # Method 4: Common heading keywords
        heading_patterns = [
            r'^abstract\s*$',
            r'^introduction\s*$',
            r'^background\s*$',
            r'^methods?\s*$',
            r'^results?\s*$',
            r'^discussion\s*$',
            r'^conclusion\s*$',
            r'^references\s*$',
            r'^appendix\s*[a-z]?\s*$',
        ]

        text_lower = text.lower().strip()
        for pattern in heading_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True, 1

        return False, 0

    def _infer_level_from_numbering(self, text: str) -> int:
        """
        Infer section level from numbering pattern.

        Examples:
        - "1. Introduction" -> level 1
        - "1.1 Background" -> level 2
        - "1.1.1 Context" -> level 3
        - "2.3.4.1 Details" -> level 4
        """
        # Pattern: digits.digits.digits... followed by space or )
        pattern = r'^(\d+(?:\.\d+)*)[.\)\s]'
        match = re.match(pattern, text)

        if match:
            numbering = match.group(1)
            level = numbering.count('.') + 1
            return level

        # Roman numerals (I, II, III, IV, V, ...)
        roman_pattern = r'^([IVX]+)[.\)\s]'
        if re.match(roman_pattern, text):
            return 1

        # Letters (A, B, C, ... or a, b, c, ...)
        letter_pattern = r'^([A-Za-z])[.\)\s]'
        if re.match(letter_pattern, text):
            return 2

        return 0

    def _assign_elements_to_sections(
        self,
        text_elements: List[Dict],
        tables: List[Dict],
        images: List[Dict],
        formulas: List[Dict]
    ):
        """
        Assign all document elements to their containing sections.
        """
        # Build section by page mapping
        sections_by_page: Dict[int, List[str]] = defaultdict(list)
        for section_id, section in self.sections.items():
            sections_by_page[section.page].append(section_id)

        # Sort sections by page and reading order
        for page_sections in sections_by_page.values():
            page_sections.sort(key=lambda sid: self.sections[sid].section_id)

        current_section = None

        # Process all elements in order
        all_elements = []
        all_elements.extend([('text', e) for e in text_elements])
        all_elements.extend([('table', t) for t in tables])
        all_elements.extend([('image', i) for i in images])
        all_elements.extend([('formula', f) for f in formulas])

        # Sort by page, then by bbox position
        all_elements.sort(key=lambda x: (x[1].get('page', 0), x[1].get('bbox', [0, 0, 0, 0])[1]))

        for element_type, element in all_elements:
            element_id = element.get('id')
            page = element.get('page', 0)

            # Check if this element is a section heading
            if element_id in self.sections:
                current_section = element_id
                continue

            # Assign to current section
            if current_section:
                self.sections[current_section].elements.append(element_id)
            elif page in sections_by_page and sections_by_page[page]:
                # Assign to first section on page
                self.sections[sections_by_page[page][0]].elements.append(element_id)

    def _extract_cross_references(self, text_elements: List[Dict]):
        """
        Extract cross-references like "Figure 3", "Table 2", "Section 4.1".
        """
        ref_patterns = {
            'figure': r'(?:Figure|Fig\.?|FIGURE)\s+(\d+)',
            'table': r'(?:Table|TABLE|Tbl\.?)\s+(\d+)',
            'section': r'(?:Section|Sec\.?|SECTION|§)\s+([\d.]+)',
            'equation': r'(?:Equation|Eq\.?|EQUATION)\s+\((\d+)\)',
            'appendix': r'(?:Appendix|APPENDIX)\s+([A-Z])',
        }

        for element in text_elements:
            element_id = element.get('id')
            text = element.get('text', '')

            for ref_type, pattern in ref_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    ref_target = f"{ref_type}_{match.group(1)}"
                    self.cross_references[element_id].append(ref_target)

                    logger.debug(f"Found cross-ref: {element_id} -> {ref_target}")

    def _build_reading_order(self):
        """
        Build reading order of all elements.
        Simple implementation: sections in order, then their elements.
        """
        def traverse_section(section_id: str):
            section = self.sections[section_id]

            # Add section itself
            self.reading_order.append(section_id)

            # Add section's elements
            self.reading_order.extend(section.elements)

            # Recursively traverse children
            for child_id in section.children:
                traverse_section(child_id)

        # Start from root sections (no parent)
        root_sections = [
            sid for sid, sec in self.sections.items()
            if sec.parent_id is None
        ]

        # Sort roots by page and section_id
        root_sections.sort(
            key=lambda sid: (self.sections[sid].page, sid)
        )

        for root_section_id in root_sections:
            traverse_section(root_section_id)

    def _generate_outline(self) -> List[Dict]:
        """
        Generate hierarchical outline (TOC).
        """
        def build_outline_entry(section_id: str) -> Dict:
            section = self.sections[section_id]
            entry = {
                'section_id': section_id,
                'title': section.title,
                'level': section.level,
                'page': section.page,
                'num_elements': len(section.elements)
            }

            if section.children:
                entry['children'] = [
                    build_outline_entry(child_id)
                    for child_id in section.children
                ]

            return entry

        # Build from roots
        root_sections = [
            sid for sid, sec in self.sections.items()
            if sec.parent_id is None
        ]

        root_sections.sort(
            key=lambda sid: (self.sections[sid].page, sid)
        )

        outline = [build_outline_entry(sid) for sid in root_sections]

        return outline

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of section hierarchy."""
        def calc_depth(section_id: str) -> int:
            section = self.sections[section_id]
            if not section.children:
                return 1
            return 1 + max(calc_depth(child) for child in section.children)

        root_sections = [
            sid for sid, sec in self.sections.items()
            if sec.parent_id is None
        ]

        if not root_sections:
            return 0

        return max(calc_depth(sid) for sid in root_sections)

    def get_section_path(self, element_id: str) -> str:
        """
        Get section path for an element.

        Example: "Introduction > Background > Context"
        """
        # Find section containing this element
        containing_section = None
        for section_id, section in self.sections.items():
            if element_id in section.elements or element_id == section_id:
                containing_section = section_id
                break

        if not containing_section:
            return ""

        # Build path from root to this section
        path_parts = []
        current_section_id = containing_section

        while current_section_id:
            section = self.sections[current_section_id]
            path_parts.insert(0, section.title)
            current_section_id = section.parent_id

        return " > ".join(path_parts)

    def get_related_elements(
        self,
        element_id: str,
        include_same_section: bool = True
    ) -> Dict[str, List[str]]:
        """
        Get related elements (tables, images referenced in cross-refs).

        Returns:
            Dict with keys: 'tables', 'images', 'sections', 'same_section_elements'
        """
        related = {
            'tables': [],
            'images': [],
            'sections': [],
            'formulas': [],
            'same_section_elements': []
        }

        # 1. Get cross-referenced elements
        refs = self.cross_references.get(element_id, [])
        for ref in refs:
            if ref.startswith('table_'):
                related['tables'].append(ref)
            elif ref.startswith('figure_'):
                related['images'].append(ref)
            elif ref.startswith('section_'):
                related['sections'].append(ref)
            elif ref.startswith('equation_'):
                related['formulas'].append(ref)

        # 2. Get elements in same section
        if include_same_section:
            for section_id, section in self.sections.items():
                if element_id in section.elements:
                    related['same_section_elements'] = section.elements.copy()
                    break

        return related


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example: Process a mock document
    doc_id = "paper_123"

    mock_content = {
        'text_elements': [
            {'id': f'{doc_id}_text_0', 'text': 'Abstract', 'type': 'heading', 'page': 0},
            {'id': f'{doc_id}_text_1', 'text': 'This paper presents...', 'type': 'text', 'page': 0},
            {'id': f'{doc_id}_text_2', 'text': '1. Introduction', 'type': 'heading', 'page': 1},
            {'id': f'{doc_id}_text_3', 'text': 'As shown in Figure 1...', 'type': 'text', 'page': 1},
            {'id': f'{doc_id}_text_4', 'text': '1.1 Background', 'type': 'heading', 'page': 1},
            {'id': f'{doc_id}_text_5', 'text': 'See Table 2 for details...', 'type': 'text', 'page': 2},
            {'id': f'{doc_id}_text_6', 'text': '2. Methods', 'type': 'heading', 'page': 3},
        ],
        'tables': [
            {'id': f'{doc_id}_table_2', 'page': 2}
        ],
        'images': [
            {'id': f'{doc_id}_image_1', 'page': 1}
        ],
        'formulas': []
    }

    # Extract structure
    extractor = DocumentStructureExtractor()
    result = extractor.extract_hierarchy(doc_id, mock_content)

    # Print results
    print("\n=== Document Outline ===")
    import json
    print(json.dumps(result['outline'], indent=2))

    print("\n=== Cross References ===")
    print(json.dumps(result['cross_refs'], indent=2))

    print("\n=== Metadata ===")
    print(json.dumps(result['metadata'], indent=2))

    # Test section path
    print("\n=== Section Paths ===")
    for element_id in [f'{doc_id}_text_3', f'{doc_id}_text_5']:
        path = extractor.get_section_path(element_id)
        print(f"{element_id}: {path}")

    # Test related elements
    print("\n=== Related Elements ===")
    element_id = f'{doc_id}_text_3'
    related = extractor.get_related_elements(element_id)
    print(f"{element_id}:")
    print(json.dumps(related, indent=2))
