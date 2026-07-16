"""
Part C — native Docling markdown.

The MarkdownExtractionAgent used to hand-rebuild the document from parsed
elements, which flattened lists/inline formatting and grouped elements by
type (breaking reading order). These tests pin the contract of the new
native-export assembly: docling's own markdown must survive verbatim, pages
must stay in reading order with the "## Page N" markers the chunker depends
on, and the LLM image/table descriptions must not be dropped.
"""

from tilellm.modules.pdf_ocr.services.markdown_extraction_agent import (
    PAGE_BREAK,
    split_segment_pages,
    assemble_markdown,
)


class TestSplitSegmentPages:
    def test_single_page_no_break(self):
        assert split_segment_pages("Hello\n\n- a\n- b", page_offset=0) == [
            (1, "Hello\n\n- a\n- b")
        ]

    def test_multi_page_numbered_with_offset(self):
        md = f"P1{PAGE_BREAK}P2{PAGE_BREAK}P3"
        pages = split_segment_pages(md, page_offset=20)
        assert [p[0] for p in pages] == [21, 22, 23]
        assert pages[1][1] == "P2"

    def test_strips_break_whitespace(self):
        md = f"A\n{PAGE_BREAK}\nB"
        assert [body for _, body in split_segment_pages(md, page_offset=0)] == ["A", "B"]


class TestAssembleMarkdown:
    def test_preserves_native_markdown_verbatim(self):
        # The core regression: docling list + inline + table survive unflattened.
        body = (
            "## Heading\n\nIntro **bold**.\n\n- item 1\n- item 2\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |"
        )
        out = assemble_markdown("doc1", [(1, body)], image_notes=[], table_notes=[])
        assert "- item 1\n- item 2" in out
        assert "**bold**" in out
        assert "| a | b |" in out
        assert "## Page 1" in out

    def test_reading_order_pages_in_order(self):
        out = assemble_markdown(
            "d", [(1, "first"), (2, "second"), (3, "third")], [], []
        )
        assert out.index("first") < out.index("second") < out.index("third")
        assert out.index("## Page 1") < out.index("## Page 2") < out.index("## Page 3")

    def test_image_and_table_descriptions_included(self):
        out = assemble_markdown(
            "d",
            [(1, "body")],
            image_notes=[(1, "A bar chart of sales")],
            table_notes=[(2, "Quarterly revenue by region")],
        )
        assert "A bar chart of sales" in out
        assert "Quarterly revenue by region" in out

    def test_no_description_sections_when_empty(self):
        out = assemble_markdown("d", [(1, "body")], [], [])
        assert "Image description" not in out
        assert "Table description" not in out

    def test_page_markers_match_chunker_regex(self):
        import re

        out = assemble_markdown("d", [(3, "x"), (4, "y")], [], [])
        found = [int(m) for m in re.findall(r"^##\s+Page\s+(\d+)", out, re.MULTILINE)]
        assert found == [3, 4]
