#!/usr/bin/env python3
"""
Serializers: ExtractedDocument <-> Markdown+frontmatter, ExtractedDocument <-> JSON.

One canonical model, two projections (per "entrambi paritari" decision):
JSON is lossless; Markdown uses an HTML-comment block marker to reconstruct
per-block structure (page/heading/type) on parse, but stays clean (no markers)
when there is a single untyped block.
"""
from tilellm.modules.ingestion.export.models import Block, ExtractedDocument
from tilellm.modules.ingestion.export.serializers import (
    from_json,
    from_md,
    to_json,
    to_md,
)


class TestJsonRoundTrip:
    def test_minimal(self):
        doc = ExtractedDocument(type="document", blocks=[Block(content="hi")])
        assert from_json(to_json(doc)) == doc

    def test_full_roundtrip(self):
        doc = ExtractedDocument(
            type="PDF Document",
            title="Capitolato",
            description="desc",
            resource="https://x/y.pdf",
            tags=["a", "b"],
            timestamp="2026-07-22T10:00:00Z",
            extra={"lot_id": "L1", "nested": {"k": 1}},
            blocks=[
                Block(content="p1", block_type="page", page=1, order=0),
                Block(content="p2", block_type="page", page=2, order=1),
            ],
        )
        assert from_json(to_json(doc)) == doc

    def test_to_json_is_valid_json_string(self):
        import json
        doc = ExtractedDocument(type="document", blocks=[Block(content="hi")])
        parsed = json.loads(to_json(doc))
        assert parsed["type"] == "document"


class TestMarkdownRoundTrip:
    def test_single_block_no_structural_markers(self):
        """Plain single-block docs (txt passthrough) must stay human-clean."""
        doc = ExtractedDocument(type="document", title="Note", blocks=[Block(content="Ciao mondo")])
        md = to_md(doc)
        assert "<!--" not in md
        assert "Ciao mondo" in md
        assert md.startswith("---\n")

    def test_frontmatter_contains_okf_fields(self):
        doc = ExtractedDocument(
            type="PDF Document", title="T", description="D",
            resource="https://x/y.pdf", tags=["a", "b"],
            timestamp="2026-07-22T10:00:00Z",
            blocks=[Block(content="body")],
        )
        md = to_md(doc)
        assert "type: PDF Document" in md
        assert "title: T" in md
        assert "resource: https://x/y.pdf" in md
        assert "- a" in md and "- b" in md

    def test_extra_keys_serialized_in_frontmatter(self):
        doc = ExtractedDocument(type="document", extra={"lot_id": "L1"}, blocks=[Block(content="x")])
        md = to_md(doc)
        assert "lot_id: L1" in md

    def test_multi_block_roundtrip_preserves_structure(self):
        doc = ExtractedDocument(
            type="PDF Document",
            title="Capitolato",
            blocks=[
                Block(content="Pagina uno", block_type="page", page=1, order=0),
                Block(content="Pagina due", block_type="page", page=2, order=1,
                      heading_path="Sezione 2"),
            ],
        )
        parsed = from_md(to_md(doc))
        assert parsed.type == doc.type
        assert parsed.title == doc.title
        assert [b.content for b in parsed.blocks] == ["Pagina uno", "Pagina due"]
        assert [b.page for b in parsed.blocks] == [1, 2]
        assert parsed.blocks[1].heading_path == "Sezione 2"
        assert [b.block_type for b in parsed.blocks] == ["page", "page"]

    def test_position_roundtrip_for_non_paginated_formats(self):
        doc = ExtractedDocument(
            type="Word Document",
            blocks=[
                Block(content="Paragrafo 1", heading_path="Intro", position=0, order=0),
                Block(content="Paragrafo 2", heading_path="Sezione 2", position=5, order=1),
            ],
        )
        parsed = from_md(to_md(doc))
        assert [b.position for b in parsed.blocks] == [0, 5]
        assert all(b.page is None for b in parsed.blocks)

    def test_from_md_single_block_no_markers(self):
        raw = "---\ntype: document\ntitle: Note\n---\n\nCiao mondo\n"
        doc = from_md(raw)
        assert doc.type == "document"
        assert doc.title == "Note"
        assert len(doc.blocks) == 1
        assert doc.blocks[0].content.strip() == "Ciao mondo"
        assert doc.blocks[0].page is None

    def test_from_md_missing_frontmatter_raises(self):
        import pytest
        with pytest.raises(ValueError):
            from_md("Just plain text, no frontmatter")

    def test_from_md_missing_type_raises(self):
        import pytest
        raw = "---\ntitle: Note\n---\n\nBody\n"
        with pytest.raises(ValueError):
            from_md(raw)

    def test_tags_and_extra_survive_md_roundtrip(self):
        doc = ExtractedDocument(
            type="document", tags=["x", "y"], extra={"lot_id": "L1"},
            blocks=[Block(content="body")],
        )
        parsed = from_md(to_md(doc))
        assert parsed.tags == ["x", "y"]
        assert parsed.extra == {"lot_id": "L1"}
