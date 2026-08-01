#!/usr/bin/env python3
"""
ExtractedDocument / Block — canonical model for the export/md redesign.

Frontmatter vocabulary follows OKF (type/title/description/resource/tags/timestamp
+ free extension keys); structural fields (page/heading_path/block_type) live on
each Block and are treated as derived/volatile, never mixed into frontmatter.
Decisions: memory/ingestion_md_redesign.md
"""
import pytest
from pydantic import ValidationError

from tilellm.modules.ingestion.export.models import Block, ExtractedDocument


class TestBlock:
    def test_defaults(self):
        b = Block(content="hello")
        assert b.block_type == "text"
        assert b.page is None
        assert b.heading_path is None
        assert b.order == 0

    def test_explicit_structural_fields(self):
        b = Block(content="Row 1", block_type="table", page=3, heading_path="Sec/Sub", order=2)
        assert (b.block_type, b.page, b.heading_path, b.order) == ("table", 3, "Sec/Sub", 2)

    def test_position_defaults_none(self):
        """Structural position (e.g. DOCX paragraph index) — distinct from `page`
        (a literal page number, only meaningful for paginated formats like PDF)."""
        assert Block(content="x").position is None

    def test_position_explicit(self):
        b = Block(content="Paragrafo 3", heading_path="Sezione 2", position=12)
        assert b.position == 12
        assert b.page is None  # DOCX has no real page number — must stay unset, not faked


class TestExtractedDocument:
    def test_requires_type(self):
        with pytest.raises(ValidationError):
            ExtractedDocument()

    def test_minimal(self):
        doc = ExtractedDocument(type="document")
        assert doc.type == "document"
        assert doc.title is None
        assert doc.tags == []
        assert doc.blocks == []
        assert doc.extra == {}

    def test_full_frontmatter(self):
        doc = ExtractedDocument(
            type="PDF Document",
            title="Capitolato",
            description="Requisiti di gara",
            resource="https://storage.example.com/gara.pdf",
            tags=["gara", "lotto1"],
            timestamp="2026-07-22T10:00:00Z",
            extra={"lot_id": "L1"},
            blocks=[Block(content="pagina 1", page=1)],
        )
        assert doc.tags == ["gara", "lotto1"]
        assert doc.extra["lot_id"] == "L1"
        assert len(doc.blocks) == 1

    def test_extra_preserves_unknown_keys(self):
        """OKF: producers may append arbitrary key/value pairs; must round-trip."""
        doc = ExtractedDocument(type="document", extra={"custom_field": 42, "nested": {"a": 1}})
        assert doc.extra == {"custom_field": 42, "nested": {"a": 1}}

    def test_body_text_concatenates_blocks_in_order(self):
        doc = ExtractedDocument(
            type="document",
            blocks=[
                Block(content="second", order=1),
                Block(content="first", order=0),
            ],
        )
        # order field controls concatenation regardless of list insertion order
        assert doc.body_text() == "first\n\nsecond"
