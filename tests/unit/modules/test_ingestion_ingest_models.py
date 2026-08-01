#!/usr/bin/env python3
"""
IngestMdRequest — POST /api/ingest/md request shape.

Exactly one of md/md_url/json/json_url must be provided (mirrors the
compliance-checker requirements-source pattern). hybrid/sparse_encoder mirror
ItemSingle so the caller's "I might want sparse embeddings" need is covered.
"""
import pytest
from pydantic import ValidationError

from tilellm.models.vector_store import Engine
from tilellm.modules.ingestion.ingest.models import IngestMdRequest

_ENGINE = {"name": "pinecone", "type": "serverless", "apikey": "k", "vector_size": 1536, "index_name": "idx"}


def _req(**over):
    kw = dict(id="doc1", namespace="ns", engine=_ENGINE, md="---\ntype: document\n---\n\nbody\n")
    kw.update(over)
    return IngestMdRequest(**kw)


class TestSourceValidation:
    def test_md_only_ok(self):
        req = _req()
        assert req.md is not None

    def test_no_source_raises(self):
        with pytest.raises(ValidationError):
            _req(md=None)

    def test_two_sources_raises(self):
        with pytest.raises(ValidationError):
            _req(md_url="https://x/doc.md")

    def test_json_source_ok(self):
        req = _req(md=None, json_content='{"type": "document", "blocks": []}')
        assert req.json_content is not None


class TestDefaults:
    def test_hybrid_defaults_false(self):
        assert _req().hybrid is False

    def test_sparse_encoder_default(self):
        assert _req().sparse_encoder == "splade"

    def test_engine_required(self):
        with pytest.raises(ValidationError):
            IngestMdRequest(id="d", namespace="ns", md="---\ntype: document\n---\n\nb\n")
