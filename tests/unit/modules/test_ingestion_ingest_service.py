#!/usr/bin/env python3
"""
ingest/service.py — frontmatter + blocks -> List[Document] (metadata built by us,
no additional_metadata gap, see memory/ingestion_md_redesign.md), then
repo.aadd_documents(). hybrid is a single sparse_encoder parameter, not two
methods (mirrors docx_processor.py's established pattern).
"""
from unittest.mock import AsyncMock, patch

import pytest

from tilellm.models.vector_store import Engine
from tilellm.modules.ingestion.export.models import Block, ExtractedDocument
from tilellm.modules.ingestion.ingest.models import IngestMdRequest
from tilellm.modules.ingestion.ingest.service import _build_documents, ingest_document

_ENGINE = {"name": "pinecone", "type": "serverless", "apikey": "k", "vector_size": 1536, "index_name": "idx"}


def _req(**over):
    kw = dict(id="doc1", namespace="ns", engine=_ENGINE, md="---\ntype: document\n---\n\nbody\n")
    kw.update(over)
    return IngestMdRequest(**kw)


class TestBuildDocumentsMetadata:
    def test_base_metadata_on_every_chunk(self):
        doc = ExtractedDocument(type="Text Document", blocks=[Block(content="short text")])
        req = _req()
        docs = _build_documents(doc, req)
        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["id"] == "doc1"
        assert meta["metadata_id"] == "doc1"
        assert meta["namespace"] == "ns"

    def test_frontmatter_fields_propagated(self):
        doc = ExtractedDocument(
            type="PDF Document", title="Capitolato", description="Requisiti",
            resource="https://x/y.pdf", timestamp="2026-07-22T10:00:00Z",
            blocks=[Block(content="text")],
        )
        docs = _build_documents(doc, _req())
        meta = docs[0].metadata
        assert meta["title"] == "Capitolato"
        assert meta["description"] == "Requisiti"
        assert meta["resource"] == "https://x/y.pdf"
        assert meta["doc_type"] == "PDF Document"

    def test_extra_keys_serialized_for_metadata_store_compat(self):
        """Nested/non-scalar extra values must be JSON-stringified (Pinecone/Qdrant/Milvus
        metadata compatibility — mirrors the col_names-must-be-str precedent in table_chunker."""
        doc = ExtractedDocument(
            type="document", extra={"lot_id": "L1", "nested": {"a": 1}},
            blocks=[Block(content="text")],
        )
        docs = _build_documents(doc, _req())
        meta = docs[0].metadata
        assert meta["lot_id"] == "L1"
        assert isinstance(meta["nested"], str)

    def test_request_tags_override_doc_tags(self):
        doc = ExtractedDocument(type="document", tags=["from-doc"], blocks=[Block(content="text")])
        docs = _build_documents(doc, _req(tags=["from-request"]))
        assert docs[0].metadata["tags"] == ["from-request"]

    def test_doc_tags_used_when_request_tags_absent(self):
        doc = ExtractedDocument(type="document", tags=["from-doc"], blocks=[Block(content="text")])
        docs = _build_documents(doc, _req())
        assert docs[0].metadata["tags"] == ["from-doc"]


class TestBuildDocumentsChunking:
    def test_text_block_chunked_by_size(self):
        long_text = "word " * 500  # forces the splitter to produce >1 chunk
        doc = ExtractedDocument(type="document", blocks=[Block(content=long_text, page=3)])
        docs = _build_documents(doc, _req(chunk_size=200, chunk_overlap=20))
        assert len(docs) > 1
        assert all(d.metadata["page"] == 3 for d in docs)

    def test_position_propagated_for_non_paginated_formats(self):
        """DOCX-style provenance (no real page): position + heading_path must
        still land on every chunk's metadata."""
        doc = ExtractedDocument(
            type="Word Document",
            blocks=[Block(content="Paragrafo", heading_path="Sezione 2", position=7)],
        )
        docs = _build_documents(doc, _req())
        assert docs[0].metadata["position"] == 7
        assert docs[0].metadata["heading_path"] == "Sezione 2"
        assert "page" not in docs[0].metadata

    def test_table_block_routed_through_table_chunker(self):
        table_md = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        doc = ExtractedDocument(
            type="document",
            blocks=[Block(content=table_md, block_type="table", heading_path="Sheet1")],
        )
        docs = _build_documents(doc, _req())
        assert all(d.metadata.get("element_type") == "table" for d in docs)


class TestIngestDocumentOrchestration:
    @pytest.mark.asyncio
    async def test_dense_only_when_hybrid_false(self):
        repo = AsyncMock()
        repo.aadd_documents = AsyncMock(return_value=["id1"])
        with patch(
            "tilellm.modules.ingestion.ingest.service._load_document",
            new=AsyncMock(return_value=ExtractedDocument(type="document", blocks=[Block(content="x")])),
        ):
            result = await ingest_document(_req(hybrid=False), repo=repo, llm_embeddings="EMB")

        _, kwargs = repo.aadd_documents.call_args
        assert kwargs["sparse_encoder"] is None
        assert result.chunks_indexed == 1
        assert result.chunk_ids == ["id1"]

    @pytest.mark.asyncio
    async def test_hybrid_forwards_sparse_encoder(self):
        repo = AsyncMock()
        repo.aadd_documents = AsyncMock(return_value=["id1"])
        with patch(
            "tilellm.modules.ingestion.ingest.service._load_document",
            new=AsyncMock(return_value=ExtractedDocument(type="document", blocks=[Block(content="x")])),
        ):
            await ingest_document(_req(hybrid=True, sparse_encoder="bge-m3"), repo=repo, llm_embeddings="EMB")

        _, kwargs = repo.aadd_documents.call_args
        assert kwargs["sparse_encoder"] == "bge-m3"

    @pytest.mark.asyncio
    async def test_metadata_id_is_request_id_for_dedup(self):
        repo = AsyncMock()
        repo.aadd_documents = AsyncMock(return_value=[])
        with patch(
            "tilellm.modules.ingestion.ingest.service._load_document",
            new=AsyncMock(return_value=ExtractedDocument(type="document", blocks=[])),
        ):
            await ingest_document(_req(id="stable-id"), repo=repo, llm_embeddings="EMB")

        _, kwargs = repo.aadd_documents.call_args
        assert kwargs["metadata_id"] == "stable-id"
