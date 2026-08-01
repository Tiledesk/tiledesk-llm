#!/usr/bin/env python3
"""
page_number is captured at ingest (handle_regex_custom_chunk) and survives in
stored chunk metadata, but extract_ids_sources/format_result (the /api/qa
response builders) never surfaced it. Citation.source_id is a direct index
into the same `docs` list used to number sources for the LLM (format_docs_with_id:
"Source ID: {i}" over enumerate(docs)) — so page can be resolved precisely per
citation, not just aggregated per source URL.

Also: content_chunks (the debug=True audit trail) is plain text with no metadata
alongside — content_chunks_metadata pairs each chunk with its page/file_name/source
for a real audit trail (which document, which page, was this chunk retrieved from).
"""
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from tilellm.controller.controller_utils import extract_ids_sources, format_result
from tilellm.models.schemas import Citation, RetrievalResult


def _doc(source, page, file_name=None, content="text", doc_id="d1"):
    meta = {"id": doc_id, "source": source, "page_number": page}
    if file_name:
        meta["file_name"] = file_name
    return Document(page_content=content, metadata=meta)


class TestCitationPageField:
    def test_page_defaults_none(self):
        assert Citation(source_id=0, source_name="x").page is None

    def test_page_explicit(self):
        assert Citation(source_id=0, source_name="x", page=3).page == 3


class TestFormatResultPageEnrichment:
    def test_page_resolved_from_source_id_index(self):
        docs = [
            _doc("http://x/a.txt", page=1, file_name="a.txt"),
            _doc("http://x/a.txt", page=7, file_name="a.txt"),
        ]
        citations = [Citation(source_id=1, source_name="http://x/a.txt")]
        result = {"context": docs, "input": "q", "chat_history": [], "answer": "ans"}
        qa = Mock(namespace="ns", debug=False, citations=True)

        out = format_result(result=result, citations=citations, question_answer=qa,
                             callback_handler=None, question_answer_list=[], success=True)

        assert out.citations[0].page == 7

    def test_page_none_when_source_id_out_of_range(self):
        """The LLM produces source_id — bounds must be defensive against hallucination."""
        docs = [_doc("http://x/a.txt", page=1)]
        citations = [Citation(source_id=99, source_name="http://x/a.txt")]
        result = {"context": docs, "input": "q", "chat_history": [], "answer": "ans"}
        qa = Mock(namespace="ns", debug=False, citations=True)

        out = format_result(result=result, citations=citations, question_answer=qa,
                             callback_handler=None, question_answer_list=[], success=True)

        assert out.citations[0].page is None

    def test_page_none_when_no_citations(self):
        docs = [_doc("http://x/a.txt", page=1)]
        result = {"context": docs, "input": "q", "chat_history": [], "answer": "ans"}
        qa = Mock(namespace="ns", debug=False, citations=False)

        out = format_result(result=result, citations=None, question_answer=qa,
                             callback_handler=None, question_answer_list=[], success=True)

        assert out.citations is None

    def test_content_chunks_metadata_reaches_retrieval_result(self):
        docs = [_doc("http://x/a.txt", page=4, file_name="a.txt", content="chunk A")]
        result = {"context": docs, "input": "q", "chat_history": [], "answer": "ans"}
        qa = Mock(namespace="ns", debug=True, citations=False)

        out = format_result(result=result, citations=None, question_answer=qa,
                             callback_handler=None, question_answer_list=[], success=True)

        assert isinstance(out, RetrievalResult)
        assert out.content_chunks_metadata == [
            {"source": "http://x/a.txt", "file_name": "a.txt", "page_number": 4},
        ]


class TestExtractIdsSourcesAudit:
    def test_content_chunks_unchanged(self):
        docs = [
            _doc("http://x/a.txt", page=2, file_name="a.txt", content="chunk A"),
            _doc("http://x/b.txt", page=5, file_name="b.txt", content="chunk B"),
        ]
        _, _, content_chunks, _, _ = extract_ids_sources(docs, debug=True)
        assert content_chunks == ["chunk A", "chunk B"]

    def test_content_chunks_metadata_parallels_content_chunks(self):
        """New audit field: per-chunk provenance (source/file_name/page), same
        order and length as content_chunks, so callers can pair chunk text with
        'which document, which page' — the actual audit trail requested."""
        docs = [
            _doc("http://x/a.txt", page=2, file_name="a.txt", content="chunk A"),
            _doc("http://x/b.txt", page=5, file_name="b.txt", content="chunk B"),
        ]
        _, _, content_chunks, _, content_chunks_metadata = extract_ids_sources(docs, debug=True)
        assert len(content_chunks_metadata) == len(content_chunks) == 2
        assert content_chunks_metadata[0] == {
            "source": "http://x/a.txt", "file_name": "a.txt", "page_number": 2,
        }
        assert content_chunks_metadata[1] == {
            "source": "http://x/b.txt", "file_name": "b.txt", "page_number": 5,
        }

    def test_content_chunks_metadata_absent_without_debug(self):
        docs = [_doc("http://x/a.txt", page=2, file_name="a.txt")]
        _, _, content_chunks, _, content_chunks_metadata = extract_ids_sources(docs, debug=False)
        assert content_chunks is None
        assert content_chunks_metadata is None
