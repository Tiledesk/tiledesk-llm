#!/usr/bin/env python3
"""
_sanitize_metadata is the one choke point every Pinecone Serverless vector
passes through in aadd_documents — the right place to reject `None`, not
each metadata-building call site individually. Real production bug
(2026-08-08), third occurrence of the same class after additional_metadata
and situated_context's extracted-metadata merge: temporal_digest's own
digest-indexing metadata set `"total_amount": None` when no amount was found
in the period, and Pinecone rejected the whole upsert with 400 ("Metadata
value must be a string, number, boolean or list of strings, got 'null'").
Fixing every caller one at a time doesn't scale — this is the third one.
"""
from tilellm.store.pinecone.pinecone_repository_serverless import _sanitize_metadata


class TestSanitizeMetadataDropsNone:
    def test_none_values_dropped(self):
        result = _sanitize_metadata({"total_amount": None, "chunk_count": 4})
        assert "total_amount" not in result
        assert result["chunk_count"] == 4

    def test_falsy_but_not_none_values_kept(self):
        """0, '', False are valid Pinecone scalars — only None is invalid."""
        result = _sanitize_metadata({"count": 0, "label": "", "flag": False})
        assert result == {"count": 0, "label": "", "flag": False}

    def test_file_content_still_dropped(self):
        """Pre-existing behavior (unrelated to None-filtering) must survive."""
        result = _sanitize_metadata({"file_content": "big blob", "id": "x"})
        assert "file_content" not in result
        assert result["id"] == "x"
