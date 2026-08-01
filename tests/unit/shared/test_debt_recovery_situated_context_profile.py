#!/usr/bin/env python3
"""
debt_recovery.yaml situated_context profile — doc_category classification
bundled into the same LLM call as the situating sentence (zero marginal cost).
doc_category is in situated_context.py's _DIRECT_FIELDS allowlist, so it lands
directly on chunk metadata (filterable via _metadata_filter), no code change needed.
"""
from tilellm.shared.situated_context import _load_profile_data


class TestDebtRecoveryProfile:
    def test_loads_successfully(self):
        data = _load_profile_data("debt_recovery")
        assert data is not None

    def test_json_mode_enabled(self):
        data = _load_profile_data("debt_recovery")
        assert data["json_mode"] is True

    def test_prompt_has_required_placeholders(self):
        data = _load_profile_data("debt_recovery")
        prompt = data["prompt"]
        assert "{doc_context}" in prompt
        assert "{chunk_text}" in prompt

    def test_prompt_requests_doc_category(self):
        """doc_category must be requested exactly (it's the field situated_context.py
        writes directly, unprefixed, to chunk metadata — any typo here silently
        produces an sc_-prefixed, non-filterable field instead)."""
        data = _load_profile_data("debt_recovery")
        assert "doc_category" in data["prompt"]

    def test_prompt_requests_valid_json_only(self):
        data = _load_profile_data("debt_recovery")
        assert "JSON" in data["prompt"]
