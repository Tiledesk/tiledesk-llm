#!/usr/bin/env python3
"""
gliner_extractor — GLiNER NER backend (A6, docs/GRAPHRAG_COST_QUALITY_PLAN.md
§4a/§8). No real model is loaded here: predict_entities is mocked, mirroring
how entity_extractor's own tests never require a real spaCy model download
for logic that doesn't need one.
"""
import pytest

from tilellm.modules.lgraph.services import gliner_extractor as ge


class TestExtractEntitiesGliner:
    def test_empty_text_returns_empty(self):
        assert ge.extract_entities_gliner("", ["PER"]) == []

    def test_maps_gliner_label_back_to_short_type(self, mocker):
        fake_model = mocker.MagicMock()
        fake_model.predict_entities.return_value = [
            {"text": "Mario Rossi", "label": "persona", "score": 0.9},
            {"text": "Comune di Bari", "label": "organizzazione o azienda", "score": 0.8},
        ]
        mocker.patch.object(ge, "_get_model", return_value=fake_model)

        result = ge.extract_entities_gliner("Mario Rossi, Comune di Bari", ["PER", "ORG"])

        assert ("mario rossi", "PER") in result
        assert ("comune di bari", "ORG") in result

    def test_only_requests_labels_within_include_types(self, mocker):
        fake_model = mocker.MagicMock()
        fake_model.predict_entities.return_value = []
        mocker.patch.object(ge, "_get_model", return_value=fake_model)

        ge.extract_entities_gliner("testo", ["PER"])

        requested_labels = fake_model.predict_entities.call_args[0][1]
        assert requested_labels == ["persona"]

    def test_empty_include_types_requests_all_model_labels(self, mocker):
        fake_model = mocker.MagicMock()
        fake_model.predict_entities.return_value = []
        mocker.patch.object(ge, "_get_model", return_value=fake_model)

        ge.extract_entities_gliner("testo", [])

        requested_labels = fake_model.predict_entities.call_args[0][1]
        assert set(requested_labels) == set(ge._TYPE_TO_LABEL.values())

    def test_deduplicates_normalized_names(self, mocker):
        fake_model = mocker.MagicMock()
        fake_model.predict_entities.return_value = [
            {"text": "ASL Bari", "label": "organizzazione o azienda", "score": 0.9},
            {"text": "asl bari", "label": "organizzazione o azienda", "score": 0.7},
        ]
        mocker.patch.object(ge, "_get_model", return_value=fake_model)

        result = ge.extract_entities_gliner("ASL Bari, asl bari", ["ORG"])

        assert result.count(("asl bari", "ORG")) == 1


class TestGetModel:
    def test_raises_actionable_error_when_gliner_not_installed(self, mocker):
        mocker.patch.object(ge, "GLINER_AVAILABLE", False)
        ge._model_cache.clear()
        try:
            with pytest.raises(ImportError, match="pip install gliner"):
                ge._get_model("some-model")
        finally:
            ge._model_cache.clear()
