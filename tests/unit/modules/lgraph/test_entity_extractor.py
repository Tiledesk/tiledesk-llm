#!/usr/bin/env python3
"""
Italian codice fiscale (CF) as an entity_extractor PA regex pattern — useful
for debt_recovery: uniquely identifies debtors/guarantors/co-obligors across
documents in a fascicolo (decision: memory/project_debt_recovery_benchmark.md).
"""
import pytest

from tilellm.modules.lgraph.services.entity_extractor import (
    _extract_pa_entities,
    build_chunk_entity_matrix,
    extract_entities,
)


class TestCodiceFiscaleExtraction:
    def test_extracts_valid_codice_fiscale(self):
        text = "Il debitore, C.F. RSSMRA80A01H501U, ha sottoscritto il contratto."
        results = _extract_pa_entities(text)
        assert ("rssmra80a01h501u", "CF") in results

    def test_extracts_codice_fiscale_without_prefix(self):
        text = "Cointestatario: RSSMRA80A01H501U garante del prestito."
        results = _extract_pa_entities(text)
        assert ("rssmra80a01h501u", "CF") in results

    def test_does_not_match_wrong_length_or_shape(self):
        text = "Il numero pratica POS3872500 non è un codice fiscale."
        results = _extract_pa_entities(text)
        assert not any(label == "CF" for _, label in results)

    def test_extracts_multiple_distinct_cf(self):
        text = "Debitore RSSMRA80A01H501U, garante BNCLGU75B41F205X."
        results = _extract_pa_entities(text)
        labels = {norm for norm, label in results if label == "CF"}
        assert labels == {"rssmra80a01h501u", "bnclgu75b41f205x"}


class TestNerBackendDispatch:
    """A5 (docs/GRAPHRAG_COST_QUALITY_PLAN.md §7/§8): ner_backend is selectable
    per-request, spaCy stays the unchanged default. GLiNER itself isn't wired in
    yet (that's A6) — an unknown/unimplemented backend must fail loud, not
    silently fall back to spaCy under a different label."""

    def test_default_backend_is_spacy_unchanged(self):
        text = "Il Comune di Bari ha pubblicato la determina n. 100."
        default = extract_entities(text, "it_core_news_lg", ["ORG", "LOC"], use_noun_chunks=False)
        explicit = extract_entities(
            text, "it_core_news_lg", ["ORG", "LOC"], use_noun_chunks=False, ner_backend="spacy",
        )
        assert default == explicit

    def test_unknown_ner_backend_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="made-up-backend"):
            extract_entities(
                "testo", "it_core_news_lg", [], use_noun_chunks=False, ner_backend="made-up-backend",
            )

    def test_build_chunk_entity_matrix_forwards_unknown_backend_error(self):
        chunks = [{"id": "c1", "text": "Comune di Bari"}]
        with pytest.raises(NotImplementedError, match="made-up-backend"):
            build_chunk_entity_matrix(
                chunks, "it_core_news_lg", [], use_noun_chunks=False, ner_backend="made-up-backend",
            )


class TestGlinerDispatch:
    """A6: ner_backend="gliner" routes to gliner_extractor.extract_entities_gliner
    for the model-based types (PER/ORG/LOC/MISC) and still adds the shared regex
    types (CIG, CUP, CF, DATE_IT, MONEY, QUANTITY) — same merge/dedup contract as
    the spaCy path, so downstream consumers see one uniform output shape."""

    def test_gliner_backend_merges_model_and_regex_entities(self, mocker):
        mocker.patch(
            "tilellm.modules.lgraph.services.gliner_extractor.extract_entities_gliner",
            return_value=[("asl bari", "ORG")],
        )
        text = "L'ASL Bari ha indetto una gara. CIG: B1EFBDA301."
        result = extract_entities(
            text, "it_core_news_lg", ["ORG", "CIG"], use_noun_chunks=False, ner_backend="gliner",
        )
        assert ("asl bari", "ORG") in result
        assert ("b1efbda301", "CIG") in result

    def test_gliner_backend_deduplicates_against_regex(self, mocker):
        """A regex hit and a GLiNER hit that normalize to the same string must
        not appear twice — same behavior as the spaCy path's `seen` set."""
        mocker.patch(
            "tilellm.modules.lgraph.services.gliner_extractor.extract_entities_gliner",
            return_value=[("b1efbda301", "MISC")],
        )
        text = "CIG: B1EFBDA301."
        result = extract_entities(
            text, "it_core_news_lg", [], use_noun_chunks=False, ner_backend="gliner",
        )
        assert result.count(("b1efbda301", "MISC")) == 1
        assert ("b1efbda301", "CIG") not in result  # first writer (GLiNER) wins
