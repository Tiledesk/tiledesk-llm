#!/usr/bin/env python3
"""
Italian codice fiscale (CF) as an entity_extractor PA regex pattern — useful
for debt_recovery: uniquely identifies debtors/guarantors/co-obligors across
documents in a fascicolo (decision: memory/project_debt_recovery_benchmark.md).
"""
from tilellm.modules.lgraph.services.entity_extractor import _extract_pa_entities


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
