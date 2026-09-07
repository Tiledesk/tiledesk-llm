#!/usr/bin/env python3
"""
make_graph_name must differentiate spaCy vs GLiNER graphs (A6 fallout,
2026-08-07): with overwrite=False, running ner_backend="gliner" on a
namespace/index pair already built with spaCy landed on the *same* FalkorDB
graph — LEntity nodes are MERGEd on (name, namespace, index_name), not type,
so a GLiNER pass silently retyped spaCy's entities (SET e.entity_type
overwrites on match). ner_backend must be part of the graph identity.

Backward compatibility is the hard constraint: every existing graph in
production (debt_recovery, the pre-A5 asl-bari demo, ...) was built before
ner_backend existed, i.e. under the "spacy" name with no suffix — the default
backend must keep producing that exact same name, or every existing graph
becomes unreachable by its old name.
"""
from tilellm.modules.lgraph.services.graph_builder import make_graph_name


class TestMakeGraphNameBackend:
    def test_default_backend_name_unchanged(self):
        """No ner_backend arg at all — every pre-A5 caller — must match the
        pre-existing naming exactly (backward compatibility)."""
        assert make_graph_name("aslbari", "determine-asl-puglia") == "lgraph_aslbari_determine-asl-puglia"

    def test_explicit_spacy_backend_name_unchanged(self):
        assert make_graph_name("aslbari", "determine-asl-puglia", "spacy") == "lgraph_aslbari_determine-asl-puglia"

    def test_gliner_backend_gets_distinct_name(self):
        assert make_graph_name("aslbari", "determine-asl-puglia", "gliner") == "lgraph_aslbari_determine-asl-puglia_gliner"

    def test_spacy_and_gliner_names_differ(self):
        spacy_name = make_graph_name("aslbari", "determine-asl-puglia", "spacy")
        gliner_name = make_graph_name("aslbari", "determine-asl-puglia", "gliner")
        assert spacy_name != gliner_name
