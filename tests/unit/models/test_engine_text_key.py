"""Engine.text_key defaults by Pinecone index type, and an explicit value is kept.

A pod→serverless migration copies metadata unchanged, so the chunk stays under
``content``. Callers must be able to pass ``text_key="content"`` on a serverless
engine; the validator used to overwrite that with ``text``.
"""
from tilellm.models import Engine


class TestEngineTextKey:
    def test_serverless_defaults_to_text(self):
        engine = Engine(name="pinecone", type="serverless", apikey="k", index_name="idx")
        assert engine.text_key == "text"

    def test_pod_defaults_to_content(self):
        engine = Engine(name="pinecone", type="pod", apikey="k", index_name="idx")
        assert engine.text_key == "content"

    def test_explicit_text_key_survives_serverless_default(self):
        engine = Engine(
            name="pinecone",
            type="serverless",
            apikey="k",
            index_name="idx",
            text_key="content",
        )
        assert engine.text_key == "content"

    def test_explicit_text_key_survives_pod_default(self):
        engine = Engine(
            name="pinecone",
            type="pod",
            apikey="k",
            index_name="idx",
            text_key="text",
        )
        assert engine.text_key == "text"

    def test_blank_text_key_falls_back_to_type_default(self):
        engine = Engine(
            name="pinecone",
            type="serverless",
            apikey="k",
            index_name="idx",
            text_key="",
        )
        assert engine.text_key == "text"

    def test_qdrant_keeps_field_default(self):
        engine = Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="idx")
        assert engine.text_key == "text"
