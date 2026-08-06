"""
docs/GRAPHRAG_COST_QUALITY_PLAN.md A0: situated_context's dual-JSON-output mode
({"context": ..., "metadata": {...}}) always merged BOTH into the chunk — the
prepended "context" sentence was polluting the lgraph entity graph on the ASL
namespace (34.3% of extracted entities turned out to be English boilerplate
from the situating sentence, see docs/GRAPHRAG_COST_QUALITY_PLAN.md §6b).

`metadata_only=True` keeps the metadata merge (act_type/topics/amount/...,
still worth the same LLM call, see A0bis — cost is unchanged either way) but
skips prepending `context` to page_content.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tilellm.shared.situated_context import enrich_chunks_with_situated_context

_JSON_RESPONSE = (
    '{"context": "Questa determina riguarda un acquisto di dispositivi medici.", '
    '"metadata": {"act_type": "acquisto_dispositivi_medici", "amount": 1200.0}}'
)

_JSON_RESPONSE_NULL_AMOUNT = (
    '{"context": "Determina di variazione di personale.", '
    '"metadata": {"act_type": "variazione_personale", "amount": null}}'
)


def _json_llm():
    async def fake_ainvoke(messages):
        resp = MagicMock()
        resp.content = _JSON_RESPONSE
        resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return resp

    llm = MagicMock()
    llm.ainvoke = fake_ainvoke
    return llm


def _json_llm_null_amount():
    async def fake_ainvoke(messages):
        resp = MagicMock()
        resp.content = _JSON_RESPONSE_NULL_AMOUNT
        resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return resp

    llm = MagicMock()
    llm.ainvoke = fake_ainvoke
    return llm


def _doc() -> Document:
    return Document(page_content="Testo originale del chunk.", metadata={})


@pytest.mark.asyncio
async def test_metadata_only_does_not_prepend_context():
    result = await enrich_chunks_with_situated_context(
        [_doc()], _json_llm(),
        metadata_extraction_prompt="estrai {chunk_text}",
        metadata_only=True,
    )
    assert result.documents[0].page_content == "Testo originale del chunk."


@pytest.mark.asyncio
async def test_metadata_only_still_merges_metadata():
    result = await enrich_chunks_with_situated_context(
        [_doc()], _json_llm(),
        metadata_extraction_prompt="estrai {chunk_text}",
        metadata_only=True,
    )
    assert result.documents[0].metadata["act_type"] == "acquisto_dispositivi_medici"
    assert result.documents[0].metadata["amount"] == 1200.0


@pytest.mark.asyncio
async def test_default_behavior_unchanged_prepends_context():
    """metadata_only defaults to False — every existing caller keeps prepending."""
    result = await enrich_chunks_with_situated_context(
        [_doc()], _json_llm(),
        metadata_extraction_prompt="estrai {chunk_text}",
    )
    assert result.documents[0].page_content.startswith(
        "Questa determina riguarda un acquisto di dispositivi medici."
    )
    assert result.documents[0].metadata["act_type"] == "acquisto_dispositivi_medici"


@pytest.mark.asyncio
async def test_extracted_metadata_null_field_dropped_not_stored():
    """Real production bug (2026-08-06): the LLM returns "amount": null for a
    chunk with no monetary figure (e.g. a personnel-variation determina), and
    that null was merged into doc.metadata verbatim -> Pinecone rejects the
    whole upsert with 400 ("Metadata value must be a string, number, boolean
    or list of strings, got 'null' for field 'amount'"). null means "not
    found in this chunk" — drop the key, don't store null."""
    result = await enrich_chunks_with_situated_context(
        [_doc()], _json_llm_null_amount(),
        metadata_extraction_prompt="estrai {chunk_text}",
        metadata_only=True,
    )
    assert "amount" not in result.documents[0].metadata
    assert result.documents[0].metadata["act_type"] == "variazione_personale"
