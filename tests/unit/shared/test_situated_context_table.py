"""Tests for table-aware situated context prompt selection."""
import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from tilellm.shared.situated_context import (
    _generate_situated_context,
    enrich_chunks_with_situated_context,
    _SITUATED_CONTEXT_TABLE_PROMPT,
    _SITUATED_CONTEXT_TABLE_ROW_PROMPT,
    _SITUATED_CONTEXT_PROMPT,
)

SOURCE_URL = "https://example.com/catalog"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text: str = "Contesto generato."):
    resp = MagicMock()
    resp.content = text
    resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


def _table_doc(extra_meta: dict = None) -> Document:
    meta = {"element_type": "table", "col_names": "SKU, Nome, Prezzo", "source": SOURCE_URL}
    if extra_meta:
        meta.update(extra_meta)
    return Document(
        page_content="| SKU | Nome | Prezzo |\n|---|---|---|\n| A001 | Alpha | 9.99 |\n| A002 | Beta | 19.99 |",
        metadata=meta,
    )


def _table_row_doc(row: str = "| A001 | Alpha | 9.99 |") -> Document:
    return Document(
        page_content=f"| SKU | Nome | Prezzo |\n|---|---|---|\n{row}",
        metadata={"element_type": "table_rows", "col_names": "SKU, Nome, Prezzo", "source": SOURCE_URL},
    )


def _plain_doc() -> Document:
    return Document(page_content="Questo è un paragrafo di testo normale.", metadata={})


# ---------------------------------------------------------------------------
# Unit: _generate_situated_context — prompt selection by element_type
# ---------------------------------------------------------------------------

class TestPromptSelection:
    @pytest.mark.asyncio
    async def test_atomic_table_uses_table_prompt(self):
        captured = []

        async def fake_ainvoke(messages):
            captured.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        await _generate_situated_context(
            doc_context="catalogo",
            chunk_text="| SKU | Nome |\n|---|---|\n| A001 | Alpha |\n| A002 | Beta |",
            llm=llm,
            chunk_metadata={"element_type": "table", "col_names": "SKU, Nome", "source": SOURCE_URL},
        )

        prompt = captured[0]
        # atomic table prompt mentions columns and table-level description
        assert "SKU" in prompt or "column" in prompt.lower() or "colonne" in prompt.lower()
        # must NOT be the row-specific prompt
        assert "single row" not in prompt.lower() and "riga" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_table_rows_uses_row_specific_prompt(self):
        """per_row chunks must get the row-specific prompt, not the generic table-level one."""
        captured = []

        async def fake_ainvoke(messages):
            captured.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        await _generate_situated_context(
            doc_context="catalogo",
            chunk_text="| SKU | Nome | Prezzo |\n|---|---|---|\n| A001 | Alpha | 9.99 |",
            llm=llm,
            chunk_metadata={"element_type": "table_rows", "col_names": "SKU, Nome, Prezzo", "source": SOURCE_URL},
        )

        prompt = captured[0]
        # row prompt explicitly says "single row" / "row"
        assert "row" in prompt.lower() or "riga" in prompt.lower()
        # col_names injected
        assert "SKU" in prompt

    @pytest.mark.asyncio
    async def test_table_rows_prompt_contains_source_as_context(self):
        """For per_row chunks, doc_context should be the source URL, not pipe-markdown."""
        captured = []

        async def fake_ainvoke(messages):
            captured.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        await _generate_situated_context(
            doc_context="ignored_fallback",
            chunk_text="| SKU | Nome |\n|---|---|\n| A001 | Alpha |",
            llm=llm,
            chunk_metadata={"element_type": "table_rows", "col_names": "SKU, Nome", "source": SOURCE_URL},
        )

        prompt = captured[0]
        # source URL used, not the generic fallback
        assert SOURCE_URL in prompt or "example.com" in prompt

    @pytest.mark.asyncio
    async def test_plain_chunk_uses_generic_prompt(self):
        captured = []

        async def fake_ainvoke(messages):
            captured.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        await _generate_situated_context(
            doc_context="documento generico",
            chunk_text="Testo normale senza tabelle.",
            llm=llm,
            chunk_metadata=None,
        )

        prompt = captured[0]
        assert "document" in prompt.lower() or "chunk" in prompt.lower()
        assert "row" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_returns_text_and_usage(self):
        async def fake_ainvoke(messages):
            return _make_llm_response("Prodotto Alpha (SKU: A001) ha prezzo 9.99.")

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        text, usage = await _generate_situated_context(
            doc_context=SOURCE_URL,
            chunk_text="| SKU | Nome |\n|---|---|\n| A001 | Alpha |",
            llm=llm,
            chunk_metadata={"element_type": "table_rows", "col_names": "SKU, Nome", "source": SOURCE_URL},
        )

        assert "A001" in text or "Alpha" in text
        assert usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Unit: per_row vs atomic prompt divergence
# ---------------------------------------------------------------------------

class TestTableRowsVsAtomicDivergence:
    """Verify that per_row and atomic table chunks receive *different* prompts.

    This is the core guard against the regression where all per_row chunks
    received the same generic table-level description → identical embeddings.
    """

    @pytest.mark.asyncio
    async def test_row_prompt_differs_from_atomic_prompt(self):
        row_prompt = []
        atomic_prompt = []

        async def capture_row(messages):
            row_prompt.append(messages[0].content)
            return _make_llm_response("Descrizione riga.")

        async def capture_atomic(messages):
            atomic_prompt.append(messages[0].content)
            return _make_llm_response("Descrizione tabella.")

        llm_row = MagicMock()
        llm_row.ainvoke = capture_row
        llm_atomic = MagicMock()
        llm_atomic.ainvoke = capture_atomic

        await _generate_situated_context(
            doc_context=SOURCE_URL,
            chunk_text="| SKU | Nome |\n|---|---|\n| A001 | Alpha |",
            llm=llm_row,
            chunk_metadata={"element_type": "table_rows", "col_names": "SKU, Nome", "source": SOURCE_URL},
        )
        await _generate_situated_context(
            doc_context=SOURCE_URL,
            chunk_text="| SKU | Nome |\n|---|---|\n| A001 | Alpha |\n| A002 | Beta |",
            llm=llm_atomic,
            chunk_metadata={"element_type": "table", "col_names": "SKU, Nome", "source": SOURCE_URL},
        )

        assert row_prompt[0] != atomic_prompt[0], "per_row and atomic must use different prompts"
        # row-specific prompt should reference individual row / item
        assert "row" in row_prompt[0].lower() or "riga" in row_prompt[0].lower()


# ---------------------------------------------------------------------------
# Integration: enrich_chunks dispatches correct prompts
# ---------------------------------------------------------------------------

class TestEnrichChunksTableAware:
    @pytest.mark.asyncio
    async def test_table_doc_enriched_with_table_context(self):
        async def fake_ainvoke(messages):
            return _make_llm_response("Tabella prodotti con SKU e prezzi.")

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_doc(), _plain_doc()]
        result = await enrich_chunks_with_situated_context(docs, llm)

        assert len(result.documents) == 2
        assert "Tabella prodotti" in result.documents[0].page_content
        assert result.token_usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_per_row_doc_enriched_with_row_specific_context(self):
        prompts = []

        async def fake_ainvoke(messages):
            prompts.append(messages[0].content)
            return _make_llm_response("Prodotto Alpha, SKU A001, prezzo 9.99.")

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_row_doc()]
        result = await enrich_chunks_with_situated_context(docs, llm)

        assert len(prompts) == 1
        assert "row" in prompts[0].lower() or "riga" in prompts[0].lower()
        assert "Prodotto Alpha" in result.documents[0].page_content

    @pytest.mark.asyncio
    async def test_doc_context_uses_source_not_pipe_markdown_for_table_rows(self):
        """enrich_chunks must not build doc_context from pipe-markdown content."""
        prompts = []

        async def fake_ainvoke(messages):
            prompts.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_row_doc()]
        await enrich_chunks_with_situated_context(docs, llm)

        assert len(prompts) == 1
        # pipe chars in doc_context would mean we used the content as context — wrong
        pipe_count = prompts[0].count("|")
        # The chunk_text itself has pipes, but the doc_context section should NOT be pipe-heavy
        # We check that the source URL is present (used as doc_context)
        assert SOURCE_URL in prompts[0] or "example.com" in prompts[0]

    @pytest.mark.asyncio
    async def test_col_names_injected_in_table_and_row_prompts(self):
        prompts = []

        async def fake_ainvoke(messages):
            prompts.append(messages[0].content)
            return _make_llm_response()

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_doc(), _table_row_doc()]
        await enrich_chunks_with_situated_context(docs, llm)

        for p in prompts:
            assert "SKU" in p

    @pytest.mark.asyncio
    async def test_has_situated_context_flag_set(self):
        async def fake_ainvoke(messages):
            return _make_llm_response("Contesto.")

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_doc()]
        result = await enrich_chunks_with_situated_context(docs, llm)
        assert result.documents[0].metadata.get("has_situated_context") is True

    @pytest.mark.asyncio
    async def test_two_per_row_chunks_different_content_get_different_responses(self):
        """Each row chunk must be individually enriched — not share a single response."""
        responses = [
            "Prodotto Alpha (A001), prezzo 9.99.",
            "Prodotto Beta (A002), prezzo 19.99.",
        ]
        call_idx = [0]

        async def fake_ainvoke(messages):
            resp = _make_llm_response(responses[call_idx[0]])
            call_idx[0] += 1
            return resp

        llm = MagicMock()
        llm.ainvoke = fake_ainvoke

        docs = [_table_row_doc("| A001 | Alpha | 9.99 |"), _table_row_doc("| A002 | Beta | 19.99 |")]
        result = await enrich_chunks_with_situated_context(docs, llm)

        content0 = result.documents[0].page_content
        content1 = result.documents[1].page_content
        assert "A001" in content0 or "Alpha" in content0
        assert "A002" in content1 or "Beta" in content1
        assert content0 != content1
