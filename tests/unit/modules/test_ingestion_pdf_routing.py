#!/usr/bin/env python3
"""
Instradamento PDF dell'endpoint unificato /api/ingestion.

Difetto trovato importando una gara reale: 112 PDF inviati con `use_ocr=True`,
`use_docling=True`, `extract_md_simple=True` sono finiti nella **pipeline legacy**,
cioe' nella lista Redis `tiledesk_ocr_queue` — per la quale in tutto il codice
esiste solo `lpush` e **nessun consumatore**. I documenti restano li' per sempre.

Catena: `_build_pdf_request` costruisce il `PDFScrapingRequest` da
`item.model_dump()`, ma `ItemSingle` non ha i campi `use_docling` / `strategy` /
`extract_md_simple` / `export_md` / `converter` / `skip_ocr`. Pydantic li scarta in
silenzio, `PDFScrapingRequest.use_docling` resta al suo default `False`, e
`use_new_pipeline = use_docling or strategy in ("auto","fast")` diventa False.

Quindi la riga della docstring di `/api/ingestion`

    type = pdf + use_ocr = True  -> Advanced OCR pipeline (Docling)

era **falsa per ogni richiesta**: nessun payload poteva raggiungere quella pipeline.

Due correzioni verificate qui:
  1. l'endpoint chiede esplicitamente la pipeline nuova (`strategy="auto"`);
  2. `pdf_options` fa passare le opzioni PDF, con **chiavi validate** — una chiave
     sconosciuta deve dare errore, non essere ignorata (e' proprio lo scarto
     silenzioso ad aver causato il guasto).
"""
import pytest

from tilellm.models import Engine
from tilellm.models.document_type import DocumentType
from tilellm.models.llm import ItemSingle
from tilellm.modules.ingestion.controllers import _build_pdf_request


def _item(**over):
    kw = dict(
        id="doc-1",
        source="http://host/files/scheda.pdf",
        type=DocumentType.PDF,
        namespace="OP1",
        use_ocr=True,
        engine=Engine(name="qdrant", type="serverless", apikey="", vector_size=1024,
                      index_name="c", host="localhost", port=6333),
    )
    kw.update(over)
    return ItemSingle(**kw)


def _uses_new_pipeline(req) -> bool:
    """Stessa condizione di pdf_ocr.controllers.scrape_pdf."""
    return bool(req.use_docling or req.strategy in ("auto", "fast"))


class TestPipelineRouting:
    def test_default_goes_to_the_docling_pipeline(self):
        """Senza questo, ogni PDF finisce nella coda legacy senza consumatore."""
        req = _build_pdf_request(_item())
        assert _uses_new_pipeline(req)

    def test_explicit_strategy_is_honoured(self):
        req = _build_pdf_request(_item(pdf_options={"strategy": "fast"}))
        assert req.strategy == "fast"
        assert _uses_new_pipeline(req)

    def test_legacy_remains_reachable_on_purpose(self):
        """La pipeline legacy resta raggiungibile, ma solo se richiesta esplicitamente."""
        req = _build_pdf_request(_item(pdf_options={"strategy": "quality"}))
        assert req.strategy == "quality"
        assert not _uses_new_pipeline(req)


class TestPdfOptionsPassthrough:
    def test_md_simple_options_arrive(self):
        req = _build_pdf_request(_item(pdf_options={
            "extract_md_simple": True, "export_md": True,
            "converter": "docling", "skip_ocr": False,
        }))
        assert req.extract_md_simple is True
        assert req.export_md is True
        assert req.converter == "docling"
        assert req.skip_ocr is False

    def test_defaults_when_no_options(self):
        req = _build_pdf_request(_item())
        assert req.extract_md_simple is False

    def test_unknown_key_is_rejected(self):
        """Lo scarto silenzioso e' la causa del guasto: qui deve fallire."""
        with pytest.raises(ValueError) as e:
            _build_pdf_request(_item(pdf_options={"extract_md_simpl": True}))
        assert "extract_md_simpl" in str(e.value)

    def test_error_lists_the_valid_keys(self):
        with pytest.raises(ValueError) as e:
            _build_pdf_request(_item(pdf_options={"nome_inventato": 1}))
        assert "extract_md_simple" in str(e.value)


class TestSourceDerivation:
    def test_file_name_and_content_from_source(self):
        req = _build_pdf_request(_item())
        assert req.file_name == "scheda.pdf"
        assert req.file_content == "http://host/files/scheda.pdf"

    def test_explicit_file_name_wins(self):
        req = _build_pdf_request(_item(pdf_options={"file_name": "vero.pdf"}))
        assert req.file_name == "vero.pdf"

    def test_shared_fields_are_preserved(self):
        req = _build_pdf_request(_item(chunk_size=1234, namespace="OP9"))
        assert req.chunk_size == 1234
        assert req.namespace == "OP9"


class TestItemSingleField:
    def test_pdf_options_defaults_to_none(self):
        assert _item().pdf_options is None

    def test_pdf_options_accepts_a_mapping(self):
        assert _item(pdf_options={"extract_md_simple": True}).pdf_options == {
            "extract_md_simple": True
        }
