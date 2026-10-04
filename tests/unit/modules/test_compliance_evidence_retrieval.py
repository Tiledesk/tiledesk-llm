"""
Evidence retrieval shared by every compliance judge (v1 Conformità, v2 Discrezionale,
agentic compliance_retrieve_evidence): retrieve -> rerank -> re-attach neighbours.

Neighbour expansion exists because document converters can split one table row into
consecutive chunks — on a real tender a checklist row "requirement | justification |
applicable" became three chunks, and the judge saw the requirement without the
manufacturer's answer. The retrieved chunk's neighbours (chunk_index +/- 1, same
document) are merged back into it, in document order.
"""
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from tilellm.models import Engine
from tilellm.modules.compliance_checker.logic import _expand_with_neighbors

ENGINE = Engine(name="qdrant", deployment="local", host="localhost", port=6333, index_name="c", apikey=None)


def _meta(doc, idx):
    return {"doc_id": doc, "chunk_index": idx, "file_name": f"{doc}.pdf", "page": 1}


class FakeRepo:
    """Serves chunks by (doc_id, chunk_index); records every lookup."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.calls = []

    async def get_chunks_by_index(self, engine, namespace, doc_id, chunk_indexes):
        self.calls.append((doc_id, sorted(chunk_indexes)))
        return [Document(page_content=self.corpus[(doc_id, i)], metadata=_meta(doc_id, i))
                for i in chunk_indexes if (doc_id, i) in self.corpus]


async def _expand(repo, chunks, metadata, top_n=5):
    return await _expand_with_neighbors(repo, ENGINE, "ns", chunks, metadata, top_n=top_n)


@pytest.mark.asyncio
async def test_neighbours_are_merged_in_document_order():
    repo = FakeRepo({("d", 77): "RIGA PRIMA", ("d", 79): "RIGA DOPO"})

    chunks, metadata = await _expand(repo, ["RIGA CENTRALE"], [_meta("d", 78)])

    assert chunks == ["RIGA PRIMA\n\nRIGA CENTRALE\n\nRIGA DOPO"]
    assert metadata == [_meta("d", 78)]  # the citation stays on the retrieved chunk


@pytest.mark.asyncio
async def test_only_the_first_top_n_items_are_expanded():
    repo = FakeRepo({("d", i): f"vicino {i}" for i in range(0, 40)})
    metadata = [_meta("d", i) for i in (10, 20, 30)]

    chunks, _ = await _expand(repo, ["a", "b", "c"], metadata, top_n=2)

    assert chunks[0] == "vicino 9\n\na\n\nvicino 11"
    assert chunks[1] == "vicino 19\n\nb\n\nvicino 21"
    assert chunks[2] == "c"


@pytest.mark.asyncio
async def test_a_neighbour_already_selected_is_not_duplicated():
    repo = FakeRepo({("d", 4): "quattro", ("d", 5): "cinque", ("d", 6): "sei"})
    metadata = [_meta("d", 5), _meta("d", 6)]  # 6 is both 5's neighbour and item [2]

    chunks, _ = await _expand(repo, ["cinque", "sei"], metadata)

    assert chunks[0] == "quattro\n\ncinque"
    assert "sei" not in chunks[0]


@pytest.mark.asyncio
async def test_a_neighbour_shared_by_two_items_goes_to_the_first():
    repo = FakeRepo({("d", 11): "undici"})
    metadata = [_meta("d", 10), _meta("d", 12)]  # both claim 11

    chunks, _ = await _expand(repo, ["dieci", "dodici"], metadata)

    assert chunks == ["dieci\n\nundici", "dodici"]


@pytest.mark.asyncio
async def test_one_lookup_per_document():
    repo = FakeRepo({})
    metadata = [_meta("a", 1), _meta("b", 7), _meta("a", 9)]

    await _expand(repo, ["x", "y", "z"], metadata)

    assert sorted(repo.calls) == [("a", [0, 2, 8, 10]), ("b", [6, 8])]


@pytest.mark.asyncio
async def test_chunks_without_position_are_left_alone():
    repo = FakeRepo({})
    metadata = [{"file_name": "legacy.pdf"}, {"doc_id": "d"}, {"chunk_index": 3}]

    chunks, _ = await _expand(repo, ["x", "y", "z"], metadata)

    assert chunks == ["x", "y", "z"]
    assert repo.calls == []


@pytest.mark.asyncio
async def test_first_chunk_of_a_document_has_no_negative_neighbour():
    repo = FakeRepo({("d", 1): "uno"})

    await _expand(repo, ["zero"], [_meta("d", 0)])

    assert repo.calls == [("d", [1])]


@pytest.mark.asyncio
async def test_repository_failure_leaves_evidence_unchanged():
    repo = AsyncMock()
    repo.get_chunks_by_index = AsyncMock(side_effect=RuntimeError("store down"))

    chunks, metadata = await _expand(repo, ["x"], [_meta("d", 5)])

    assert chunks == ["x"]
    assert metadata == [_meta("d", 5)]


@pytest.mark.asyncio
async def test_doc_id_falls_back_to_the_id_field():
    """Older ingestion paths store the document id only as metadata.id."""
    repo = FakeRepo({("d", 6): "sei"})

    chunks, _ = await _expand(repo, ["cinque"], [{"id": "d", "chunk_index": 5}])

    assert chunks == ["cinque\n\nsei"]


@pytest.mark.asyncio
async def test_float_chunk_index_is_accepted():
    """Pinecone stores numeric metadata as floats (78.0): expansion must still work."""
    repo = FakeRepo({("d", 79): "dopo"})

    chunks, _ = await _expand(repo, ["centro"], [{"doc_id": "d", "chunk_index": 78.0}])

    assert chunks == ["centro\n\ndopo"]


# ---------------------------------------------------------------------------
# _retrieve_evidence — the one pipeline (retrieve -> rerank -> neighbours) shared by
# v1 check_compliance, v2 _evaluate_criterion_once and agentic retrieve_evidence_core.
# ---------------------------------------------------------------------------

import json  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from pydantic import SecretStr  # noqa: E402

from tilellm.models import QuestionAnswer  # noqa: E402
from tilellm.models.schemas.retrieval_schemas import RetrievalChunksResult  # noqa: E402
from tilellm.modules.compliance_checker.logic import _retrieve_evidence  # noqa: E402

_LOGIC = "tilellm.modules.compliance_checker.logic"


def _qa(top_k=45):
    return QuestionAnswer(question="q", namespace="ns", engine=ENGINE, top_k=top_k,
                          gptkey=SecretStr("k"), search_type="hybrid")


def _repo_with(chunks, metadata, corpus=None):
    repo = FakeRepo(corpus or {})
    repo.get_chunks_from_repo = AsyncMock(
        return_value=RetrievalChunksResult(namespace="ns", chunks=chunks, metadata=metadata))
    return repo


@pytest.mark.asyncio
async def test_retrieve_evidence_reranks_then_expands():
    repo = _repo_with(["b", "a"], [_meta("d", 2), _meta("d", 8)], corpus={("d", 9): "nove"})
    rerank = AsyncMock(return_value=(["a"], [_meta("d", 8)]))

    with patch(f"{_LOGIC}._rerank_chunks", rerank):
        chunks, metadata = await _retrieve_evidence(repo, _qa(), "testo del criterio", "reranker", 1, "C1")

    rerank.assert_awaited_once_with("testo del criterio", ["b", "a"], [_meta("d", 2), _meta("d", 8)], "reranker", 1)
    assert chunks == ["a\n\nnove"]
    assert metadata == [_meta("d", 8)]


@pytest.mark.asyncio
async def test_retrieve_evidence_returns_nothing_when_retrieval_fails():
    repo = FakeRepo({})
    repo.get_chunks_from_repo = AsyncMock(side_effect=ValueError("No chunks found with the current filters."))

    assert await _retrieve_evidence(repo, _qa(), "q", None, 15, "C1") == ([], [])


@pytest.mark.asyncio
async def test_retrieve_evidence_proceeds_without_reranking_when_it_fails():
    repo = _repo_with(["x"], [_meta("d", 5)], corpus={("d", 6): "sei"})

    with patch(f"{_LOGIC}._rerank_chunks", AsyncMock(side_effect=RuntimeError("reranker down"))):
        chunks, _ = await _retrieve_evidence(repo, _qa(), "q", "reranker", 15, "C1")

    assert chunks == ["x\n\nsei"]


# ---------------------------------------------------------------------------
# Every judge actually receives the neighbour — behaviour, not just wiring.
# ---------------------------------------------------------------------------

_SPLIT_ROW = {("doc-er", 79): "ISO 10993-5 Ensayos de citotoxicidad"}
_RETRIEVED = (["10.1 b) compatibilidad con los tejidos biológicos, teniendo en"], [_meta("doc-er", 78)])


def _llm_capturing(payload):
    response = MagicMock()
    response.content = json.dumps(payload)
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


def _prompt_seen_by(llm) -> str:
    return "\n".join(m.content for m in llm.ainvoke.call_args.args[0])


@pytest.mark.asyncio
async def test_v1_conformity_judge_sees_the_neighbour():
    from tilellm.modules.compliance_checker.logic import check_compliance
    from tilellm.modules.compliance_checker.models import ComplianceRequest, RequirementItem
    from tilellm.modules.compliance_checker.prompts import get_builtin_config

    request = ComplianceRequest(config=get_builtin_config("e_procurement"),
                                requirements=[RequirementItem(id="C9", text="Biocompatibile")],
                                namespace="ns", engine=ENGINE)
    llm = _llm_capturing({"judgment": "compliant", "confidence": 0.9, "source_chunk_index": 1,
                          "evidence_text": "citotoxicidad", "justification": "ok"})

    await check_compliance.__wrapped__.__wrapped__(request, repo=_repo_with(*_RETRIEVED, corpus=_SPLIT_ROW), llm=llm)

    assert "ISO 10993-5 Ensayos de citotoxicidad" in _prompt_seen_by(llm)


@pytest.mark.asyncio
async def test_v2_discretionary_judge_sees_the_neighbour():
    from tilellm.modules.compliance_checker.models_v2 import ComplianceRequestV2, DiscretionaryCriterion
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )

    request = ComplianceRequestV2(
        requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n", namespace="ns", engine=ENGINE,
    )
    llm = _llm_capturing({"coefficient": 1.0, "measured_value": None, "measured_quantity": None,
                          "motivation": "ok", "confidence": 0.9, "source_chunk_index": 1,
                          "evidence_text": "citotoxicidad", "capitolato_discrepancy": None})
    service = DiscretionaryCheckService(repo=_repo_with(*_RETRIEVED, corpus=_SPLIT_ROW), llm=llm, request=request)

    await service._evaluate_criterion_once(
        DiscretionaryCriterion(id="P9", text="Biocompatibile", mode="on_off", max_points=1))

    assert "ISO 10993-5 Ensayos de citotoxicidad" in _prompt_seen_by(llm)


@pytest.mark.asyncio
async def test_agentic_retrieve_evidence_stores_the_neighbour():
    import fakeredis.aioredis

    from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
    from tilellm.modules.agentic_compliance_checker.services.tools_core import retrieve_evidence_core
    from tilellm.modules.compliance_checker.models_v2 import (
        BulkComplianceRequestV2, DiscretionaryCriterion, OperatorRef, TenderInfo, TenderLotRequirements,
        _RequirementsBlock,
    )

    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        bulk = BulkComplianceRequestV2(
            requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n",
            operators=[OperatorRef(namespace="ns")], engine=ENGINE, gptkey=SecretStr("k"),
        )
        lot = TenderLotRequirements(
            tender=TenderInfo(title="t", lot_id="L1", lot_name="n"),
            requirements=_RequirementsBlock(discretionary=[
                DiscretionaryCriterion(id="P9", text="Biocompatibile", mode="on_off", max_points=1)]),
        )
        session_id = await SessionStore.create(bulk, lot)
        repo = _repo_with(*_RETRIEVED, corpus=_SPLIT_ROW)

        with patch("tilellm.modules.agentic_compliance_checker.services.tools_core._resolve_deps",
                   AsyncMock(return_value=(repo, AsyncMock()))):
            body = json.loads(await retrieve_evidence_core(session_id=session_id, criterion_id="P9"))

        entry = await SessionStore.get_evidence(session_id, body["evidence_ref"])
        assert "ISO 10993-5 Ensayos de citotoxicidad" in entry.chunks[0]
    finally:
        SessionStore._client = None


# ---------------------------------------------------------------------------
# Clarification documents excluded on the Conformità path too (review pattern E):
# answers to tender clarifications can only point back to offer pages, never add
# evidence. v2 applied this only to scored criteria; Conformità (v1) still used them.
# ---------------------------------------------------------------------------

from tilellm.modules.compliance_checker.logic import EXCLUDE_CHIARIMENTI_FILTER  # noqa: E402


async def _v1_qa_sent_to_repo(**request_overrides):
    from tilellm.modules.compliance_checker.logic import check_compliance
    from tilellm.modules.compliance_checker.models import ComplianceRequest, RequirementItem
    from tilellm.modules.compliance_checker.prompts import get_builtin_config

    request = ComplianceRequest(config=get_builtin_config("e_procurement"),
                                requirements=[RequirementItem(id="C1", text="Radiopaco")],
                                namespace="ns", engine=ENGINE, **request_overrides)
    repo = _repo_with(["x"], [{"file_name": "a.pdf", "page": 1}])
    llm = _llm_capturing({"judgment": "not_verifiable", "confidence": 0.0, "source_chunk_index": 0,
                          "evidence_text": "", "justification": "-"})
    await check_compliance.__wrapped__.__wrapped__(request, repo=repo, llm=llm)
    return repo.get_chunks_from_repo.call_args.args[0]


@pytest.mark.asyncio
async def test_v1_excludes_clarifications_when_asked():
    qa = await _v1_qa_sent_to_repo(exclude_chiarimenti=True)

    assert getattr(qa, "_metadata_filter", None) == EXCLUDE_CHIARIMENTI_FILTER


@pytest.mark.asyncio
async def test_v1_standalone_default_is_unchanged():
    qa = await _v1_qa_sent_to_repo()

    assert getattr(qa, "_metadata_filter", None) is None


@pytest.mark.asyncio
async def test_v2_passes_its_clarification_policy_to_the_conformity_path():
    from tilellm.modules.compliance_checker.models import ComplianceReport, ComplianceSummary
    from tilellm.modules.compliance_checker.models_v2 import (
        ComplianceRequestV2, TabularRequirementV2, TenderInfo, TenderLotRequirements, _RequirementsBlock,
    )
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )

    request = ComplianceRequestV2(requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n",
                                  namespace="ns", engine=ENGINE)  # exclude_chiarimenti defaults to True
    lot = TenderLotRequirements(tender=TenderInfo(title="t", lot_id="L1", lot_name="n"),
                                requirements=_RequirementsBlock(tabular=[TabularRequirementV2(id="C1", text="x")]))
    fake_v1 = AsyncMock(return_value=ComplianceReport(domain="e_procurement", namespace="ns",
                                                      summary=ComplianceSummary(total=0), results=[]))

    with patch("tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance", fake_v1):
        await DiscretionaryCheckService(repo=AsyncMock(), llm=AsyncMock(), request=request)._check_tabular(lot)

    assert fake_v1.call_args.args[0].exclude_chiarimenti is True


def test_compliance_requests_default_to_a_multilingual_reranker_and_a_wider_pool():
    """Measured on the real tender: the English-only ms-marco cross-encoder pushed the
    right chunk from #17 of the hybrid pool to #45 (last) — an English "GS1 data matrix"
    passage against an Italian requirement; bge-reranker-v2-m3 put it at #2. Offers are
    multilingual, so is the reranker. The pool goes from 3x to 6x top_k."""
    from tilellm.modules.compliance_checker.models import ComplianceRequest
    from tilellm.modules.compliance_checker.models_v2 import (
        BulkComplianceRequestV2,
        ComplianceRequestV2,
    )

    for model in (ComplianceRequest, ComplianceRequestV2, BulkComplianceRequestV2):
        fields = model.model_fields
        assert fields["reranker_model"].default == "BAAI/bge-reranker-v2-m3", model.__name__
        assert fields["reranking_multiplier"].default == 6, model.__name__


# ---------------------------------------------------------------------------
# search_query: a declared search wording, distinct from the official text.
# Real tender: "Latex free" (two English words) never retrieved an offer saying
# "non contengono lattice" (not in the top 90); "Latex free (privo di lattice)" put it
# at #1. The tender text can't be edited, so the wording is declared per criterion
# in the criteria table. It drives retrieval and reranking only — the judge and the
# report keep the official text.
# ---------------------------------------------------------------------------

_SEARCH = "Latex free, privo di lattice"


def _capturing_rerank():
    async def rerank(query, chunks, metadata, config, top_k):
        rerank.queries.append(query)
        return chunks, metadata
    rerank.queries = []
    return rerank


@pytest.mark.asyncio
async def test_v1_searches_with_the_declared_query_and_judges_the_official_text():
    from tilellm.modules.compliance_checker.logic import check_compliance
    from tilellm.modules.compliance_checker.models import ComplianceRequest, RequirementItem
    from tilellm.modules.compliance_checker.prompts import get_builtin_config

    request = ComplianceRequest(
        config=get_builtin_config("e_procurement"), namespace="ns", engine=ENGINE, reranking=True,
        requirements=[RequirementItem(id="C1", text="Latex free", search_query=_SEARCH)])
    repo = _repo_with(["non contengono lattice"], [{"file_name": "st.pdf", "page": 6}])
    llm = _llm_capturing({"judgment": "compliant", "confidence": 0.9, "source_chunk_index": 1,
                          "evidence_text": "non contengono lattice", "justification": "ok"})
    rerank = _capturing_rerank()

    with patch(f"{_LOGIC}._rerank_chunks", rerank):
        await check_compliance.__wrapped__.__wrapped__(request, repo=repo, llm=llm)

    assert repo.get_chunks_from_repo.call_args.args[0].question == _SEARCH
    assert rerank.queries == [_SEARCH]
    assert "Testo: Latex free\n" in _prompt_seen_by(llm)
    assert _SEARCH not in _prompt_seen_by(llm)


@pytest.mark.asyncio
async def test_v1_without_a_search_query_searches_with_the_text():
    qa = await _v1_qa_sent_to_repo()

    assert qa.question == "Radiopaco"


@pytest.mark.asyncio
async def test_v2_conformity_path_forwards_the_search_query():
    from tilellm.modules.compliance_checker.models import ComplianceReport, ComplianceSummary
    from tilellm.modules.compliance_checker.models_v2 import (
        ComplianceRequestV2, TabularRequirementV2, TenderInfo, TenderLotRequirements, _RequirementsBlock,
    )
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )

    request = ComplianceRequestV2(requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n",
                                  namespace="ns", engine=ENGINE)
    lot = TenderLotRequirements(
        tender=TenderInfo(title="t", lot_id="L1", lot_name="n"),
        requirements=_RequirementsBlock(tabular=[
            TabularRequirementV2(id="C1", text="Latex free", search_query=_SEARCH)]))
    fake_v1 = AsyncMock(return_value=ComplianceReport(domain="e_procurement", namespace="ns",
                                                      summary=ComplianceSummary(total=0), results=[]))

    with patch("tilellm.modules.compliance_checker.services.discretionary_check_service.check_compliance", fake_v1):
        await DiscretionaryCheckService(repo=AsyncMock(), llm=AsyncMock(), request=request)._check_tabular(lot)

    assert fake_v1.call_args.args[0].requirements[0].search_query == _SEARCH


@pytest.mark.asyncio
async def test_v2_discretionary_searches_with_the_declared_query_and_judges_the_official_text():
    from tilellm.modules.compliance_checker.models_v2 import ComplianceRequestV2, DiscretionaryCriterion
    from tilellm.modules.compliance_checker.services.discretionary_check_service import (
        DiscretionaryCheckService,
    )

    request = ComplianceRequestV2(requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n",
                                  namespace="ns", engine=ENGINE, reranking=True)
    repo = _repo_with(["non contengono lattice"], [{"file_name": "st.pdf", "page": 6}])
    llm = _llm_capturing({"coefficient": 1.0, "measured_value": None, "measured_quantity": None,
                          "motivation": "ok", "confidence": 0.9, "source_chunk_index": 1,
                          "evidence_text": "non contengono lattice", "capitolato_discrepancy": None})
    rerank = _capturing_rerank()

    with patch(f"{_LOGIC}._rerank_chunks", rerank):
        await DiscretionaryCheckService(repo=repo, llm=llm, request=request)._evaluate_criterion_once(
            DiscretionaryCriterion(id="P9", text="Latex free", mode="on_off", max_points=1,
                                   search_query=_SEARCH))

    assert repo.get_chunks_from_repo.call_args.args[0].question == _SEARCH
    assert rerank.queries == [_SEARCH]
    assert "Testo: Latex free\n" in _prompt_seen_by(llm)
    assert _SEARCH not in _prompt_seen_by(llm)


@pytest.mark.asyncio
async def test_agentic_retrieve_evidence_by_criterion_uses_the_search_query():
    import fakeredis.aioredis

    from tilellm.modules.agentic_compliance_checker.services.session_store import SessionStore
    from tilellm.modules.agentic_compliance_checker.services.tools_core import retrieve_evidence_core
    from tilellm.modules.compliance_checker.models_v2 import (
        BulkComplianceRequestV2, DiscretionaryCriterion, OperatorRef, TenderInfo, TenderLotRequirements,
        _RequirementsBlock,
    )

    SessionStore._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        bulk = BulkComplianceRequestV2(
            requirements_yaml="tender:\n  title: t\n  lot_id: L1\n  lot_name: n\n",
            operators=[OperatorRef(namespace="ns")], engine=ENGINE, gptkey=SecretStr("k"),
        )
        lot = TenderLotRequirements(
            tender=TenderInfo(title="t", lot_id="L1", lot_name="n"),
            requirements=_RequirementsBlock(discretionary=[DiscretionaryCriterion(
                id="P9", text="Latex free", mode="on_off", max_points=1, search_query=_SEARCH)]),
        )
        session_id = await SessionStore.create(bulk, lot)
        repo = _repo_with(["non contengono lattice"], [{"file_name": "st.pdf", "page": 6}])

        with patch("tilellm.modules.agentic_compliance_checker.services.tools_core._resolve_deps",
                   AsyncMock(return_value=(repo, AsyncMock()))):
            await retrieve_evidence_core(session_id=session_id, criterion_id="P9")

        assert repo.get_chunks_from_repo.call_args.args[0].question == _SEARCH
    finally:
        SessionStore._client = None


def test_criteria_workbook_round_trips_the_search_query():
    from tilellm.modules.compliance_checker.models_v2 import (
        DiscretionaryCriterion, TabularRequirementV2, TenderInfo, TenderLotRequirements, _RequirementsBlock,
    )
    from tilellm.modules.compliance_checker.services.requirements_xlsx_service import RequirementsXlsxService

    lot = TenderLotRequirements(
        tender=TenderInfo(title="t", lot_id="1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            tabular=[TabularRequirementV2(id="C1", text="Latex free", search_query=_SEARCH),
                     TabularRequirementV2(id="C2", text="Radiopaco")],
            discretionary=[DiscretionaryCriterion(id="P1", text="Gamma", mode="proporzionale",
                                                  max_points=5, search_query="numero di referenze")]),
    )
    svc = RequirementsXlsxService()

    parsed = svc.parse_workbook(svc.build_workbook([lot]))[0].requirements

    assert [r.search_query for r in parsed.tabular] == [_SEARCH, None]
    assert parsed.discretionary[0].search_query == "numero di referenze"
