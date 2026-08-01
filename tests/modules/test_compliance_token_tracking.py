"""
Integration tests for token tracking in the compliance v2 discretionary service.

Exercises the real DiscretionaryCheckService.evaluate_lot path with fake repo + llm
to assert per-call token usage is collected and surfaced.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tilellm.models import Engine
from tilellm.modules.compliance_checker.models_v2 import (
    ComplianceRequestV2,
    DiscretionaryCriterion,
    DiscretionaryMode,
    TenderInfo,
    TenderLotRequirements,
    _RequirementsBlock,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import (
    DiscretionaryCheckService,
)


class _FakeLLM:
    """Returns an AIMessage-like object carrying JSON content + usage_metadata."""

    def __init__(self, payload: dict, usage: dict):
        self._payload = payload
        self._usage = usage
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return SimpleNamespace(
            content=json.dumps(self._payload),
            usage_metadata=self._usage,
            response_metadata={},
        )


class _FakeRepo:
    async def get_chunks_from_repo(self, qa):
        return SimpleNamespace(
            chunks=["L'offerta include la caratteristica richiesta."],
            metadata=[{"id": "d1", "doc_id": "d1", "source": "offerta.pdf", "page": 3}],
        )


def _request(**over):
    base = dict(
        namespace="ns-op1",
        engine=Engine(name="qdrant"),
        requirements_yaml="dummy: true",  # satisfies the requirements-source validator
        model="gpt-4o",
        debug=True,
        id_project="proj-1",
    )
    base.update(over)
    return ComplianceRequestV2(**base)


def _lot_with_one_discretionary():
    return TenderLotRequirements(
        tender=TenderInfo(title="Gara X", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            tabular=[],
            discretionary=[
                DiscretionaryCriterion(
                    id="C1", text="Qualità della soluzione",
                    mode=DiscretionaryMode.VARIABILE, max_points=10,
                ),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_discretionary_service_collects_token_usage():
    llm = _FakeLLM(
        payload={
            "coefficient": 0.8, "motivation": "Fondato sull'evidenza.",
            "confidence": 0.9, "source_chunk_index": 1,
            "evidence_text": "L'offerta include la caratteristica richiesta.",
        },
        usage={"input_tokens": 700, "output_tokens": 150, "total_tokens": 850},
    )
    svc = DiscretionaryCheckService(repo=_FakeRepo(), llm=llm, request=_request())

    report = await svc.evaluate_lot(_lot_with_one_discretionary())

    assert llm.calls == 1
    # one judge call recorded
    usage = svc.tokens.to_dict()
    assert usage["total"] == {"prompt": 700, "completion": 150, "total": 850}
    assert usage["calls"][0]["op"] == "discretionary_judge"
    assert usage["calls"][0]["model"] == "gpt-4o"
    # report built and one discretionary result produced
    assert len(report.discretionary_results) == 1


@pytest.mark.asyncio
async def test_human_only_criterion_makes_no_llm_call_and_no_tokens():
    llm = _FakeLLM(payload={}, usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    lot = TenderLotRequirements(
        tender=TenderInfo(title="Gara X", lot_id="L1", lot_name="Lotto 1"),
        requirements=_RequirementsBlock(
            tabular=[],
            discretionary=[
                DiscretionaryCriterion(
                    id="C1", text="Valutazione soggettiva",
                    mode=DiscretionaryMode.VARIABILE, max_points=5, human_only=True,
                ),
            ],
        ),
    )
    svc = DiscretionaryCheckService(repo=_FakeRepo(), llm=llm, request=_request())

    await svc.evaluate_lot(lot)

    assert llm.calls == 0
    assert svc.tokens.is_empty()
