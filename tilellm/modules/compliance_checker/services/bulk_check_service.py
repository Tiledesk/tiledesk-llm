"""
BulkComplianceService — massive multi-operator evaluation for a single lot.

Runs the per-operator v2 check (one namespace each) in parallel, then resolves the
`proporzionale` criteria across operators ("ampiezza di gamma"): the operator with the
largest comparable quantity gets the full points, the others get a proportional share.

Per the agreed policy the proportional score is **semi-automatic**: the proposal is
filled in (`score` + `proportional_auto=True`) but the result stays
`human_review_required=True` so a human confirms it before it becomes final.

Entry point: `check_compliance_v2_bulk(request)`.
"""
import asyncio
import logging
from typing import List

from tilellm.modules.compliance_checker.models_v2 import (
    BulkComplianceReport,
    BulkComplianceRequestV2,
    BulkOperatorReport,
    ComplianceReportV2,
    DiscretionaryDirection,
    DiscretionaryMode,
)
from tilellm.modules.compliance_checker.services.discretionary_check_service import (
    check_compliance_v2,
)
from tilellm.shared import token_tracking

logger = logging.getLogger(__name__)


def resolve_proportional(reports: List[ComplianceReportV2]) -> None:
    """
    Resolve `proporzionale` scores across operators, mutating the DiscretionaryResult
    objects in place.

    For each proportional criterion: **diretto** (default) — qmax = max measured_quantity
    across operators, each operator's proposed score = (q / qmax) × max_points (the
    largest quantity wins). **inverso** — qmin = min measured_quantity across operators,
    score = (qmin / q) × max_points (the smallest quantity wins, e.g. "minor temperatura
    di polimerizzazione", "minor tempo di miscelazione"). Operators without a measurable
    quantity are left unscored (human review). If no operator has a usable quantity the
    criterion is left untouched.
    """
    by_criterion: dict = {}
    for rep in reports:
        for d in rep.discretionary_results:
            if d.mode == DiscretionaryMode.PROPORZIONALE:
                by_criterion.setdefault(d.criterion_id, []).append(d)

    for criterion_id, results in by_criterion.items():
        # All results for the same criterion_id carry the same direction (it's a
        # per-criterion, not per-operator, property) — any one is representative.
        direction = results[0].direction
        quantities = [d.measured_quantity for d in results if d.measured_quantity is not None]
        if direction == DiscretionaryDirection.INVERSO:
            positive = [q for q in quantities if q > 0]
            reference = min(positive) if positive else None
        else:
            reference = max(quantities) if quantities else None
        if not reference or reference <= 0:
            logger.info(
                "Proporzionale '%s' (%s): nessuna quantità confrontabile tra gli operatori — "
                "lasciato in revisione umana.", criterion_id, direction.value,
            )
            continue
        for d in results:
            if d.measured_quantity is None:
                continue
            if direction == DiscretionaryDirection.INVERSO:
                if d.measured_quantity <= 0:
                    continue  # can't divide by a non-positive quantity
                d.score = round(min(reference / d.measured_quantity, 1.0) * d.max_points, 2)
                comparison = f"valore {d.measured_quantity:g} su minimo {reference:g}"
            else:
                d.score = round((d.measured_quantity / reference) * d.max_points, 2)
                comparison = f"valore {d.measured_quantity:g} su massimo {reference:g}"
            d.proportional_auto = True
            d.human_review_required = True
            d.human_review_reason = (
                f"Punteggio proporzionale ({direction.value}) calcolato sul confronto tra "
                f"operatori ({comparison}): proposta da confermare."
            )


async def check_compliance_v2_bulk(request: BulkComplianceRequestV2) -> BulkComplianceReport:
    """Evaluate one lot across all operators and resolve proportional criteria."""
    semaphore = asyncio.Semaphore(request.max_concurrent_operators)

    async def _run(operator):
        async with semaphore:
            # Pass the OperatorRef, not just the namespace: each OE files its own L01.
            report = await check_compliance_v2(request.to_operator_request(operator))
            return operator, report

    results = await asyncio.gather(*[_run(op) for op in request.operators])

    reports = [report for _, report in results]
    resolve_proportional(reports)

    first = reports[0]
    operators = [
        BulkOperatorReport(
            namespace=op.namespace,
            operator_label=op.operator_label or op.namespace,
            report=report,
        )
        for op, report in results
    ]
    # Per-operator analytics were already emitted by each check_compliance_v2 call.
    # Here we only roll up the per-operator debug token_usage into a bulk aggregate.
    bulk_token_usage = None
    if request.debug:
        bulk_token_usage = token_tracking.aggregate_token_usage(
            [report.token_usage for report in reports]
        )

    return BulkComplianceReport(
        lot_id=first.tender.lot_id,
        lot_name=first.tender.lot_name,
        operators=operators,
        token_usage=bulk_token_usage,
    )
