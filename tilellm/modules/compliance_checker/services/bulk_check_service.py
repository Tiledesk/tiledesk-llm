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


# Spellings of the same unit a judge may write. Anything else is compared verbatim
# (case-insensitive): an unknown spelling fails the check rather than passing it.
_UNIT_SYNONYMS = {
    "s": "s", "sec": "s", "secondo": "s", "secondi": "s", "second": "s", "seconds": "s", '"': "s",
    "min": "min", "minuto": "min", "minuti": "min", "minute": "min", "minutes": "min", "'": "min",
    "°c": "°c", "c": "°c", "gradi": "°c", "gradi centigradi": "°c", "celsius": "°c", "° c": "°c",
    "mpa": "mpa",
}


def _norm_unit(unit):
    if unit is None or not str(unit).strip():
        return None
    u = " ".join(str(unit).strip().lower().split())
    return _UNIT_SYNONYMS.get(u, u)


def _flag_unscored(d, reason: str) -> None:
    d.score = None
    d.proportional_auto = False
    d.human_review_required = True
    d.human_review_reason = reason


def _comparable(results) -> list:
    """Results whose quantity can be compared. A quantity whose unit can't be
    verified is never scored: in a tender a wrong unit silently rewrites the
    ranking (45 seconds compared as 45 minutes)."""
    measured = [d for d in results if d.measured_quantity is not None]
    declared = _norm_unit(results[0].unit)
    if declared:
        ok = []
        for d in measured:
            got = _norm_unit(d.measured_unit)
            if got == declared:
                ok.append(d)
            else:
                _flag_unscored(d, (
                    f"Quantità {d.measured_quantity:g} espressa in unità "
                    f"'{d.measured_unit or 'non indicata'}' invece di '{results[0].unit}' "
                    f"dichiarata nel criterio: esclusa dal confronto, da verificare."
                ))
        return ok
    units = {_norm_unit(d.measured_unit) for d in measured if d.measured_unit}
    if len(units) > 1:
        for d in measured:
            _flag_unscored(d, (
                f"Unità di misura non omogenee tra gli operatori ({', '.join(sorted(units))}) "
                f"e nessuna unità dichiarata nel criterio: confronto non eseguito, da verificare."
            ))
        return []
    return measured


def resolve_proportional(reports: List[ComplianceReportV2]) -> None:
    """
    Resolve `proporzionale` scores across operators, mutating the DiscretionaryResult
    objects in place.

    For each proportional criterion: **diretto** (default) — qmax = max measured_quantity
    across operators, each operator's proposed score = (q / qmax) × max_points (the
    largest quantity wins). **inverso** — qmin = min measured_quantity across operators,
    score = (qmin / q) × max_points (the smallest quantity wins, e.g. "minor temperatura
    di polimerizzazione", "minor tempo di miscelazione").

    With a declared zero-point `baseline` the score is proportional to the distance
    from it: (q − b)/(best − b) × max_points (inverso: (b − q)/(b − best)); an operator
    that doesn't beat the baseline gets 0.

    Only quantities in a verifiable unit are compared (see `_comparable`). Operators
    without a comparable quantity are left unscored (human review). If no operator has
    one the criterion is left untouched.
    """
    by_criterion: dict = {}
    for rep in reports:
        for d in rep.discretionary_results:
            if d.mode == DiscretionaryMode.PROPORZIONALE:
                by_criterion.setdefault(d.criterion_id, []).append(d)

    for criterion_id, results in by_criterion.items():
        # direction/baseline/unit are per-criterion properties copied on every
        # result — any one is representative.
        direction = results[0].direction
        baseline = results[0].baseline
        inverse = direction == DiscretionaryDirection.INVERSO
        comparable = _comparable(results)
        if baseline is None and inverse:
            comparable = [d for d in comparable if d.measured_quantity > 0]  # can't divide by it
        quantities = [d.measured_quantity for d in comparable]
        if not quantities:
            logger.info(
                "Proporzionale '%s' (%s): nessuna quantità confrontabile tra gli operatori — "
                "lasciato in revisione umana.", criterion_id, direction.value,
            )
            continue
        best = min(quantities) if inverse else max(quantities)
        if baseline is None and best <= 0:
            continue
        unit = f" {results[0].unit}" if results[0].unit else ""
        for d in comparable:
            q = d.measured_quantity
            if baseline is not None:
                gap = (baseline - q) if inverse else (q - baseline)
                span = (baseline - best) if inverse else (best - baseline)
                d.score = round(max(gap, 0) / span * d.max_points, 2) if span > 0 else 0.0
                formula = (
                    f"({baseline:g} − {q:g}) / ({baseline:g} − {best:g})" if inverse
                    else f"({q:g} − {baseline:g}) / ({best:g} − {baseline:g})"
                )
                comparison = (
                    f"valore {q:g}{unit}, soglia a zero punti {baseline:g}{unit}, "
                    f"{'minimo' if inverse else 'massimo'} {best:g}{unit}: {formula} × {d.max_points:g}"
                )
            elif inverse:
                d.score = round(min(best / q, 1.0) * d.max_points, 2)
                comparison = f"valore {q:g}{unit} su minimo {best:g}{unit}"
            else:
                d.score = round((q / best) * d.max_points, 2)
                comparison = f"valore {q:g}{unit} su massimo {best:g}{unit}"
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
