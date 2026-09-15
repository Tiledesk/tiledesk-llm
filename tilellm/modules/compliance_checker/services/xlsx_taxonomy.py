"""
Shared taxonomy + cell helpers for the standardized Compliance v2 xlsx templates.

The de-facto market template (`ipotesi tabella criteri.xlsx`) classifies each
criterion with a 3-valued "Tipo criterio":

    Conformità   → requisito minimo a pena di esclusione   → internal `tabular`
    Tabellare    → punteggio attribuito da una regola      → internal `discretionary` (on_off | proporzionale)
    Discrezionale→ punteggio soggettivo della commissione   → internal `discretionary` (variabile)

NOTE the word "Tabellare" means a *scored* criterion here — the OPPOSITE of how an
earlier internal draft used it. This module is the single source of truth for the
mapping so the two meanings never collide again.

The market template does NOT carry an explicit mode column: for "Tabellare" rows the
mode (ON/OFF vs PROPORZIONALE) is embedded in the criterion text. `derive_mode_from_text`
recovers it deterministically when a `Modalità` column is absent.
"""
import re
import unicodedata
from typing import Optional, Tuple

from tilellm.modules.compliance_checker.models_v2 import (
    DiscretionaryCriterion,
    DiscretionaryDirection,
    DiscretionaryMode,
    TabularRequirementV2,
)

# --- Business "Tipo criterio" canonical values ------------------------------
TYPE_CONFORMITA = "Conformità"
TYPE_TABELLARE = "Tabellare"
TYPE_DISCREZIONALE = "Discrezionale"

_TRUE_TOKENS = {"si", "sì", "yes", "y", "true", "vero", "x", "1"}
_FALSE_TOKENS = {"no", "n", "false", "falso", "0", ""}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_type(value) -> Optional[str]:
    """Return the canonical business type, accent/case-insensitive, or None."""
    if value is None:
        return None
    norm = _strip_accents(str(value).strip().lower())
    if norm.startswith("conformit"):
        return TYPE_CONFORMITA
    if norm.startswith("tabellar"):
        return TYPE_TABELLARE
    if norm.startswith("discrezional"):
        return TYPE_DISCREZIONALE
    return None


def normalize_type_and_mode_hint(value) -> Tuple[Optional[str], Optional[DiscretionaryMode]]:
    """Like `normalize_type`, but also recognizes a real-world 4-way variant of the
    market template where "Tipo criterio" carries the mode DIRECTLY — 'Proporzionale',
    'ON/OFF', 'ON-OFF' — instead of the standard 'Tabellare' + a separate 'Modalità'
    column. Returns (canonical_type, mode_hint); mode_hint is only set for this variant
    (None for the standard 3-way Conformità/Tabellare/Discrezionale values, which carry
    no mode information of their own).
    """
    canonical = normalize_type(value)
    if canonical is not None or value is None:
        return canonical, None
    norm = _strip_accents(str(value).strip().lower()).translate(
        {ord(c): None for c in "/-_ "}
    )
    if norm == "proporzionale":
        return TYPE_TABELLARE, DiscretionaryMode.PROPORZIONALE
    if norm == "onoff":
        return TYPE_TABELLARE, DiscretionaryMode.ON_OFF
    return None, None


def model_type(item) -> str:
    """Map an internal requirement/criterion to the business 'Tipo criterio'."""
    if isinstance(item, TabularRequirementV2):
        return TYPE_CONFORMITA
    if isinstance(item, DiscretionaryCriterion):
        if item.mode == DiscretionaryMode.VARIABILE:
            return TYPE_DISCREZIONALE
        return TYPE_TABELLARE
    raise TypeError(f"Unsupported item type: {type(item)!r}")


def model_mode_label(item) -> str:
    """Mode string for the 'Modalità' column ('' for Conformità)."""
    if isinstance(item, DiscretionaryCriterion):
        return item.mode.value
    return ""


def type_from_mode(mode: DiscretionaryMode) -> str:
    """Business 'Tipo criterio' from a discretionary mode (for output rows)."""
    return TYPE_DISCREZIONALE if mode == DiscretionaryMode.VARIABILE else TYPE_TABELLARE


def derive_mode_from_text(text: str) -> Optional[DiscretionaryMode]:
    """Recover the discretionary mode from free criterion text (market template)."""
    if not text:
        return None
    norm = _strip_accents(str(text).lower())
    if re.search(r"on[\s/_\-]?off", norm):
        return DiscretionaryMode.ON_OFF
    if "proporzional" in norm:
        return DiscretionaryMode.PROPORZIONALE
    if "discrezional" in norm:
        return DiscretionaryMode.VARIABILE
    return None


def resolve_mode(
    explicit_mode, criterion_text: str, *, default: DiscretionaryMode
) -> Tuple[DiscretionaryMode, Optional[str]]:
    """
    Resolve a DiscretionaryMode from an explicit cell value or, failing that, from
    the criterion text. Returns (mode, warning_or_None).
    """
    raw = (str(explicit_mode or "")).strip().lower().replace("/", "_").replace(" ", "_")
    valid = {m.value for m in DiscretionaryMode}
    if raw in valid:
        return DiscretionaryMode(raw), None
    derived = derive_mode_from_text(criterion_text)
    if derived is not None:
        return derived, None
    return default, (
        f"Modalità non specificata e non deducibile dal testo — impostata a "
        f"'{default.value}', verificare manualmente."
    )


# Words that hint a criterion is "inverso" (lower value wins) — e.g. "minor
# temperatura", "tempo di miscelazione più basso". Advisory only: it never
# overrides an unset direction, it only warns the reviewer to check.
_INVERSE_HINT_RE = re.compile(
    r"\bminor[ei]?\b|\bpiu\s+bass[oa]\b|\binferior[ei]\b|\bpiu\s+breve\b|\bpiu\s+piccol[oa]\b|\bminim[oa]\b"
)


def resolve_direction(
    explicit_direction, criterion_text: str, *,
    default: DiscretionaryDirection = DiscretionaryDirection.DIRETTO,
) -> Tuple[DiscretionaryDirection, Optional[str]]:
    """
    Resolve a DiscretionaryDirection from an explicit cell value.

    Unlike `resolve_mode`, the direction is NEVER inferred from the criterion text
    and silently applied — it must be declared by whoever compiles the criteria
    table (see `DiscretionaryCriterion.direction`). If the cell is blank but the
    text hints at an inverse criterion ("minor...", "più basso", ...), a warning
    is returned so the gap surfaces in review; the resolved value still falls
    back to *default* ('diretto').
    """
    raw = (str(explicit_direction or "")).strip().lower()
    valid = {d.value for d in DiscretionaryDirection}
    if raw in valid:
        return DiscretionaryDirection(raw), None
    if default == DiscretionaryDirection.DIRETTO and _INVERSE_HINT_RE.search(
        _strip_accents(str(criterion_text or "").lower())
    ):
        return default, (
            "Direzione non specificata e il testo sembra indicare un criterio "
            "'inverso' (vince il valore più basso, es. minor temperatura/tempo) — "
            "verificare e impostare esplicitamente 'diretto' o 'inverso'."
        )
    return default, None


def cell_to_bool(value, *, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    # Unknown token → keep the safe default rather than failing the whole import
    return default


def bool_to_cell(value: bool) -> str:
    return "SI" if value else "NO"


def parse_float(value, *, row_id: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Punteggio previsto mancante per il criterio '{row_id}'. "
            f"Inserire un valore numerico > 0."
        )
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"Punteggio previsto '{value}' non numerico per il criterio '{row_id}'."
        )
