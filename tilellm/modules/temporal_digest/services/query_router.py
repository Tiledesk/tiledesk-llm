"""
Query router for temporal_digest: classifies a natural language question
as 'temporal' (needs digest/aggregation retrieval) or 'semantic' (standard
vector search on raw chunks).
"""
import re
from typing import Literal

QueryMode = Literal["temporal", "semantic"]

# Italian + English temporal/aggregative patterns
_TEMPORAL_PATTERNS = [
    # Explicit time references
    r"\boggi\b", r"\bieri\b", r"\bdomani\b",
    r"\bstamattina\b", r"\bstasera\b",
    r"\bquesta\s+(?:settimana|giornata|mattina|sera)\b",
    r"\bquesto\s+(?:mese|anno|trimestre|semestre)\b",
    r"\bl[\'']?\s*(?:anno|mese|settimana)\s+scorso\b",
    r"\bnell[ae]?\s+giornata\b", r"\bnel\s+corso\b",
    r"\bnel\s+periodo\b", r"\bnel\s+mese\b",
    r"\bultim[ioe]\s+\d+\s+(?:giorni|settimane|mesi)\b",
    r"\bdal\b.{0,40}\b(?:al|fino|a)\b",  # "dal X al Y"
    r"\bdal\s+\d", r"\bdal\s+\w+\s+\d{4}\b",
    r"\bdal\s+(?:primo|1°?)\b",
    r"\bfino\s+a\b", r"\bentro\s+il\b",
    # Month names (IT)
    r"\b(?:gennaio|febbraio|marzo|aprile|maggio|giugno|"
    r"luglio|agosto|settembre|ottobre|novembre|dicembre)\b",
    # English time refs
    r"\btoday\b", r"\byesterday\b", r"\bthis\s+(?:week|month|year)\b",
    r"\blast\s+(?:week|month|year)\b", r"\bsince\b.{0,20}\d{4}",
    # Aggregative intent
    r"\briassunto\b", r"\briepilogo\b", r"\bsommario\b", r"\bsintesi\b",
    r"\bcosa\s+hanno\s+(?:fatto|deliberato|deciso|acquistato|assunto)\b",
    r"\bcosa\s+[èe]\s+stato\b", r"\bcosa\s+[èe]\s+successo\b",
    r"\btipo\s+di\s+attivit[àa]\b", r"\bche\s+attivit[àa]\b",
    r"\bquante?\s+(?:determine|delibere|atti|acquisti|assunzioni)\b",
    r"\bn[°º]?\s*(?:determine|delibere)\b",
    r"\bquanti\s+(?:medici|infermieri|dipendenti)\b",
    r"\bsummary\b", r"\bsummarize\b", r"\bwhat\s+(?:happened|did\s+they\s+do)\b",
    r"\bhow\s+many\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]


def classify_query(question: str) -> QueryMode:
    """Return 'temporal' if the query requires aggregative/time-based retrieval,
    'semantic' for specific fact-finding queries."""
    for compiled, raw in zip(_COMPILED, _TEMPORAL_PATTERNS):
        if compiled.search(question):
            return "temporal"
    return "semantic"


def classify_query_debug(question: str) -> tuple[QueryMode, str | None]:
    """Like classify_query but also returns the matched pattern string (or None)."""
    for compiled, raw in zip(_COMPILED, _TEMPORAL_PATTERNS):
        if compiled.search(question):
            return "temporal", raw
    return "semantic", None
