"""
LLM-free entity extraction using spaCy NER + noun chunks + PA regex patterns.

For each chunk, extracts:
- Named entities (PER, ORG, LOC, …) from spaCy NER  (model: it_core_news_lg)
- Noun chunks flagged as CONCEPT (when use_noun_chunks=True)
- Italian PA structured entities via regex (CIG, CUP, DATE_IT, MONEY, QUANTITY)

All surface forms are lowercased and stripped for canonicalization.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore

# Module-level cache to avoid reloading the model for every chunk
_nlp_cache: Dict[str, object] = {}

# ---------------------------------------------------------------------------
# Italian PA regex patterns
# ---------------------------------------------------------------------------

_PA_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # CIG: "CIG" + spazio opzionale + 10 char alfanumerico
    ("CIG",      re.compile(r'\bCIG\s*[:\.]?\s*([A-Z0-9]{10})\b', re.IGNORECASE)),
    # CUP: "CUP" + 15 char alfanumerico
    ("CUP",      re.compile(r'\bCUP\s*[:\.]?\s*([A-Z0-9]{15})\b', re.IGNORECASE)),
    # Date italiane: DD/MM/YYYY o DD-MM-YYYY o DD.MM.YYYY
    ("DATE_IT",  re.compile(r'\b(\d{1,2}[/\-\.]\d{2}[/\-\.]\d{4})\b')),
    # Importi con €: € 10.500,00 oppure 10.500,00 Euro/€
    ("MONEY",    re.compile(
        r'€\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?'
        r'|\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:euro|€)',
        re.IGNORECASE,
    )),
    # Quantità: n. 50 / nr. 100 / n° 5 / pz. 10
    ("QUANTITY", re.compile(r'\b(?:n\.?°?|nr\.?|pz\.?)\s*(\d+)\b', re.IGNORECASE)),
]


def _extract_pa_entities(text: str) -> List[Tuple[str, str]]:
    """Apply Italian PA regex patterns on raw text; return (norm, label) pairs."""
    results: List[Tuple[str, str]] = []
    for label, pattern in _PA_PATTERNS:
        for m in pattern.finditer(text):
            # Use group(1) if capturing group exists, else full match
            value = (m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)).strip()
            norm = value.lower()
            if norm:
                results.append((norm, label))
    return results


# ---------------------------------------------------------------------------
# spaCy pipeline
# ---------------------------------------------------------------------------

def _get_nlp(model_name: str):
    if model_name not in _nlp_cache:
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is not installed. Run: pip install spacy")
        try:
            nlp = spacy.load(model_name)  # type: ignore
        except OSError:
            # Graceful fallback: try xx_ent_wiki_sm if the requested model is missing
            fallback = "xx_ent_wiki_sm"
            if model_name != fallback:
                logger.warning(
                    f"spaCy model '{model_name}' not found — falling back to '{fallback}'. "
                    f"Download the full model with: python -m spacy download {model_name}"
                )
                try:
                    nlp = spacy.load(fallback)  # type: ignore
                except OSError:
                    raise OSError(
                        f"Neither '{model_name}' nor '{fallback}' found. "
                        f"Run: python -m spacy download {model_name}"
                    )
            else:
                raise OSError(
                    f"spaCy model '{model_name}' not found. "
                    f"Download it with: python -m spacy download {model_name}"
                )
        _nlp_cache[model_name] = nlp
    return _nlp_cache[model_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(
    text: str,
    spacy_model: str,
    include_types: List[str],
    use_noun_chunks: bool,
) -> List[Tuple[str, str]]:
    """Return (normalized_name, entity_type) pairs extracted from text.

    Pipeline:
      1. spaCy NER  →  PER, ORG, LOC, MISC (and whatever the model supports)
      2. noun chunks  →  CONCEPT  (if use_noun_chunks and model supports it)
      3. Italian PA regex  →  CIG, CUP, DATE_IT, MONEY, QUANTITY

    include_types: if non-empty, only these labels are kept.
    Deduplicates by normalized form across all three steps.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp(spacy_model)
    doc = nlp(text)  # type: ignore

    seen: Set[str] = set()
    entities: List[Tuple[str, str]] = []

    # ---- 1. spaCy NER -------------------------------------------------------
    for ent in doc.ents:
        if include_types and ent.label_ not in include_types:
            continue
        norm = ent.text.strip().lower()
        if len(norm) > 2 and norm not in seen:
            seen.add(norm)
            entities.append((norm, ent.label_))

    # ---- 2. Noun chunks (CONCEPT) ------------------------------------------
    if use_noun_chunks:
        try:
            noun_chunks_iter = list(doc.noun_chunks)
        except NotImplementedError:
            noun_chunks_iter = []  # model has no dependency parser (e.g. xx_ent_wiki_sm)
        for chunk in noun_chunks_iter:
            if include_types and "CONCEPT" not in include_types:
                continue
            norm = chunk.text.strip().lower()
            if len(norm) <= 2 or norm in seen:
                continue
            if all(tok.is_stop for tok in chunk):
                continue
            seen.add(norm)
            entities.append((norm, "CONCEPT"))

    # ---- 3. Italian PA regex patterns --------------------------------------
    for norm, label in _extract_pa_entities(text):
        if include_types and label not in include_types:
            continue
        if norm not in seen:
            seen.add(norm)
            entities.append((norm, label))

    return entities


# ---------------------------------------------------------------------------
# Italian stop words (used for query keyword extraction)
# ---------------------------------------------------------------------------

_IT_STOP_WORDS: set = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "da", "in", "con", "su", "per", "tra", "fra", "a",
    "e", "o", "ma", "che", "se", "del", "della", "dei", "degli",
    "delle", "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dalla", "dai", "dagli", "dalle", "nel", "nella",
    "nei", "negli", "nelle", "sul", "sulla", "sui", "sugli",
    "sulle", "col", "come", "non", "si", "ho", "ha", "è",
    "sono", "era", "uno", "una", "ogni", "questo", "questa",
    "questi", "queste", "quello", "quella", "quelli", "quelle",
    "altro", "altri", "altre", "cui", "che", "chi", "quando",
    "dove", "anche", "già", "più", "meno", "così", "come",
}

# ---------------------------------------------------------------------------
# Italian month-year date expansion
# ---------------------------------------------------------------------------

_MONTH_NAMES_IT: Dict[str, int] = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}

_MONTH_YEAR_RE = re.compile(
    r'\b(' + '|'.join(_MONTH_NAMES_IT.keys()) + r')\s+(\d{4})\b',
    re.IGNORECASE,
)


def expand_date_references(text: str) -> List[str]:
    """Convert 'aprile 2026' → list of DATE_IT strings '01/04/2026' … '30/04/2026'.

    Useful as additional PPR seeds for queries that express dates in natural
    language instead of the DD/MM/YYYY format stored in the graph.
    """
    import calendar
    results: List[str] = []
    for m in _MONTH_YEAR_RE.finditer(text):
        month = _MONTH_NAMES_IT[m.group(1).lower()]
        year = int(m.group(2))
        _, days_in_month = calendar.monthrange(year, month)
        for day in range(1, days_in_month + 1):
            results.append(f"{day:02d}/{month:02d}/{year}")
    return results


def extract_query_keywords(text: str, min_length: int = 4) -> List[str]:
    """Extract content-bearing keywords from a short query string.

    Splits on whitespace/punctuation, removes Italian stop words and tokens
    shorter than min_length.  Used as a last-resort seed list when NLP entity
    extraction produces nothing.
    """
    tokens = re.split(r'[\s,;:.!?()\[\]]+', text.lower())
    return [
        t for t in tokens
        if len(t) >= min_length and t not in _IT_STOP_WORDS
    ]


def _split_into_subwindows(text: str, window_size: int, overlap: int) -> List[str]:
    """Split text into overlapping character-level sub-windows (Percorso A)."""
    if not text or window_size <= 0:
        return [text] if text else []
    step = max(1, window_size - overlap)
    windows = []
    for start in range(0, len(text), step):
        windows.append(text[start:start + window_size])
    return windows


def build_chunk_entity_matrix(
    chunks: List[Dict],
    spacy_model: str,
    include_types: List[str],
    use_noun_chunks: bool,
    sub_window_size: int = 0,
    sub_window_overlap: int = 50,
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[Tuple[str, str], int]]:
    """Process all chunks in batch and return entity occurrence data.

    When sub_window_size > 0, each chunk text is split into overlapping sub-windows
    before entity extraction. Entities from all sub-windows are unioned (deduped by
    normalized form) and attributed to the original chunk_id.  The LChunk node in
    FalkorDB still stores the full original text for LLM context.

    Returns:
        chunk_entities: chunk_id → [(normalized_name, etype), …]
        entity_doc_freq: (normalized_name, etype) → number of chunks containing it
    """
    chunk_entities: Dict[str, List[Tuple[str, str]]] = {}
    entity_doc_freq: Counter = Counter()

    for chunk in chunks:
        chunk_id = chunk["id"]
        text = chunk.get("text", "")
        if not text:
            chunk_entities[chunk_id] = []
            continue

        if sub_window_size > 0:
            windows = _split_into_subwindows(text, sub_window_size, sub_window_overlap)
        else:
            windows = [text]

        seen: Set[str] = set()
        entities: List[Tuple[str, str]] = []
        for window in windows:
            for norm, etype in extract_entities(window, spacy_model, include_types, use_noun_chunks):
                if norm not in seen:
                    seen.add(norm)
                    entities.append((norm, etype))

        chunk_entities[chunk_id] = entities
        for e in set(entities):
            entity_doc_freq[e] += 1

    return chunk_entities, dict(entity_doc_freq)
