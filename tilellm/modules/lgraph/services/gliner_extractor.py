"""
GLiNER-based entity extraction — alternative NER backend to spaCy, selectable
per request via ner_backend="gliner" (A6, docs/GRAPHRAG_COST_QUALITY_PLAN.md
§4a/§8). Zero-shot: labels are plain-language strings passed at inference
time, no fine-tuning.

Same output contract as entity_extractor.extract_entities — a list of
(normalized_name, entity_type) pairs — so nothing downstream (FalkorDB
storage, NPMI, Leiden, PPR, the §6d spaCy-vs-GLiNER comparison) needs to
know which backend produced them.

Scope: only the model-based types (PER, ORG, LOC, MISC) go through GLiNER.
CIG, CUP, CF, DATE_IT, MONEY, QUANTITY stay deterministic regex — already
shared via entity_extractor._extract_pa_entities regardless of backend, no
reason to ask a model for what a regex gets exactly right. CONCEPT (spaCy's
noun-chunk parser) has no GLiNER equivalent — ponytail: not implemented,
dropped silently like any other type absent from include_types would be.
"""
import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    GLiNER = None  # type: ignore

# Module-level cache to avoid reloading the model for every chunk (mirrors
# entity_extractor._nlp_cache).
_model_cache: Dict[str, object] = {}

DEFAULT_GLINER_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_THRESHOLD = 0.5

# GLiNER performs best with descriptive natural-language labels, not short
# codes — but for a fair A4-vs-A6 comparison (same include_entity_types, only
# the extractor changes) the *type it returns* stays PER/ORG/LOC/MISC, the
# same vocabulary spaCy produces. Retuning MISC into finer PA-specific labels
# is future work, not this step.
_TYPE_TO_LABEL: Dict[str, str] = {
    "PER": "persona",
    "ORG": "organizzazione o azienda",
    "LOC": "luogo o indirizzo",
    "MISC": "altro",
}
_LABEL_TO_TYPE: Dict[str, str] = {v: k for k, v in _TYPE_TO_LABEL.items()}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        if not GLINER_AVAILABLE:
            raise ImportError(
                "gliner is not installed. Run: pip install gliner "
                "(or `poetry install -E graph`, which bundles it)."
            )
        _model_cache[model_name] = GLiNER.from_pretrained(model_name)
    return _model_cache[model_name]


def extract_entities_gliner(
    text: str,
    include_types: List[str],
    model_name: str = DEFAULT_GLINER_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Tuple[str, str]]:
    """Model-based subset only (PER/ORG/LOC/MISC) — the caller (extract_entities
    in entity_extractor.py) adds the regex types and dedupes across both.
    """
    if not text or not text.strip():
        return []

    model_types = [t for t in _TYPE_TO_LABEL if not include_types or t in include_types]
    if not model_types:
        return []
    labels = [_TYPE_TO_LABEL[t] for t in model_types]

    model = _get_model(model_name)
    predictions = model.predict_entities(text, labels, threshold=threshold)

    seen: Set[str] = set()
    entities: List[Tuple[str, str]] = []
    for pred in predictions:
        etype = _LABEL_TO_TYPE.get(pred["label"], "MISC")
        norm = pred["text"].strip().lower()
        if len(norm) > 2 and norm not in seen:
            seen.add(norm)
            entities.append((norm, etype))
    return entities
