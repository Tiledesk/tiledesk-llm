"""
Prompt per la valutazione di criteri discrezionali (L. 36/2023).

Il judge valuta UN criterio alla volta contro le evidenze recuperate dal vector store.
L'output è un JSON con: coefficient, measured_value, motivation, confidence,
source_chunk_index, evidence_text.

Modalità supportate:
  - variabile    : coefficiente 0.0–1.0 (scala ancorata) × max_points
  - proporzionale: estrazione del valore misurabile; nessun punteggio
  - on_off       : coefficiente 1.0 (presente) o 0.0 (assente) × max_points
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Modello di output del judge (usato per parsing/validazione della risposta)
# ---------------------------------------------------------------------------

class DiscretionaryJudgeOutput(BaseModel):
    """
    Output strutturato del judge LLM per un criterio discrezionale.

    Usato sia per il parsing manuale del JSON grezzo (tutti i provider)
    sia con `with_structured_output` (provider che lo supportano).
    """
    coefficient: Optional[float] = Field(
        default=None,
        description=(
            "Coefficiente proposto: float 0.0–1.0 per 'variabile'/'on_off'; "
            "null per 'proporzionale'."
        ),
    )
    measured_value: Optional[str] = Field(
        default=None,
        description=(
            "Valore misurabile estratto dall'offerta (solo per 'proporzionale'). "
            "Esempio: '14 misure da 4.0 a 10.5 mm'. Null per le altre modalità."
        ),
    )
    motivation: str = Field(
        ...,
        description="Motivazione del punteggio proposto (2-4 frasi in italiano).",
    )
    confidence: float = Field(
        ...,
        description="Confidenza del giudizio (0.0–1.0).",
    )
    source_chunk_index: int = Field(
        default=0,
        description=(
            "Indice 1-based del chunk [N] che supporta meglio il giudizio. "
            "0 se nessun chunk è rilevante."
        ),
    )
    evidence_text: str = Field(
        default="",
        description="Citazione verbatim dal chunk selezionato (max 300 caratteri). Stringa vuota se assente.",
    )

    @field_validator("coefficient")
    @classmethod
    def _clamp_coefficient(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return max(0.0, min(1.0, v))
        return v

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DISCRETIONARY_JUDGE_SYSTEM_PROMPT = """\
Sei un commissario esperto di gare d'appalto pubbliche (D.Lgs. 36/2023) specializzato \
nella valutazione tecnica delle offerte.

Il tuo compito è valutare se e quanto la documentazione tecnica dell'operatore economico \
soddisfa un criterio discrezionale della gara, basandoti ESCLUSIVAMENTE sulle evidenze \
testuali recuperate dal documento dell'offerente.

## Modalità di valutazione

### Modalità VARIABILE
Assegna un coefficiente decimale da 0.0 a 1.0 secondo questa scala ancorata:
  0.00 — requisito completamente assente o non documentato
  0.25 — citazione generica senza specifiche tecniche
  0.50 — presenza sufficiente ma con lacune rilevanti
  0.75 — buona documentazione, risponde alla maggior parte dei sub-criteri
  1.00 — eccellente, documenta pienamente tutti gli aspetti richiesti

Usa valori intermedi (0.1, 0.3, 0.6, 0.8 …) per sfumature tra le soglie.

### Modalità PROPORZIONALE
NON assegnare un coefficiente. Invece:
- Estrai il valore misurabile che determinerà il punteggio (es. numero di taglie, range di misure, anni di follow-up).
- Il punteggio reale sarà calcolato dalla commissione confrontando tutti gli operatori.
- Riporta il valore estratto nel campo `measured_value` (formato: numero + unità o descrizione breve).
- Imposta `coefficient: null`.

### Modalità ON_OFF
Il criterio vale tutto o niente:
  1.0 — il documento dimostra inequivocabilmente il possesso del requisito
  0.0 — il documento non menziona il requisito o lo esclude esplicitamente

## Regole operative

1. Basa la valutazione ESCLUSIVAMENTE sulle evidenze recuperate e fornite. Non usare conoscenze pregresse.
2. Se le evidenze non contengono informazioni pertinenti: coefficient = 0.0, confidence ≤ 0.2.
3. Cita sempre un testo verbatim nell'evidence_text (max 300 caratteri).
4. La motivazione deve essere in italiano, 2–4 frasi, e spiegare concretamente il ragionamento.
5. La confidence deve riflettere la chiarezza dell'evidenza (non la qualità dell'offerta):
   - Alta (> 0.8) se l'evidenza è esplicita e inequivocabile
   - Media (0.4–0.8) se l'evidenza è implicita o parziale
   - Bassa (< 0.4) se l'evidenza è vaga o assente

RISPONDI con un singolo oggetto JSON valido — nessun fence markdown, nessun preambolo — \
con esattamente queste chiavi:
  "coefficient"        : float 0.0–1.0 oppure null (solo per proporzionale)
  "measured_value"     : stringa con valore estratto oppure null
  "motivation"         : stringa (2–4 frasi in italiano)
  "confidence"         : float 0.0–1.0
  "source_chunk_index" : intero (1-based, 0 se nessun chunk rilevante)
  "evidence_text"      : stringa (citazione verbatim, max 300 char, stringa vuota se assente)\
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

DISCRETIONARY_JUDGE_USER_TEMPLATE = """\
<criterion>
ID: {criterion_id}
Testo: {criterion_text}
Modalità: {mode}
Punteggio massimo: {max_points}
</criterion>

<retrieved_evidence>
{evidence_block}
</retrieved_evidence>

Valuta il criterio basandoti ESCLUSIVAMENTE sulle evidenze sopra.
{mode_specific_instruction}
Rispondi con un singolo oggetto JSON valido.\
"""

# Istruzioni aggiuntive per modalità specifiche (inserite nel template)
_MODE_INSTRUCTIONS: dict[str, str] = {
    "variabile": (
        "Assegna un coefficiente 0.0–1.0 secondo la scala ancorata del sistema prompt. "
        "Ricorda: 0.5 = sufficiente, 0.75 = buono, 1.0 = eccellente."
    ),
    "proporzionale": (
        "NON assegnare un coefficiente (imposta coefficient: null). "
        "Estrai e riporta in measured_value il valore misurabile presente nell'offerta "
        "(es. numero di taglie/misure, range dimensionale, anni di follow-up). "
        "Se il valore non è presente, imposta measured_value: null e confidence ≤ 0.2."
    ),
    "on_off": (
        "Assegna coefficient = 1.0 se il documento dimostra il possesso del requisito, "
        "coefficient = 0.0 se è assente o non documentato. Nessun valore intermedio."
    ),
}


def build_judge_user_prompt(
    criterion_id: str,
    criterion_text: str,
    mode: str,
    max_points: float,
    evidence_block: str,
) -> str:
    """Costruisce il messaggio utente per il judge LLM."""
    mode_instruction = _MODE_INSTRUCTIONS.get(mode, "")
    return DISCRETIONARY_JUDGE_USER_TEMPLATE.format(
        criterion_id=criterion_id,
        criterion_text=criterion_text,
        mode=mode,
        max_points=max_points,
        evidence_block=evidence_block,
        mode_specific_instruction=mode_instruction,
    )
