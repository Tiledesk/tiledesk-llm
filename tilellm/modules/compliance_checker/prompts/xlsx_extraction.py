"""
Prompt per l'estrazione strutturata dei requisiti di gara da file xlsx.

Il servizio XlsxExtractionService passa il contenuto testuale delle celle
(estratto con openpyxl, un sheet alla volta) all'LLM che deve restituire
la struttura di un lotto in formato JSON.

Uso: `llm.with_structured_output(_LLMLotExtractionResult)`
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Modelli per l'output strutturato dell'LLM (lightweight — nessun validator)
# ---------------------------------------------------------------------------

class _LLMTabularItem(BaseModel):
    """Requisito tabellare: presenza/assenza a pena di esclusione."""
    id: str = Field(..., description="Identificatore univoco (es. 'REQ-001', '1', 'A').")
    text: str = Field(..., description="Testo completo del requisito.")
    mandatory: bool = Field(default=True, description="True se obbligatorio a pena di esclusione.")


class _LLMDiscretionaryCriterion(BaseModel):
    """Criterio discrezionale oggetto di punteggio."""
    id: str = Field(..., description="Identificatore univoco (es. 'P1', 'a', '1').")
    text: str = Field(..., description="Testo completo del criterio.")
    mode: str = Field(
        ...,
        description=(
            "Modalità di attribuzione del punteggio. "
            "Valori ammessi: 'variabile' (coefficiente soggettivo), "
            "'proporzionale' (comparativo tra offerte), 'on_off' (presenza/assenza con punteggio)."
        ),
    )
    max_points: float = Field(..., description="Punteggio massimo attribuibile al criterio.")
    human_only: bool = Field(
        default=False,
        description=(
            "True se il criterio è soggettivo e NON può essere valutato da un sistema automatico. "
            "Segnali tipici: 'soggettivo', 'non gestibile da IA', 'maneggevolezza', "
            "'ergonomia', 'facilità d'uso percepita', 'qualità visiva'."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description="Note aggiuntive o annotazioni presenti nel foglio (es. spiegazione del criterio).",
    )


class _LLMLotExtractionResult(BaseModel):
    """Risultato dell'estrazione strutturata per un singolo lotto/sheet."""
    lot_id: str = Field(
        ...,
        description="Identificatore del lotto (es. '6', 'Lotto 6', '02'). Derivare dal nome del sheet o dal testo.",
    )
    lot_name: str = Field(
        ...,
        description="Nome descrittivo del lotto (es. 'Tubi endotracheali armati'). Derivare dal titolo o dal contenuto.",
    )
    tabular: List[_LLMTabularItem] = Field(
        default_factory=list,
        description="Requisiti minimi / caratteristiche di conformità (a pena di esclusione).",
    )
    discretionary: List[_LLMDiscretionaryCriterion] = Field(
        default_factory=list,
        description="Criteri di valutazione qualitativa con punteggio.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Avvisi sull'estrazione (es. righe ambigue, criteri marcati human_only, colonne mancanti).",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

XLSX_EXTRACTION_SYSTEM_PROMPT = """\
Sei un esperto di gare d'appalto pubbliche italiane (D.Lgs. 36/2023 — Codice dei Contratti Pubblici).
Il tuo compito è analizzare il contenuto testuale di un foglio Excel estratto da un capitolato tecnico \
e strutturarlo in requisiti di gara.

## Classificazione dei requisiti

Devi distinguere due tipologie:

### 1. Requisiti tabellari (conformità a pena di esclusione)
Sono i requisiti MINIMI che il prodotto/servizio deve obbligatoriamente possedere.
Segnali tipici nel foglio:
- Sezioni intestate "caratteristiche di conformità", "requisiti minimi", "a pena di esclusione"
- Elenco di proprietà senza colonne modalità/punteggio (es. "sterile", "monouso", "latex free")
- La risposta attesa è SI/NO

### 2. Criteri discrezionali (valutazione qualitativa con punteggio)
Sono i criteri che la commissione usa per attribuire un punteggio all'offerta tecnica.
Segnali tipici nel foglio:
- Sezione intestata "Criteri di valutazione", "requisiti premiali", "criteri di aggiudicazione"
- Presenza di colonne "modalità" e "punt. max" (o simili)
- I criteri hanno un identificatore (P1, a), b), 1, …) e un punteggio massimo

## Modalità dei criteri discrezionali

Determina la modalità da una delle tre seguenti:

| Modalità | Descrizione | Segnali nel foglio |
|----------|-------------|-------------------|
| variabile | Punteggio soggettivo 0–max, assegnato discrezionalmente dalla commissione | Colonna "modalità" = "variabile" o assente con criterio qualitativo |
| proporzionale | Punteggio proporzionale alla quantità offerta (confronto tra operatori) | Colonna "modalità" = "proporzionale" o testo che indica "maggior ampiezza", "in proporzione", "max fornitori" |
| on_off | Presenza/assenza con punteggio fisso (tutto o niente) | Colonna "modalità" = "on/off", "on_off", oppure "certificazione" con 1 pt |

## Criteri soggettivi (human_only = true)

Imposta `human_only: true` quando il criterio richiede valutazione sensoriale o soggettiva non verificabile \
da un testo scritto:
- Frasi come: "non gestibile da IA", "soggettivo", "al tatto", "maneggevolezza", "ergonomicità percepita"
- Criteri che richiedono ispezione fisica del campione

## Regole operative

1. Estrai TUTTI i criteri presenti, senza omissioni.
2. Per i punteggi: leggi il valore numerico dalla colonna "punt. max" o dal testo (es. "max 8 punti" → 8).
3. Se lo stesso testo ha più criteri in celle distinte, creane uno per ogni cella.
4. Se il lot_id/lot_name non è esplicito, derivalo dal nome del sheet o dal titolo del foglio.
5. Inserisci in `warnings` qualsiasi ambiguità rilevata (modalità non riconoscibile, punteggio non trovato, ecc.).
6. Rispondi ESCLUSIVAMENTE con l'oggetto JSON strutturato richiesto — nessun preambolo, nessun commento.\
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

XLSX_EXTRACTION_USER_TEMPLATE = """\
Analizza il seguente contenuto estratto dal foglio Excel "{sheet_name}":

<sheet_content>
{sheet_content}
</sheet_content>

Struttura tutti i requisiti in formato JSON secondo lo schema fornito.\
"""
