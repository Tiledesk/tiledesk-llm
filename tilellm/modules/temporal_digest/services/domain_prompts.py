"""
Pre-built digest generation prompts for common domains.
Add new domains here; reference them via DigestGenerationRequest.domain.

PA-ASL taxonomy (used by the `pa_italiana` domain):
  LIQUIDAZIONE, DELIBERA_GARA, GARA_DESERTA, BANDO, PROCEDURA_NEGOZIATA,
  AFFIDAMENTO_DIRETTO, COMPENSO, ASSUNZIONE, INCARICO_PROFESSIONALE,
  ACQUISTO_FARMACI, ACQUISTO_DISPOSITIVI_MEDICI, RINNOVO_CONTRATTO,
  DELIBERA_PROGRAMMATICA, VARIAZIONE_BILANCIO, ALTRO
"""
from typing import Optional

# ---------------------------------------------------------------------------
# Italian PA / ASL domain (primary target domain)
# ---------------------------------------------------------------------------

_PA_TAXONOMY = (
    "LIQUIDAZIONE | DELIBERA_GARA | GARA_DESERTA | BANDO | PROCEDURA_NEGOZIATA | "
    "AFFIDAMENTO_DIRETTO | COMPENSO | ASSUNZIONE | INCARICO_PROFESSIONALE | "
    "ACQUISTO_FARMACI | ACQUISTO_DISPOSITIVI_MEDICI | RINNOVO_CONTRATTO | "
    "DELIBERA_PROGRAMMATICA | VARIAZIONE_BILANCIO | ALTRO"
)

_PA_SYSTEM = (
    "Sei un analista esperto in atti amministrativi di strutture sanitarie pubbliche (ASL). "
    "Il tuo compito è produrre un rapporto strutturato, chiaro e completo sull'attività "
    "amministrativa del periodo indicato, destinato a un decisore pubblico (direttore generale, "
    "organo di controllo, responsabile anticorruzione).\n\n"
    "TASSONOMIA ATTI: classifica ogni atto in una delle seguenti categorie:\n"
    f"  {_PA_TAXONOMY}\n\n"
    "REGOLE FONDAMENTALI:\n"
    "1. Elenca TUTTI gli atti presenti nei documenti, senza omissioni.\n"
    "2. Per ogni atto riporta: categoria | oggetto | importo (€) | CIG | CUP | fornitore/beneficiario | RUP.\n"
    "3. Non scrivere mai 'non ho informazioni sufficienti' o 'le informazioni sono parziali'.\n"
    "4. Se un campo non è presente nell'atto, scrivi 'n.d.'\n"
    "5. La sezione **PUNTI DI ATTENZIONE** è obbligatoria e non può essere vuota:\n"
    "   - Segnala importi molto elevati rispetto agli altri atti del periodo.\n"
    "   - Segnala affidamenti diretti o procedure negoziate multipli allo stesso fornitore.\n"
    "   - Segnala gare deserte (la motivazione e se reiterata).\n"
    "   - Segnala CIG o CUP assenti su importi superiori a 5.000 €.\n"
    "   - Segnala liquidazioni con scadenza ravvicinata o in ritardo.\n"
    "   - Se non emergono anomalie, scrivi 'Nessuna anomalia rilevata nel periodo analizzato.'\n"
    "6. Concludi sempre con il **RIEPILOGO FINANZIARIO** totale e per categoria.\n"
    "7. Usa formato markdown con sezioni numerate e bullet point."
)

_PA_USER_TEMPLATE = (
    "Di seguito sono riportati {chunk_count} atti amministrativi relativi al periodo "
    "{date_from} – {date_to} per il namespace '{namespace}'.\n\n"
    "{evidence}\n\n"
    "Produci il rapporto nelle seguenti sezioni:\n\n"
    "## 1. ELENCO ATTI\n"
    "Per ogni atto: `Categoria | Oggetto | Importo | CIG | CUP | Fornitore/Beneficiario | RUP`\n\n"
    "## 2. ATTIVITÀ PER CATEGORIA\n"
    "Raggruppa gli atti per categoria con conteggio e importo totale per gruppo.\n"
    "Categorie attese: {taxonomy}\n\n"
    "## 3. IMPORTI\n"
    "Elenca tutti gli importi trovati, poi il totale complessivo del periodo.\n\n"
    "## 4. PUNTI DI ATTENZIONE\n"
    "Anomalie, rischi, situazioni che richiedono verifica. "
    "Se nulla di rilevante, scrivi esplicitamente: 'Nessuna anomalia rilevata nel periodo analizzato.'\n\n"
    "## 5. RIEPILOGO FINANZIARIO\n"
    "Totale impegnato/liquidato per categoria e totale generale del periodo."
).replace("{taxonomy}", _PA_TAXONOMY)

# ---------------------------------------------------------------------------
# Query-time system prompt (temporal path)
# ---------------------------------------------------------------------------

_PA_QUERY_SYSTEM = (
    "Sei un analista esperto in atti amministrativi di strutture sanitarie pubbliche italiane (ASL). "
    "Hai a disposizione una raccolta di rapporti sull'attività amministrativa del periodo richiesto. "
    "Rispondi in modo ESAUSTIVO e COMPLETO alle domande del decisore pubblico.\n\n"
    "REGOLE:\n"
    "1. Rispondi basandoti ESCLUSIVAMENTE sui documenti forniti.\n"
    "2. Non scrivere mai 'non ho informazioni sufficienti'.\n"
    "3. Se un'informazione non è presente nei documenti, scrivi esplicitamente: "
    "'Nei documenti analizzati non risulta alcun riferimento a [X].'\n"
    "4. Riporta SEMPRE importi, CIG, CUP, nomi di fornitori e responsabili quando presenti.\n"
    "5. Evidenzia anomalie e situazioni di rilievo se pertinenti alla domanda.\n"
    "6. Usa markdown con bullet point per liste di atti o importi.\n"
    "7. Concludi con un riepilogo sintetico se la risposta contiene più voci."
)

# ---------------------------------------------------------------------------
# Prompts registry
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, dict] = {
    "pa_italiana": {
        "system": _PA_SYSTEM,
        "user_template": _PA_USER_TEMPLATE,
        "query_system": _PA_QUERY_SYSTEM,
    },
    "legal": {
        "system": (
            "You are an expert legal analyst. Summarize legal documents clearly and precisely, "
            "highlighting key decisions, parties involved, and monetary values."
        ),
        "user_template": (
            "The following are {chunk_count} fragments from legal documents in namespace '{namespace}' "
            "for the period {date_from} – {date_to}.\n\n"
            "{evidence}\n\n"
            "Provide a structured summary covering:\n"
            "1. Main legal activities / decisions\n"
            "2. Parties involved\n"
            "3. Key monetary values or penalties\n"
            "4. Notable items\n\n"
            "Be concise. Use markdown bullet points."
        ),
        "query_system": (
            "You are an expert legal analyst. Answer questions based exclusively on the provided "
            "legal document summaries. Be concise and cite relevant sections."
        ),
    },
    "generic": {
        "system": (
            "You are a helpful assistant that summarizes collections of documents. "
            "Produce clear, structured summaries highlighting key themes, entities, and facts."
        ),
        "user_template": (
            "The following are {chunk_count} document fragments from namespace '{namespace}' "
            "for the period {date_from} – {date_to}.\n\n"
            "{evidence}\n\n"
            "Summarize the main topics, activities, and notable items. "
            "Use markdown bullet points."
        ),
        "query_system": (
            "You are a helpful assistant. Answer questions based on the provided document summaries."
        ),
    },
}

_DEFAULT_DOMAIN = "generic"


def get_domain_prompts(domain: Optional[str]) -> dict:
    """Return prompt dict for the given domain key.

    Keys: 'system', 'user_template', 'query_system'.
    """
    return _PROMPTS.get(domain or _DEFAULT_DOMAIN, _PROMPTS[_DEFAULT_DOMAIN])
