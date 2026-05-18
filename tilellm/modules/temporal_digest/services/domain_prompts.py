"""
Pre-built digest generation prompts for common domains.
Add new domains here; reference them via DigestGenerationRequest.domain.
"""
from typing import Optional

_PROMPTS: dict[str, dict] = {
    "pa_italiana": {
        "system": (
            "Sei un assistente esperto in atti amministrativi della Pubblica Amministrazione italiana. "
            "Il tuo compito è produrre un riepilogo ESAUSTIVO e COMPLETO delle attività amministrative "
            "svolte nel periodo indicato, partendo dai frammenti di determine, delibere e altri atti.\n\n"
            "REGOLE OBBLIGATORIE:\n"
            "1. Elenca TUTTI gli atti presenti nei frammenti, senza omissioni.\n"
            "2. Per ogni atto riporta: tipo, oggetto, importo (se presente), CIG/CUP (se presenti), "
            "responsabile (RUP/Direttore se indicato).\n"
            "3. Non scrivere MAI 'non ho informazioni sufficienti' o frasi analoghe.\n"
            "4. Se un'informazione specifica non compare, dì 'non risulta nei documenti analizzati'.\n"
            "5. Concludi con un riepilogo finanziario (totale importi, se disponibili)."
        ),
        "user_template": (
            "Di seguito sono riportati {chunk_count} frammenti di atti amministrativi del namespace '{namespace}' "
            "relativi al periodo {date_from} – {date_to}.\n\n"
            "{evidence}\n\n"
            "Produci un riepilogo strutturato COMPLETO con:\n"
            "1. **Elenco atti** — uno per riga con: tipo | oggetto | importo | CIG/CUP | responsabile\n"
            "2. **Categorie di attività** con conteggio "
            "(acquisti farmaci, dispositivi medici, assunzione personale, appalti, liquidazioni, ecc.)\n"
            "3. **Importi** — elenca TUTTI gli importi presenti, poi il totale\n"
            "4. **Atti di rilievo** — importi elevati, figure chiave, gare significative\n\n"
            "Usa formato markdown. Sii esaustivo: includi TUTTI gli atti presenti nei frammenti."
        ),
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
    },
}

_DEFAULT_DOMAIN = "generic"


def get_domain_prompts(domain: Optional[str]) -> dict:
    """Return (system_prompt, user_template) for the given domain key."""
    return _PROMPTS.get(domain or _DEFAULT_DOMAIN, _PROMPTS[_DEFAULT_DOMAIN])
