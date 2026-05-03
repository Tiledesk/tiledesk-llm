"""
Pre-built digest generation prompts for common domains.
Add new domains here; reference them via DigestGenerationRequest.domain.
"""
from typing import Optional

_PROMPTS: dict[str, dict] = {
    "pa_italiana": {
        "system": (
            "Sei un assistente esperto in atti amministrativi della Pubblica Amministrazione italiana. "
            "Il tuo compito è sintetizzare in modo chiaro e strutturato le attività amministrative "
            "svolte in un determinato periodo, partendo dai frammenti di determine, delibere e altri atti."
        ),
        "user_template": (
            "Di seguito sono riportati {chunk_count} frammenti di atti amministrativi del namespace '{namespace}' "
            "relativi al periodo {date_from} – {date_to}.\n\n"
            "{evidence}\n\n"
            "Produci un riepilogo strutturato con:\n"
            "1. **Numero totale di atti** (se desumibile)\n"
            "2. **Categorie principali di attività** con conteggio approssimativo "
            "(acquisti farmaci, dispositivi medici, assunzione personale, appalti, liquidazioni, ecc.)\n"
            "3. **Importi rilevanti** (se presenti)\n"
            "4. **Eventuali atti di rilievo** (importi elevati, assunzioni di figure chiave, ecc.)\n\n"
            "Sii conciso ma completo. Usa il formato markdown con bullet points."
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
