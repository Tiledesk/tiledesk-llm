# tool_registry.py
from typing import Dict, Any, Callable, List
from langchain_core.tools import BaseTool

from tilellm.modules.conversion.services.conversion_service import (
    convert_pdf_to_text_tool,
    convert_xlsx_to_csv_tool,
    convert_pdf_to_images_tool
)
from tilellm.tools.multimodal_llm_tool import create_multimodal_llm_tool

# Tipo per chiarezza: una factory è una funzione che restituisce un BaseTool
ToolFactory = Callable[..., BaseTool]

# --- Il REGISTRO CENTRALE ---
# Mappa: nome_tool -> {descrizione, implementazione, is_factory}
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {

    "convert_pdf_to_text": {
        "description": "Extracts plain text from PDF files. Accepts URLs, base64, or data URIs.",
        "implementation": convert_pdf_to_text_tool,
        "is_factory": False  # È già un tool decorato con @tool
    },

    "convert_xlsx_to_csv": {
        "description": "Converts Excel (XLSX) files to CSV format. Accepts URLs, base64, or data URIs.",
        "implementation": convert_xlsx_to_csv_tool,
        "is_factory": False  # È già un tool decorato con @tool
    },

    "convert_pdf_to_images": {
        "description": "Converts PDF files to PNG images (one per page). Accepts URLs, base64, or data URIs. Useful for visual analysis and OCR.",
        "implementation": convert_pdf_to_images_tool,
        "is_factory": False  # È già un tool decorato con @tool
    },

    "multimodal_tool": {
        "description": "Analyzes images, documents and text with multimodal capabilities.",
        "implementation": create_multimodal_llm_tool,  # Questa è una funzione (factory)
        "is_factory": True  # Flag che ci dice di CHIAMARE questa funzione
    }

    # ... aggiungi qui tutti gli altri tuoi tool
}


# --- Funzioni Helper per usare il Registro ---

def get_available_tools_list() -> List[Dict[str, str]]:
    """
    Funzione per l'API REST.
    Restituisce solo i nomi e le descrizioni per l'utente.
    """
    return [
        {"name": name, "description": data["description"]}
        for name, data in TOOL_REGISTRY.items()
    ]


def resolve_tools(
        selected_tool_names: List[str],
        **factory_kwargs: Any  # Argomenti di runtime (es. chat_model, base64_storage)
) -> List[BaseTool]:
    """
    Funzione per il Backend.
    Dato un elenco di nomi di tool, li istanzia e li restituisce.
    """
    resolved_tools: List[BaseTool] = []

    for name in selected_tool_names:
        config = TOOL_REGISTRY.get(name)

        if not config:
            print(f"Attenzione: Tool '{name}' richiesto ma non trovato nel registro.")
            continue

        implementation = config["implementation"]

        if config["is_factory"]:
            # È una factory (es. create_multimodal_llm_tool)
            # Dobbiamo chiamarla, passando gli argomenti di runtime
            try:
                # La factory riceverà tutti i kwargs (chat_model, ecc.)
                # e userà solo quelli che le servono.
                tool_instance = implementation(**factory_kwargs)
                resolved_tools.append(tool_instance)
            except TypeError as e:
                # Errore comune se mancano argomenti (es. chat_model non passato)
                print(f"Errore nell'istanza della factory '{name}': {e}")
        else:
            # È un oggetto BaseTool statico, basta aggiungerlo
            resolved_tools.append(implementation)

    return resolved_tools