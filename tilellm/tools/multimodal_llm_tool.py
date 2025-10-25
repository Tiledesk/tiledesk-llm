"""
Tool multimodale interno per invocare LLM con supporto a immagini e documenti.

Questo tool consente agli agenti MCP di invocare l'LLM iniettato nel controller
per effettuare operazioni multimodali (analisi di immagini, documenti, etc.)
"""

import logging
import base64
from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class MultimodalLLMInput(BaseModel):
    """Input schema for the multimodal LLM tool."""

    prompt: str = Field(description="The primary text prompt to send to the LLM.")

    images_base64: Optional[List[str]] = Field(
        default=None,
        description="A list of base64-encoded images. The data prefix (e.g., 'data:image/jpeg;base64,') should be omitted."
    )

    documents_base64: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="A list of base64-encoded documents. Each item must be a dict with 'data' (the base64 string) and 'mime_type' (e.g., 'application/pdf', 'text/plain')."
    )

    system_prompt: Optional[str] = Field(
        default="You are a helpful AI assistant. Analyze the provided content and respond accurately.",
        description="The system prompt to guide the LLM's behavior, persona, or response style."
    )

    max_tokens: Optional[int] = Field(
        default=2048,
        description="The maximum number of tokens to generate in the response."
    )

    temperature: Optional[float] = Field(
        default=0.0,
        description="Controls randomness in sampling (0.0 = deterministic, 1.0 = max creativity). Range: 0.0 to 1.0."
    )

class MultimodalLLMInput_old(BaseModel):
    """Input schema per il tool multimodale LLM"""
    prompt: str = Field(description="Il prompt testuale da inviare all'LLM")
    images_base64: Optional[List[str]] = Field(
        default=None,
        description="Lista di immagini in formato base64 (senza prefisso data:image/...)"
    )
    documents_base64: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Lista di documenti in formato base64. Ogni elemento deve contenere 'data' (base64) e 'mime_type' (es. application/pdf)"
    )
    system_prompt: Optional[str] = Field(
        default="You are a helpful AI assistant. Analyze the provided content and respond accurately.",
        description="Prompt di sistema per l'LLM"
    )
    max_tokens: Optional[int] = Field(
        default=2048,
        description="Numero massimo di token nella risposta"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        description="Temperature per il sampling (0.0-1.0)"
    )


def create_multimodal_llm_tool(llm_instance: Any, base64_storage: Optional[Dict[str, Dict]] = None):
    """
    Factory function that creates the multimodal tool by injecting the LLM instance.

    Args:
        llm_instance: The LLM model instance to be used
                      (e.g., ChatOpenAI, ChatAnthropic, etc.).
        base64_storage: Optional storage (e.g., a dict) for base64 references,
                        used to avoid context window overflow with large media.

    Returns:
        The configured tool, ready for use within LangGraph
        (e.g., with create_react_agent).

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o", api_key="...")
        >>> storage = {}  # Shared storage
        >>> multimodal_tool = create_multimodal_llm_tool(llm, storage)
        >>> # The tool can now be used with create_react_agent
    """

    # shared storage support
    if base64_storage is None:
        base64_storage = {}

    # Rileva il provider dall'istanza LLM
    llm_class_name = llm_instance.__class__.__name__
    # OpenAI e Google (Gemini) usano lo stesso formato image_url con data URI
    is_openai = "OpenAI" in llm_class_name or "ChatOpenAI" in llm_class_name
    is_google = "Google" in llm_class_name or "ChatGoogle" in llm_class_name
    uses_openai_format = is_openai or is_google
    logger.info(f"Multimodal tool created for provider: {llm_class_name}, uses_openai_format={uses_openai_format}")

    @tool(args_schema=MultimodalLLMInput)
    async def invoke_multimodal_llm(
        prompt: str,
        images_base64: Optional[List[str]] = None,
        documents_base64: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "You are a helpful AI assistant. Analyze the provided content and respond accurately.",
        max_tokens: int = 2048,
        temperature: float = 0.0
    ) -> str:
        """
        Invoca un LLM multimodale con supporto per immagini e documenti.

        Questo tool interno consente all'agente di analizzare contenuti multimodali
        utilizzando l'LLM configurato nel sistema. È particolarmente utile quando
        un tool MCP esterno fornisce dati (es. un file convertito) e l'agente deve
        poi analizzare questi dati con capacità multimodali.

        Args:
            prompt: Il testo della domanda/richiesta
            images_base64: Lista opzionale di immagini in base64
            documents_base64: Lista opzionale di documenti con 'data' e 'mime_type'
            system_prompt: Prompt di sistema per guidare il comportamento dell'LLM
            max_tokens: Limite di token per la risposta
            temperature: Controllo della casualità (0.0 = deterministico, 1.0 = creativo)

        Returns:
            La risposta testuale dell'LLM dopo aver analizzato tutti i contenuti

        Example:
            # Analisi di un'immagine
            result = await invoke_multimodal_llm(
                prompt="What do you see in this image?",
                images_base64=["iVBORw0KGgoAAAANS..."]  # base64 dell'immagine
            )

            # Analisi di un documento PDF (già convertito in immagini da un tool MCP)
            result = await invoke_multimodal_llm(
                prompt="Summarize this document",
                documents_base64=[{
                    "data": "JVBERi0xLj...",
                    "mime_type": "application/pdf"
                }]
            )
        """
        try:
            logger.info(f"Invoking multimodal LLM with prompt: {prompt[:100]}...")

            # VECCHIO CODICE (commentato):
            # Costruisce il messaggio multimodale
            # message_content = [{"type": "text", "text": prompt}]

            # NUOVO CODICE: Costruzione messaggio multimodale con supporto per riferimenti
            message_content = []

            # Aggiungi il testo del prompt
            message_content.append({"type": "text", "text": prompt})

            # --- RISOLUZIONE RIFERIMENTI BASE64 ---
            # Se images_base64 o documents_base64 contengono riferimenti (es. "<base64_ref_1>")
            # invece di dati base64 reali, risolvili usando lo storage

            def resolve_base64_reference(value: str) -> str:
                """
                Risolve un riferimento base64 se necessario.
                Se value è un riferimento (es. "<base64_ref_1>"), cerca nello storage.
                Altrimenti restituisce il valore originale.
                """
                if value.startswith("<") and value.endswith(">"):
                    # È un riferimento
                    ref_id = value[1:-1]  # Rimuovi < e >
                    if ref_id in base64_storage:
                        resolved = base64_storage[ref_id]
                        logger.info(f"Resolved reference {ref_id} from storage: {resolved['type']}")
                        return resolved['data']
                    else:
                        logger.warning(f"Reference {ref_id} not found in storage")
                        return value
                return value

            # Aggiungi le immagini se presenti
            if images_base64:
                logger.info(f"Processing {len(images_base64)} images")
                for idx, img_b64 in enumerate(images_base64):
                    # VECCHIO CODICE (commentato):
                    # Non risolveva i riferimenti base64

                    # NUOVO CODICE: Risolvi riferimenti se necessario
                    img_b64 = resolve_base64_reference(img_b64)

                    # Determina il media type dall'immagine
                    media_type = "image/jpeg"  # Default

                    # Rimuovi eventuali prefissi data:image se presenti ed estrai il media type
                    if img_b64.startswith("data:"):
                        # Formato: data:image/jpeg;base64,<base64_data>
                        parts = img_b64.split(",", 1)
                        if len(parts) == 2:
                            # Estrai il media type dal prefisso
                            header = parts[0]  # es: "data:image/jpeg;base64"
                            if "image/" in header:
                                # Estrai la parte tra "image/" e ";"
                                media_start = header.find("image/")
                                media_end = header.find(";", media_start)
                                if media_start != -1:
                                    if media_end != -1:
                                        media_type = header[media_start:media_end]
                                    else:
                                        media_type = header[media_start:]
                            img_b64 = parts[1]
                    else:
                        # Se non c'è prefisso, prova a dedurre dal magic number
                        try:
                            # Decodifica i primi byte per identificare il formato
                            decoded_start = base64.b64decode(img_b64[:32])
                            if decoded_start.startswith(b'\x89PNG'):
                                media_type = "image/png"
                            elif decoded_start.startswith(b'\xff\xd8\xff'):
                                media_type = "image/jpeg"
                            elif decoded_start.startswith(b'GIF87a') or decoded_start.startswith(b'GIF89a'):
                                media_type = "image/gif"
                            elif decoded_start.startswith(b'RIFF') and b'WEBP' in decoded_start:
                                media_type = "image/webp"
                            logger.debug(f"Detected image format from magic number: {media_type}")
                        except Exception as e:
                            logger.warning(f"Could not detect image format: {e}, using default {media_type}")

                    # Formato diverso per OpenAI/Google vs Anthropic
                    if uses_openai_format:
                        # OpenAI e Google (Gemini) usano il formato image_url con data URI
                        data_uri = f"data:{media_type};base64,{img_b64}"
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        })
                    else:
                        # Anthropic usa il formato source con base64
                        message_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64
                            }
                        })
                    logger.debug(f"Added image {idx + 1}/{len(images_base64)} with type {media_type}")

            # Aggiungi i documenti se presenti
            if documents_base64:
                logger.info(f"Processing {len(documents_base64)} documents")
                for idx, doc in enumerate(documents_base64):
                    doc_data = doc.get("data", "")
                    mime_type = doc.get("mime_type", "application/pdf")

                    # VECCHIO CODICE (commentato):
                    # Non risolveva i riferimenti base64

                    # NUOVO CODICE: Risolvi riferimenti se necessario
                    doc_data = resolve_base64_reference(doc_data)

                    # Rimuovi eventuali prefissi data: se presenti
                    if doc_data.startswith("data:"):
                        parts = doc_data.split(",", 1)
                        if len(parts) == 2:
                            doc_data = parts[1]

                    message_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": doc_data
                        }
                    })
                    logger.debug(f"Added document {idx + 1}/{len(documents_base64)} ({mime_type})")

            # Prepara i messaggi per l'LLM
            from langchain.schema import HumanMessage, SystemMessage

            messages = []

            # Aggiungi il system prompt se fornito
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            # Aggiungi il messaggio multimodale dell'utente
            messages.append(HumanMessage(content=message_content))

            # Invoca l'LLM con i parametri configurati
            # VECCHIO CODICE (commentato):
            # response = await llm_instance.ainvoke(messages)

            # NUOVO CODICE: Invocazione con gestione parametri
            # Crea una copia dell'istanza LLM con i parametri specificati
            # Questo permette di sovrascrivere temperature e max_tokens
            llm_with_params = llm_instance.bind(
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.debug(f"Calling LLM with {len(messages)} messages, max_tokens={max_tokens}, temperature={temperature}")
            response = await llm_with_params.ainvoke(messages)

            # Estrai il contenuto della risposta
            result_text = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"LLM response received: {len(result_text)} characters")
            return result_text

        except Exception as e:
            error_msg = f"Error invoking multimodal LLM: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"ERROR: {error_msg}"

    # Assegna metadati al tool per migliore identificazione
    invoke_multimodal_llm.name = "invoke_multimodal_llm"
    invoke_multimodal_llm.description = (
        "Invokes the internal multimodal LLM to analyze text, images, and documents. "
        "Use this tool when you need to perform vision tasks, document analysis, "
        "or any multimodal reasoning that the main LLM should handle."
    )

    return invoke_multimodal_llm


def create_provider_specific_multimodal_tool(provider: str, llm_instance: Any):
    """
    Crea un tool multimodale ottimizzato per un provider specifico.

    Args:
        provider: Nome del provider ('openai', 'anthropic', 'google', etc.)
        llm_instance: Istanza dell'LLM

    Returns:
        Tool configurato per il provider specifico
    """
    # Per ora usa la stessa implementazione, ma può essere estesa
    # per gestire differenze specifiche dei provider
    tool = create_multimodal_llm_tool(llm_instance)

    # Personalizzazioni future basate sul provider
    if provider == "anthropic":
        # Claude ha limiti diversi, potrebbe richiedere gestione speciale
        pass
    elif provider == "google":
        # Gemini ha un formato leggermente diverso
        pass

    return tool
