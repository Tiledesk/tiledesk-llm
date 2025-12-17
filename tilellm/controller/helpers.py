import ast
import json
import logging
import base64
import httpx
from typing import Union, List, Any, Dict, Tuple

logger = logging.getLogger(__name__)


async def _preprocess_documents_for_mcp(
    message_content: List[Dict[str, Any]],
    tools: List[Any],
    mcp_client: Any
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Pre-processa i documenti nel message_content invocando i tool MCP di conversione.

    Questa funzione:
    1. Cerca documenti con URL o base64 nel message_content
    2. Identifica il tool MCP appropriato per la conversione (es. convert_pdf_to_images)
    3. Invoca il tool per convertire il documento
    4. Sostituisce il documento con le immagini/risultati della conversione

    Args:
        message_content: Lista di contenuti del messaggio (testo, immagini, documenti)
        tools: Lista dei tool MCP disponibili
        mcp_client: Client MCP per invocare i tool

    Returns:
        Tuple[List[Dict], bool]: (message_content trasformato, ha_invocato_tool)
    """
    transformed_content = []
    has_converted_documents = False

    # Cerca tool di conversione disponibili
    conversion_tools = {}
    for tool in tools:
        tool_name = tool.name.lower()
        # Cerca tool che potrebbero convertire documenti (pdf, docx, etc.)
        if 'convert' in tool_name or 'pdf' in tool_name or 'document' in tool_name:
            conversion_tools[tool_name] = tool
            logger.info(f"Found conversion tool: {tool_name}")

    for item in message_content:
        item_type = item.get("type", "text")

        # Se è un documento, prova a convertirlo
        if item_type == "document":
            document_source = item.get("source", {})
            mime_type = item.get("mime_type", "application/pdf")

            url = None
            base64_data = None

            if isinstance(document_source, str):
                # È una stringa: può essere URL o base64
                if document_source.startswith(("http://", "https://")):
                    url = document_source
                    logger.info(f"Document URL detected: {url}")
                else:
                    # Assumiamo sia base64
                    base64_data = document_source
                    logger.info(f"Document base64 string detected (length: {len(base64_data)})")
            elif isinstance(document_source, dict):
                # È un dict: formato {type: "base64", data: "..."}
                if document_source.get("type") == "base64":
                    base64_data = document_source.get("data", "")
                    mime_type = document_source.get("media_type", mime_type)
                    logger.info(f"Document base64 dict detected (length: {len(base64_data)})")
                else:
                    logger.warning(f"Unknown dict format for document source: {document_source}")
                    transformed_content.append(item)
                    continue
            else:
                logger.warning(f"Unknown document source type: {type(document_source)}")
                transformed_content.append(item)
                continue

            # Cerca il tool appropriato per questo tipo di documento
            conversion_tool = None
            if "pdf" in mime_type.lower():
                # Cerca tool per PDF
                for tool_name, tool in conversion_tools.items():
                    if 'pdf' in tool_name:
                        conversion_tool = tool
                        break

            if conversion_tool is None:
                logger.warning(f"No conversion tool found for {mime_type}, keeping document as-is")
                transformed_content.append(item)
                continue

            # Invoca il tool MCP per convertire il documento
            try:
                logger.info(f"Invoking MCP tool: {conversion_tool.name}")

                # Prepara gli argomenti per il tool
                tool_args = {}

                # Se è un URL, scaricalo prima
                if url:
                    logger.info(f"Downloading document from URL: {url}")
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, timeout=30.0)
                        response.raise_for_status()
                        file_bytes = response.content
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        logger.info(f"Document downloaded, size: {len(file_bytes)} bytes, base64 length: {len(base64_data)}")

                # Prova a estrarre lo schema in modo sicuro
                schema_dict = None
                try:
                    if hasattr(conversion_tool, 'inputSchema'):
                        schema_obj = conversion_tool.inputSchema
                        # Potrebbe essere un dict, un oggetto Pydantic, o altro
                        if isinstance(schema_obj, dict):
                            schema_dict = schema_obj
                        elif hasattr(schema_obj, 'model_json_schema'):
                            # Pydantic v2
                            schema_dict = schema_obj.model_json_schema()
                        elif hasattr(schema_obj, 'schema'):
                            # Pydantic v1
                            schema_dict = schema_obj.schema()
                    elif hasattr(conversion_tool, 'input_schema'):
                        schema_obj = conversion_tool.input_schema
                        if isinstance(schema_obj, dict):
                            schema_dict = schema_obj
                        elif hasattr(schema_obj, 'model_json_schema'):
                            schema_dict = schema_obj.model_json_schema()
                        elif hasattr(schema_obj, 'schema'):
                            schema_dict = schema_obj.schema()
                except Exception as schema_error:
                    logger.warning(f"Could not extract schema from tool: {schema_error}")

                # Se abbiamo lo schema, trova il parametro giusto
                if schema_dict and isinstance(schema_dict, dict) and 'properties' in schema_dict:
                    logger.info(f"Tool schema properties: {list(schema_dict['properties'].keys())}")
                    for param_name in schema_dict['properties'].keys():
                        if any(keyword in param_name.lower() for keyword in ['base64', 'file', 'pdf', 'data', 'content', 'document']):
                            tool_args[param_name] = base64_data
                            logger.info(f"Using parameter '{param_name}' for base64 data")
                            break

                # Fallback: prova nomi comuni se non abbiamo trovato nulla
                if not tool_args:
                    logger.warning("Could not determine parameter name from schema, trying common names")
                    # Prova diversi nomi comuni in ordine di probabilità
                    tool_args = {"pdf_base64": base64_data}

                logger.info(f"Tool arguments keys: {list(tool_args.keys())}")

                # Invoca il tool
                result = await mcp_client.call_tool(conversion_tool.name, tool_args)

                logger.info(f"Tool result type: {type(result)}")
                logger.info(f"Tool result content: {result}")

                # Processa il risultato
                # Il risultato dovrebbe contenere immagini in base64
                if hasattr(result, 'content'):
                    result_content = result.content
                elif isinstance(result, dict):
                    result_content = result.get('content', [])
                else:
                    result_content = []

                # Estrai le immagini dal risultato
                images_added = False
                for result_item in result_content:
                    if isinstance(result_item, dict):
                        # Cerca immagini nel risultato
                        if result_item.get('type') == 'image' or 'image' in str(result_item).lower():
                            # Aggiungi l'immagine al contenuto trasformato
                            image_data = result_item.get('data', result_item.get('source', ''))
                            if image_data:
                                transformed_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data
                                    }
                                })
                                images_added = True
                                logger.info("Image extracted from tool result and added to content")
                    elif hasattr(result_item, 'type') and result_item.type == 'image':
                        # Oggetto Pydantic con immagine
                        transformed_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": getattr(result_item, 'mimeType', 'image/png'),
                                "data": getattr(result_item, 'data', '')
                            }
                        })
                        images_added = True
                        logger.info("Image from Pydantic object added to content")

                if images_added:
                    has_converted_documents = True
                    logger.info(f"Successfully converted document using tool {conversion_tool.name}")
                else:
                    logger.warning("Tool executed but no images found in result, keeping original document")
                    transformed_content.append(item)

            except Exception as e:
                logger.error(f"Error invoking conversion tool: {e}", exc_info=True)
                # In caso di errore, mantieni il documento originale
                transformed_content.append(item)
        else:
            # Non è un documento, mantieni come-è
            transformed_content.append(item)

    return transformed_content, has_converted_documents


def _get_question_list(question_input: Union[str, List[Any]]) -> List[Dict | Any]:
    """
    Normalizza il campo flessibile 'question' in una lista per il processamento.
    L'input può essere:
    1. Una lista (ideale, di modelli Pydantic o dict)
    2. Una stringa JSON che rappresenta una lista
    3. Una stringa letterale Python che rappresenta una lista
    4. Una stringa di testo semplice
    """
    if isinstance(question_input, list):
        # Caso 1: Già una lista (es. List[MultimodalContent])
        return question_input

    if not isinstance(question_input, str):
        # Fallback per tipi inattesi (int, float, etc.)
        return [{"type": "text", "text": str(question_input)}]

    # È una stringa, proviamo a parsarla
    try:
        # Caso 2: Stringa JSON
        parsed = json.loads(question_input)
        if isinstance(parsed, list):
            return parsed  # Era una lista JSON di dict
        else:
            # Era una stringa, numero, o oggetto JSON. Trattalo come testo.
            return [{"type": "text", "text": str(parsed)}]
    except (json.JSONDecodeError, TypeError):
        # Non era JSON, prova ast.literal_eval
        try:
            # Caso 3: Stringa letterale Python (es. "[{'type': 'text', ...}]")
            parsed = ast.literal_eval(question_input)
            if isinstance(parsed, list):
                return parsed
            else:
                return [{"type": "text", "text": str(parsed)}]
        except (ValueError, SyntaxError, TypeError):
            # Caso 4: Fallimento di tutti i parsing, è testo semplice
            return [{"type": "text", "text": question_input}]