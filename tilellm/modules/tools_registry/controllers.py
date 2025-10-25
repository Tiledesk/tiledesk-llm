
from fastapi import APIRouter

from tilellm.modules.tools_registry.services.tool_registry import get_available_tools_list

# 1. Crea il router per questo modulo
router = APIRouter(
    prefix="/api",
    tags=["Tools"] # Tag per la documentazione OpenAPI (Swagger)
)

@router.get("/tools")
async def list_available_tools():
    """
    Endpoint pubblico che mostra all'utente i tool
    disponibili con nome e descrizione.
    """
    return get_available_tools_list()