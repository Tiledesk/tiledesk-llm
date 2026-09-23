"""
LangChain @tool adapters over tools_core's _core coroutines.

No business logic and no tracing here — both live in tools_core.py, shared
with the MCP server adapter (services/mcp_server.py, P6). This file only
translates argument/return shapes for langchain.agents.create_agent, the
agent builder tilellm/controller/controller.py already uses for /api/ask.
"""
import json
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tilellm.modules.agentic_compliance_checker.models import SessionNotFound
from tilellm.modules.agentic_compliance_checker.services.tools_core import list_requirements_core


class ListRequirementsArgs(BaseModel):
    session_id: str = Field(
        description="Identificativo della sessione di verifica, fornito dall'utente."
    )
    kind: Literal["all", "tabular", "discretionary"] = "all"
    status: Literal["all", "pending", "done", "human_review"] = "all"
    operator: Optional[str] = Field(
        default=None,
        description="Etichetta o namespace dell'operatore economico. Omettere se la gara ha un solo operatore.",
    )


@tool(args_schema=ListRequirementsArgs)
async def compliance_list_requirements(
    session_id: str, kind: str = "all", status: str = "all", operator: Optional[str] = None,
) -> str:
    """Elenca i requisiti e i criteri di una gara pubblica aperta in una sessione di
    verifica, con il loro stato di valutazione. Da chiamare SEMPRE per prima cosa:
    restituisce gli id dei criteri necessari a tutti gli altri tool di compliance.
    Non recupera evidenze e non valuta nulla."""
    try:
        return await list_requirements_core(
            session_id=session_id, kind=kind, status=status, operator=operator,
        )
    except SessionNotFound as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# name -> LangChain tool object, mirrors tools_registry.TOOL_REGISTRY's shape
# for the entries controllers.py registers there.
AGENTIC_COMPLIANCE_TOOLS = {
    "compliance_list_requirements": compliance_list_requirements,
}
