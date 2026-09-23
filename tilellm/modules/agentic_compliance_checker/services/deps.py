"""
The DI seam every tool uses to get a (repo, llm) pair for one operator's
ComplianceRequestV2. Separate from logic.py (which needs to import
services.langchain_tools for the registered tool names) specifically to avoid
a circular import: tools_core.py needs this too, and
tools_core.py -> logic.py -> langchain_tools.py -> tools_core.py would cycle.
"""
from tilellm.modules.compliance_checker.models_v2 import ComplianceRequestV2
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async


@inject_llm_chat_async
@inject_repo_async
async def _resolve_deps(request: ComplianceRequestV2, repo=None, llm=None, **kwargs):
    """repo/llm are never stored in the session (unpicklable, and already
    TimedCache-cached per-process on the same config fields) — they are
    re-derived from the stored config on every tool call; after the first
    call per worker process this is a cache hit, not a fresh construction.
    """
    return repo, llm
