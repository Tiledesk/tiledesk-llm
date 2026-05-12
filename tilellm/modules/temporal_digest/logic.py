"""
Temporal Digest — business logic entry points.
DI decorators inject LLM + repo; service layer does the actual work.
"""
from tilellm.shared.utility import inject_llm_chat_async, inject_repo_async
from tilellm.modules.temporal_digest.models.schemas import (
    DigestAgentRequest,
    DigestAgentResponse,
    DigestGenerationRequest,
    DigestGenerationResponse,
    DigestQueryRequest,
    DigestQueryResponse,
)
from tilellm.modules.temporal_digest.services.digest_service import DigestService

_service = DigestService()


@inject_llm_chat_async
@inject_repo_async
async def generate_digest(
    request: DigestGenerationRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestGenerationResponse:
    return await _service.generate(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)


@inject_llm_chat_async
@inject_repo_async
async def query_digest(
    request: DigestQueryRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestQueryResponse:
    return await _service.query(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)


@inject_llm_chat_async
@inject_repo_async
async def agent_query_digest(
    request: DigestAgentRequest,
    repo=None,
    llm=None,
    llm_embeddings=None,
    callback_handler=None,
    embedding_config_key=None,
    **kwargs,
) -> DigestAgentResponse:
    return await _service.agent_query(request, repo=repo, llm=llm, llm_embeddings=llm_embeddings)
