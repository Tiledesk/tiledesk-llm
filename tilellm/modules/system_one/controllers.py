"""
System One API: typed, calibrated decisions (noul / choice / score) from any server
speaking the Jev wire protocol — TypeSafe Jev (hosted), laya-serve and clm-serve
(self-hosted, see the "laya" and "clm" profiles of docker-compose-redis-tei-qdrant.yml).
"""
from typing import List

from fastapi import APIRouter, HTTPException

from tilellm.modules.system_one import service
from tilellm.modules.system_one.models import SystemOneRequest, SystemOneResponse
from tilellm.modules.system_one.providers import list_providers

router = APIRouter(prefix="/api/v1", tags=["System One"])


@router.post("/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def systemone(request: SystemOneRequest) -> SystemOneResponse:
    """Answer typed questions about a state with the provider named in `model`."""
    try:
        return await service.decide(request)
    except service.SystemOneError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail, headers=e.headers)


@router.get("/systemone/providers")
async def providers() -> List[dict]:
    """Registered providers with their defaults."""
    return [spec.public() for spec in list_providers()]
