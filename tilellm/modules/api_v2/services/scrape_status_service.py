from redis.asyncio import Redis

from tilellm.models.schemas import ScrapeStatusResponse
from tilellm.modules.api_v2.dependencies import EXPIRATION_SECONDS


class ScrapeStatusService:
    """Writes Redis indexing lifecycle status for a single document.

    Encapsulates all Redis I/O for the scrape status key ``{namespace}/{doc_id}``,
    keeping the controller free from Redis details (SRP).
    """

    def __init__(self, redis_client: Redis, expiration: int = EXPIRATION_SECONDS) -> None:
        self._redis = redis_client
        self._expiration = expiration

    async def set_started(self, namespace: str, doc_id: str) -> None:
        await self._write(namespace, doc_id, message="Indexing started", code=2)

    async def set_finished(self, namespace: str, doc_id: str) -> None:
        await self._write(namespace, doc_id, message="Indexing finish", code=3)

    async def set_error(self, namespace: str, doc_id: str) -> None:
        await self._write(namespace, doc_id, message="Error", code=4)

    async def _write(self, namespace: str, doc_id: str, message: str, code: int) -> None:
        status = ScrapeStatusResponse(status_message=message, status_code=code)
        await self._redis.set(
            f"{namespace}/{doc_id}",
            status.model_dump_json(),
            ex=self._expiration,
        )
