import os
from typing import AsyncGenerator

from redis.asyncio import Redis, from_url

EXPIRATION_SECONDS: int = 48 * 60 * 60


async def get_redis_client() -> AsyncGenerator[Redis, None]:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await from_url(redis_url)
    try:
        yield client
    finally:
        await client.aclose()
