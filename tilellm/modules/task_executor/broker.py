import os
import logging

from redis.asyncio import Redis, from_url
from taskiq import TaskiqEvents, TaskiqState, SimpleRetryMiddleware
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend
from tilellm.shared.utility import get_service_config

logger = logging.getLogger(__name__)

# 1. Get config from environment variables
config = get_service_config()
redis_conf = config.get("redis", {})

redis_host = redis_conf.get("host", "localhost")
redis_port = redis_conf.get("port", 6379)
redis_db = redis_conf.get("db", 0)
redis_password = redis_conf.get("password", None)

# Construct Redis URL
query_params = "?max_connections=100&socket_timeout=5.0&socket_connect_timeout=5.0&socket_keepalive=True&retry_on_timeout=True&health_check_interval=30"
redis_url = os.environ.get("REDIS_URL", f"redis://{redis_host}:{redis_port}/{redis_db}{query_params}")

# Overwrite with env if present (optional, but good practice)
redis_url = os.environ.get("REDIS_URL", redis_url)

# TTL per i risultati: 48h
RESULT_TTL_SECONDS = int(os.environ.get("TASKIQ_RESULT_TTL", str(48 * 60 * 60)))

# Idle timeout PEL: dopo quanti ms un messaggio non-ACK viene reclamato (default 5 min)
IDLE_TIMEOUT_MS = int(os.environ.get("TASKIQ_IDLE_TIMEOUT_MS", "300000"))

# --- 2. Resilient Backend Configuration ---
result_backend = RedisAsyncResultBackend(
    redis_url=redis_url,
    result_ex_time=RESULT_TTL_SECONDS,
    password=redis_password
)

# --- 3. Resilient Broker Configuration (STREAM) ---
broker = RedisStreamBroker(
    url=redis_url,
    password=redis_password,
    xread_count=1,
    idle_timeout=IDLE_TIMEOUT_MS,      # Recovery automatico PEL: messaggi stuck vengono reclamati
    unacknowledged_batch_size=50,      # Quanti messaggi PEL processare all'avvio
)

broker.add_middlewares(
    SimpleRetryMiddleware(
        default_retry_count=3,
        default_retry_label=True,      # CRITICO: abilita retry automatico per TUTTI i task
        no_result_on_retry=True,
    )
)

broker.with_result_backend(result_backend)

redis_health_client: Redis = None

# --- 4. Logging e Monitoraggio (Observability) ---
@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState) -> None:
    global redis_health_client
    logger.info("🚀 TaskIQ Worker (STREAM) STARTED")
    try:
        redis_health_client = await from_url(redis_url, password=redis_password)
        await redis_health_client.ping()
        logger.info("✅ Redis connection OK")
    except Exception as e:
        logger.error(f"❌ Redis connection FAILED: {e}")
        raise

@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState) -> None:
    logger.info("🛑 TaskIQ Worker SHUTDOWN")
    if redis_health_client:
        await redis_health_client.aclose()



