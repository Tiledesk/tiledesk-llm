import gc
import os
import logging
from typing import Any

from redis.asyncio import Redis, from_url
from taskiq import TaskiqEvents, TaskiqState, SimpleRetryMiddleware, TaskiqMiddleware
from taskiq.message import TaskiqMessage
from taskiq.result import TaskiqResult
from taskiq_redis import RedisStreamBroker, RedisAsyncResultBackend
from tilellm.shared.utility import get_service_config
from tilellm.tools._gpu_concurrency import cuda_empty_cache_safe

logger = logging.getLogger(__name__)

# Task types that perform local GPU inference (sparse encoder + reranker).
# After each of these tasks, the allocator cache is flushed to reclaim
# fragmented VRAM blocks that accumulate under sustained ingestion load.
_GPU_TASK_TYPES = {"pdf_ocr", "scraping", "raptor_build"}


class GpuCleanupMiddleware(TaskiqMiddleware):
    """
    Calls torch.cuda.empty_cache() after GPU-heavy tasks complete.

    PyTorch's caching allocator retains freed blocks for reuse, which causes
    VRAM fragmentation under sustained load with variable-length inputs
    (chunk sizes differ per document). The allocator does not defragment
    automatically; calling empty_cache() at task boundaries — not per-batch —
    is the correct trade-off: 100-200ms cost once per task vs. degrading
    OOM-recovery paths that slow every batch.
    """

    def post_execute(
        self,
        message: "TaskiqMessage",
        result: "TaskiqResult[Any]",
    ) -> None:
        task_type = (message.labels or {}).get("task_type", "")
        if task_type not in _GPU_TASK_TYPES:
            return

        free_before: int | None = None
        total: int | None = None
        try:
            import torch
            if torch.cuda.is_available():
                free_before, total = torch.cuda.mem_get_info()
        except Exception:
            pass

        cuda_empty_cache_safe()
        gc.collect()

        if free_before is not None and total is not None:
            try:
                import torch
                free_after, _ = torch.cuda.mem_get_info()
                freed_mb = (free_after - free_before) / 1024 ** 2
                used_before_mb = (total - free_before) / 1024 ** 2
                used_after_mb = (total - free_after) / 1024 ** 2
                total_mb = total / 1024 ** 2
                logger.info(
                    f"[GPU] cache flushed after {task_type}: "
                    f"freed {freed_mb:.0f} MB | "
                    f"VRAM {used_before_mb:.0f} → {used_after_mb:.0f} MB "
                    f"/ {total_mb:.0f} MB"
                )
            except Exception:
                logger.info(f"[GPU] cache flushed after {task_type}")
        else:
            logger.debug(f"[GPU] cache flush skipped after {task_type} (CUDA not available)")


# 1. Get config from environment variables
config = get_service_config()
redis_conf = config.get("redis", {})

redis_host = redis_conf.get("host", "localhost")
redis_port = redis_conf.get("port", 6379)
redis_db = redis_conf.get("db", 0)
redis_password = redis_conf.get("password", None)

# Construct Redis URL with more generous timeouts for heavy tasks
query_params = "?max_connections=100&socket_timeout=60.0&socket_connect_timeout=30.0&socket_keepalive=True&retry_on_timeout=True&health_check_interval=60"
redis_url = os.environ.get("REDIS_URL", f"redis://{redis_host}:{redis_port}/{redis_db}{query_params}")

# Overwrite with env if present (optional, but good practice)
redis_url = os.environ.get("REDIS_URL", redis_url)

# TTL per i risultati: 48h
RESULT_TTL_SECONDS = int(os.environ.get("TASKIQ_RESULT_TTL", str(48 * 60 * 60)))

# Idle timeout PEL: dopo quanti ms un messaggio non-ACK viene reclamato dal loop periodico.
# DEVE essere > durata massima di un singolo task (PDF Docling può durare 30+ min).
# Se troppo basso, xautoclaim ri-dispatcha task ancora in esecuzione → OOM → crash.
# Il crash-recovery rapido è gestito dallo startup-reclaim nel WORKER_STARTUP handler.
IDLE_TIMEOUT_MS = int(os.environ.get("TASKIQ_IDLE_TIMEOUT_MS", "3600000"))  # 60 min

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
    ),
    GpuCleanupMiddleware(),
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

        # Startup reclaim: rivendica immediatamente i messaggi in PEL lasciati da un
        # worker precedente crashato. Il loop periodico usa idle_timeout (60 min) per
        # evitare di reclamare task ancora in esecuzione, ma al restart è sicuro
        # reclamare tutto perché il processo precedente è già morto.
        # min_idle_time=1000ms (1s) evita race condition con dispatch appena arrivati.
        await _startup_reclaim(redis_health_client)

    except Exception as e:
        logger.error(f"❌ Redis connection FAILED: {e}")
        raise


async def _startup_reclaim(redis_client) -> None:
    """Re-queue PEL messages from dead workers as fresh stream entries.

    XAUTOCLAIM alone only transfers ownership inside the PEL; the broker's main
    loop uses XREADGROUP with '>' which reads only *undelivered* messages, never
    the PEL. Messages left in the PEL would wait idle_timeout (60 min) before
    the periodic autoclaim re-dispatches them.

    Strategy: claim each pending message, XADD it back to the stream as a brand-new
    entry (the normal broker format: {b"data": <payload>}), then XACK the old entry
    to clean up the PEL. The broker's '>' loop will pick up the new entries
    immediately on the next xreadgroup call.
    """
    try:
        stream_name = broker.queue_name          # default "taskiq"
        group_name = broker.consumer_group_name  # default "taskiq"
        consumer_name = broker.consumer_name     # UUID of this new worker

        total_requeued = 0
        start_id = "0-0"

        while True:
            result = await redis_client.xautoclaim(
                name=stream_name,
                groupname=group_name,
                consumername=consumer_name,
                min_idle_time=1000,  # 1 s — skip brand-new messages (race at startup)
                start_id=start_id,
                count=100,
            )
            # result = (next_start_id, [(msg_id, {b"data": payload}), ...], deleted_ids)
            claimed = result[1] if len(result) > 1 else []

            for msg_id, msg_data in claimed:
                payload = msg_data.get(b"data")
                if payload is None:
                    # unknown format — just ack to avoid permanent PEL pollution
                    await redis_client.xack(stream_name, group_name, msg_id)
                    continue
                # Re-publish as new undelivered message; broker '>' loop picks it up
                await redis_client.xadd(stream_name, {b"data": payload})
                # Clean up the old PEL entry
                await redis_client.xack(stream_name, group_name, msg_id)
                total_requeued += 1

            next_id = result[0] if result else b"0-0"
            if not claimed or next_id in (b"0-0", "0-0"):
                break
            start_id = next_id

        if total_requeued:
            logger.info(f"♻️ Startup reclaim: {total_requeued} messaggi re-pubblicati nello stream (pronti per il consumer)")
        else:
            logger.info("✅ Nessun messaggio PEL da reclamare all'avvio")

    except Exception as e:
        logger.warning(f"Startup reclaim non riuscito (non critico): {e}")

@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState) -> None:
    logger.info("🛑 TaskIQ Worker SHUTDOWN")
    if redis_health_client:
        await redis_health_client.aclose()



