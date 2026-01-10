import os
from taskiq import TaskiqEvents, TaskiqState
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from tilellm.shared.utility import get_service_config

# Get config from service_conf.yaml
config = get_service_config()
redis_conf = config.get("redis", {})

redis_host = redis_conf.get("host", "localhost")
redis_port = redis_conf.get("port", 6379)
redis_db = redis_conf.get("db", 0)

# Construct Redis URL
redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

# Overwrite with env if present (optional, but good practice)
redis_url = os.environ.get("REDIS_URL", redis_url)

# Configure Redis Result Backend
result_backend = RedisAsyncResultBackend(
    redis_url=redis_url,
)

# Configure Redis Broker with Result Backend
broker = ListQueueBroker(url=redis_url)

broker.with_result_backend(result_backend)

@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState) -> None:
    print("Taskiq worker started.")

@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState) -> None:
    print("Taskiq worker shutting down.")
