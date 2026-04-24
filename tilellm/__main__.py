import os
import json
import logging.config
import warnings

# Suppress Pydantic V2 serialization warnings emitted by the OpenAI SDK v2 when
# calling model_dump() on ParsedResponse[T] (TypeVar-based generics).
# These warnings are cosmetic: the structured output is correctly extracted by
# langchain_openai. Root cause: openai SDK v2 + Pydantic V2 TypeVar generics
# incompatibility (tracked upstream in openai-python).
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module=r"pydantic\.main",
)

from contextlib import asynccontextmanager
from typing import Union, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi.exception_handlers import http_exception_handler

# Load .env early so all os.environ.get() calls pick up values from .env
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_cprofile.profiler import CProfileMiddleware
from fastapi.responses import JSONResponse

import asyncio
import time
from redis.asyncio import Redis, from_url
import aiohttp


from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.utility import decode_jwt
from tilellm.models import ItemSingle, Engine, QuestionToLLM, QuestionAnswer
from tilellm.models.schemas import (
    RepositoryItem,
    RepositoryNamespace,
    ScrapeStatusReq,
    ScrapeStatusResponse,
    IndexingResult,
    RetrievalResult,
    RepositoryNamespaceResult,
    RepositoryDescNamespaceResult,
    RepositoryItems,
    SimpleAnswer,
    RepositoryEngine,
    RetrievalChunksResult,
)
from tilellm.models.schemas.general_schemas import AsyncTaskResponse
from tilellm.modules.knowledge_graph.models.schemas import TaskPollResponse

try:
    from tilellm.modules.task_executor.tasks import task_scrape_item_single
    from tilellm.modules.task_executor.broker import broker

    TASKIQ_AVAILABLE = True
except ImportError:
    TASKIQ_AVAILABLE = False
    task_scrape_item_single = None
    broker = None

ENABLE_TASKIQ = os.environ.get("ENABLE_TASKIQ", "false").lower() == "true"
ENABLE_TASKIQ = ENABLE_TASKIQ and TASKIQ_AVAILABLE

from tilellm.shared.llm_config import serialize_with_secrets

import tilellm.analytics as analytics

from tilellm.store.redis_repository import redis_xgroup_create
from tilellm.controller.controller import (
    ask_with_memory,
    ask_hybrid_with_memory,
    ask_for_chunks,
    add_item,
    add_item_hybrid,
    delete_namespace,
    delete_id_from_namespace,
    delete_chunk_id_from_namespace,
    get_ids_namespace,
    get_listitems_namespace,
    get_desc_namespace,
    get_list_namespace,
    get_sources_namespace,
    ask_to_llm,
    ask_reason_llm,
    ask_mcp_agent_llm,
    ask_mcp_agent_llm_simple,
)

import logging

import sys


def setup_logging():
    with open("log_conf.json", "r") as f:
        config = json.load(f)

    # Leggi ENV
    log_level_stdout = os.getenv("LOG_LEVEL_STDOUT", "INFO").upper()
    log_level_file = os.getenv("LOG_LEVEL_FILE", "INFO").upper()
    log_level_sys = os.getenv("LOG_LEVEL_SYS", "INFO").upper()

    # 1. Aggiorna Handler
    config["handlers"]["stdout"]["level"] = log_level_stdout

    log_file_path = os.getenv("LOG_FILE_PATH", "app.log")
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    config["handlers"]["file_handler"]["filename"] = log_file_path
    config["handlers"]["file_handler"]["level"] = log_level_file

    # 2. Aggiorna i Logger Specifici
    # Imposta tilellm allo stesso livello dello stdout (o crea una ENV apposita)
    if "tilellm" in config["loggers"]:
        config["loggers"]["tilellm"]["level"] = log_level_stdout

    # Aggiorna logger di sistema
    for logger_name in [
        "gunicorn.error",
        "uvicorn.error",
        "gunicorn.access",
        "uvicorn.access",
    ]:
        if logger_name in config["loggers"]:
            config["loggers"][logger_name]["level"] = log_level_sys

    # 3. Aggiorna il Root Logger (Importante come fallback)
    config["root"]["level"] = log_level_stdout

    logging.config.dictConfig(config)


setup_logging()

# 1. Trova il percorso del file corrente (__main__.py)
current_file_path = Path(__file__).resolve()

# 2. Risali alla cartella root del progetto.
#    In questo caso, siccome il file è in 'tilellm/', dobbiamo salire di un livello ('.parent')
#    per arrivare alla cartella che contiene sia 'tilellm' che 'modules'.
project_root = current_file_path

#    Se __main__.py fosse nella root, basterebbe .parent

# 3. Aggiungi la root del progetto al percorso di ricerca di Python
sys.path.append(str(project_root))


expiration_in_seconds = 48 * 60 * 60

logger = logging.getLogger(__name__)

redis_url = os.environ.get("REDIS_URL")
tilellm_role = os.environ.get("TILELLM_ROLE")

security = HTTPBearer()


async def get_redis_client():
    redis_client = None
    try:
        redis_client = await from_url(redis_url)
        yield redis_client
    finally:
        if redis_client:
            await redis_client.aclose()


async def reader(channel: Redis):
    """
    The reader consume the redis queue
    :param channel:
    :return:
    """

    from tilellm.shared import const

    logger.debug(f"My role is {tilellm_role}")
    webhook = ""
    token = ""
    item = {}
    if tilellm_role == "train":
        while True:
            try:
                messages = await channel.xreadgroup(
                    groupname=const.STREAM_CONSUMER_GROUP,
                    consumername=const.STREAM_CONSUMER_NAME,
                    streams={const.STREAM_NAME: ">"},
                    count=1,
                    block=0,  # Set block to 0 for non-blocking
                )

                for stream, message_data in messages:
                    for message_id, message_values in message_data:
                        logger.debug(f"My role is {tilellm_role} consume message")
                        # message_id, message_values= message
                        import ast

                        byte_str = message_values[b"single"]
                        dict_str = byte_str.decode("UTF-8")
                        logger.info(dict_str)
                        item = ast.literal_eval(dict_str)
                        item_single = ItemSingle(**item)
                        scrape_status_response = ScrapeStatusResponse(
                            status_message="Indexing started", status_code=2
                        )
                        add_to_queue = await channel.set(
                            f"{item.get('namespace')}/{item.get('id')}",
                            scrape_status_response.model_dump_json(),
                            ex=expiration_in_seconds,
                        )

                        logger.debug(f"Start {add_to_queue}")

                        raw_webhook = item.get("webhook", "")
                        if "?" in raw_webhook:
                            webhook, raw_token = raw_webhook.split("?")

                            if raw_token.startswith("token="):
                                _, token = raw_token.split("=")
                        else:
                            webhook = raw_webhook

                        logger.info(f"webhook: {webhook}, token: {token}")

                        if webhook:
                            res = IndexingResult(id=item.get("id"), status=200)
                            try:
                                async with aiohttp.ClientSession() as session:
                                    res = await session.post(
                                        webhook,
                                        json=res.model_dump(exclude_none=True),
                                        headers={
                                            "Content-Type": "application/json",
                                            "X-Auth-Token": token,
                                        },
                                    )
                                    logger.info(f"200 {await res.json()}")
                            except Exception as ewh:
                                logger.error(ewh)
                                pass

                        _idx_t0 = time.monotonic()
                        _idx_error: str | None = None
                        try:
                            pc_result = await add_item(item_single)
                        except Exception as _idx_exc:
                            _idx_error = str(_idx_exc)
                            pc_result = IndexingResult(
                                id=item.get("id"), status=500, error=_idx_error
                            )
                        _idx_duration_ms = int((time.monotonic() - _idx_t0) * 1000)

                        # Analytics: emit kb.content_indexed (fire-and-forget)
                        from tilellm.analytics import events as an_events

                        _emb_model = an_events.get_embedding_model_name(
                            item_single.embedding
                        )
                        _engine_val = an_events.get_engine_value(item_single.engine)
                        _et, _pl = an_events.content_indexed(
                            kb_id=item_single.namespace,
                            kb_name=item_single.namespace,
                            embedding_model=_emb_model,
                            engine=_engine_val,
                            duration_ms=_idx_duration_ms,
                            success=_idx_error is None,
                            source_url=item_single.source,
                            source_type=item_single.type,
                            chunks_indexed=getattr(pc_result, "chunks", 0) or 0,
                            error_message=_idx_error,
                            request_id=item_single.request_id,
                        )
                        analytics.publish_nowait(_et, item_single.id_project, _pl)

                        # import datetime
                        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")

                        # pc_result["date"]= current_time
                        # pc_result["status"] = current_time

                        # A POST request to the API

                        scrape_status_response = ScrapeStatusResponse(
                            status_message="Indexing finish", status_code=3
                        )
                        add_to_queue = await channel.set(
                            f"{item.get('namespace')}/{item.get('id')}",
                            scrape_status_response.model_dump_json(),
                            ex=expiration_in_seconds,
                        )

                        logger.debug(f"End {add_to_queue}")
                        if webhook:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    res = await session.post(
                                        webhook,
                                        json=pc_result.model_dump(exclude_none=True),
                                        headers={
                                            "Content-Type": "application/json",
                                            "X-Auth-Token": token,
                                        },
                                    )
                                    logger.info(f"300 {await res.json()}")
                            except Exception as ewh:
                                logger.error(ewh)
                                pass

                        await channel.xack(
                            const.STREAM_NAME, const.STREAM_CONSUMER_GROUP, message_id
                        )
                        logger.info(f"xack to message_id: {message_id}")

            except Exception as e:
                scrape_status_response = ScrapeStatusResponse(
                    status_message="Error", status_code=4
                )
                add_to_queue = await channel.set(
                    f"{item.get('namespace')}/{item.get('id')}",
                    scrape_status_response.model_dump_json(),
                    ex=expiration_in_seconds,
                )

                logger.error(f"Error {add_to_queue}")
                import traceback

                if webhook:
                    res = IndexingResult(id=item.get("id"), status=400, error=repr(e))
                    async with aiohttp.ClientSession() as session:
                        response = await session.post(
                            webhook,
                            json=res.model_dump(exclude_none=True),
                            headers={
                                "Content-Type": "application/json",
                                "X-Auth-Token": token,
                            },
                        )
                        logger.error(response)
                        logger.error(f"{await response.json()}")
                    logger.error(f"Error {e}, webhook: {webhook}")
                traceback.print_exc()
                logger.error(e)
                pass
    else:
        logger.debug(f"My role is {tilellm_role}")


@asynccontextmanager
async def redis_consumer(app: FastAPI):
    redis_client = None
    try:
        # ✅ 1. Aggiungi AWAIT (from_url è asincrono)
        redis_client = await from_url(redis_url, decode_responses=True)

        # ✅ 2. Crea il consumer group se necessario
        try:
            await redis_xgroup_create(redis_client)
        except Exception as e:
            logger.warning(f"Consumer group creation skipped or failed: {e}")

        # ✅ 3. Avvia il task di lettura
        reader_task = asyncio.create_task(reader(redis_client))

        # ✅ 4. Avvia il broker Taskiq per l'invio di task (kicker side)
        if ENABLE_TASKIQ and broker is not None:
            try:
                await broker.startup()
                logger.info("TaskIQ broker started successfully")
            except Exception as e:
                logger.error(f"Failed to start TaskIQ broker: {e}")

        logger.info("App startup complete - Redis & FalkorDB ready")
        await analytics.init()
        from tilellm.analytics.config import config as _an_cfg
        logger.info(
            "analytics startup: enabled=%s ingest_url=%s",
            _an_cfg.is_enabled,
            _an_cfg.ingest_url or "<ANALYTICS_INGEST_URL not set>",
        )

        yield

    finally:
        # ✅ 5. Cleanup ordinato allo shutdown
        logger.info("Shutting down Redis consumer...")
        await analytics.shutdown()

        # Chiudi il broker Taskiq
        if ENABLE_TASKIQ and broker is not None:
            try:
                await broker.shutdown()
                logger.info("TaskIQ broker shut down")
            except Exception as e:
                logger.debug(f"TaskIQ broker shutdown error (ignored): {e}")

        # Cancella il task di lettura se attivo
        if "reader_task" in locals() and not reader_task.done():
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

        # Chiudi FalkorDB
        try:
            from tilellm.modules.knowledge_graph_falkor.logic import repository

            if repository:
                await repository.close()
                logger.info("FalkorDB connection closed")
        except Exception as e:
            logger.debug(f"FalkorDB cleanup skipped: {e}")

        # Chiudi Redis
        if redis_client:
            await redis_client.aclose()
            logger.info("Redis connection closed")

        # Pulisci cache
        await TimedCache.async_clear_cache("vector_store_wrapper")


app = FastAPI(lifespan=redis_consumer)


# Leggi la variabile d'ambiente per la profilazione
ENABLE_PROFILER = os.getenv("ENABLE_PROFILER", "False").lower() == "true"

if ENABLE_PROFILER:
    app.add_middleware(CProfileMiddleware, enable=True, print_each_request=True)


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # 1. Se l'errore è una HTTPException (quella che lanci tu con i tuoi messaggi)
    # la lasciamo passare senza loggare il traceback chilometrico (se non vuoi)
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)

    # 2. Se è un errore NON GESTITO (il crash che hai visto nel file log)
    # Lo scriviamo nel file log con TUTTO il traceback
    logger.error(
        f"Uncaught Exception: {request.method} {request.url.path} - {type(exc).__name__}: {str(exc)}",
        exc_info=True,
    )

    # 3. Restituiamo una risposta di fallback per evitare che il client riceva il vuoto
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "type": type(exc).__name__},
    )


@app.post(
    "/api/scrape/enqueue",
    response_model=Union[AsyncTaskResponse, IndexingResult],
    tags=["Scrape"],
)
async def enqueue_scrape_item_main(
    item: ItemSingle, redis_client: Redis = Depends(get_redis_client)
):
    """
    Enqueue item for async processing via Taskiq or process synchronously if Taskiq is disabled.
    When Taskiq is enabled, returns AsyncTaskResponse with task_id.
    When Taskiq is disabled, processes the item synchronously like /api/scrape/single.
    :param item:
    :param redis_client:
    :return: AsyncTaskResponse or IndexingResult
    """
    logger.debug(item)

    if ENABLE_TASKIQ:
        scrape_status_response = ScrapeStatusResponse(
            status_message="Document added to queue", status_code=0
        )
        await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        payload = serialize_with_secrets(item.model_dump(mode="python"))
        task = await task_scrape_item_single.kiq(payload)

        logger.info(f"Task enqueued with id: {task.task_id}")
        return AsyncTaskResponse(task_id=task.task_id)

    webhook = ""
    token = ""
    try:
        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing started", status_code=2
        )
        await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        logger.debug(f"Start processing item {item.id}")

        raw_webhook = item.webhook
        if raw_webhook and "?" in raw_webhook:
            webhook, raw_token = raw_webhook.split("?")
            if raw_token.startswith("token="):
                _, token = raw_token.split("=")
        else:
            webhook = raw_webhook or ""

        logger.info(f"webhook: {webhook}, token: {token}")

        if item.hybrid:
            pc_result = await add_item_hybrid(item)
        else:
            pc_result = await add_item(item)

        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing finish", status_code=3
        )
        await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        return JSONResponse(content=pc_result.model_dump(exclude_none=True))

    except Exception as e:
        scrape_status_response = ScrapeStatusResponse(
            status_message="Error", status_code=4
        )
        await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        logger.error(f"Error processing item {item.id}: {e}")
        import traceback

        traceback.print_exc()
        logger.error(e)
        return JSONResponse(status_code=400, content=e.args[0])


@app.get(
    "/api/enqueue/status/{task_id}", response_model=TaskPollResponse, tags=["Scrape"]
)
async def get_enqueue_task_status(task_id: str):
    """
    Check the status of an asynchronous scrape task.
    """
    if not (ENABLE_TASKIQ and TASKIQ_AVAILABLE):
        raise HTTPException(status_code=501, detail="Async tasks not enabled")

    try:
        result_backend = broker.result_backend
        is_ready = await result_backend.is_result_ready(task_id)

        if not is_ready:
            return TaskPollResponse(task_id=task_id, status="in_progress")

        result = await result_backend.get_result(task_id)

        if result.is_err:
            return TaskPollResponse(
                task_id=task_id,
                status="failed",
                error=str(result.error)
                if hasattr(result, "error")
                else str(result.return_value),
            )

        return TaskPollResponse(
            task_id=task_id, status="success", result=result.return_value
        )

    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve task status: {str(e)}"
        )


@app.post("/api/scrape/single", response_model=IndexingResult, tags=["Scrape"])
async def create_scrape_item_single(
    item: ItemSingle, redis_client: Redis = Depends(get_redis_client)
):
    """
    Add item to namespace
    :param item:
    :param redis_client:
    :return: PineconeIndexingResult
    """
    webhook = ""
    token = ""
    try:
        logger.debug(item)
        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing started", status_code=2
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        logger.debug(f"Start {add_to_queue}")

        raw_webhook = item.webhook
        if "?" in raw_webhook:
            webhook, raw_token = raw_webhook.split("?")

            if raw_token.startswith("token="):
                _, token = raw_token.split("=")
        else:
            webhook = raw_webhook

        logger.info(f"webhook: {webhook}, token: {token}")

        _idx_t0 = time.monotonic()
        _idx_error: str | None = None
        pc_result = None
        try:
            if item.hybrid:
                pc_result = await add_item_hybrid(item)
            else:
                pc_result = await add_item(item)
        except Exception as _idx_exc:
            _idx_error = str(_idx_exc)
            raise
        finally:
            _idx_duration_ms = int((time.monotonic() - _idx_t0) * 1000)
            _et, _pl = analytics.events.content_indexed(
                kb_id=item.namespace,
                kb_name=item.namespace,
                embedding_model=analytics.events.get_embedding_model_name(
                    item.embedding
                ),
                engine=analytics.events.get_engine_value(item.engine),
                duration_ms=_idx_duration_ms,
                success=_idx_error is None,
                source_url=item.source,
                source_type=item.type,
                chunks_indexed=(pc_result.chunks or 0)
                if _idx_error is None and pc_result is not None
                else 0,
                error_message=_idx_error,
                request_id=item.request_id,
            )
            analytics.publish_nowait(_et, item.id_project, _pl)

        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing finish", status_code=3
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        # logger.debug(f"End {add_to_queue}")
        # if webhook:
        #    try:
        #        async with aiohttp.ClientSession() as session:
        #            res = await session.post(webhook,
        #                                     json=pc_result.model_dump(exclude_none=True),
        #                                     headers={"Content-Type": "application/json",
        #                                              "X-Auth-Token": token})
        #            logger.info(f"300 {await res.json()}")
        #    except Exception as ewh:
        #        logger.error(ewh)
        #        pass

        return JSONResponse(content=pc_result.model_dump(exclude_none=True))

    except Exception as e:
        scrape_status_response = ScrapeStatusResponse(
            status_message="Error", status_code=4
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(by_alias=True),
            ex=expiration_in_seconds,
        )

        logger.error(f"Error {add_to_queue}")
        import traceback

        # if webhook:
        #    res = PineconeIndexingResult(id=item.id, status=400, error=repr(e))
        #    async with aiohttp.ClientSession() as session:
        #        response = await session.post(webhook, json=res.model_dump(exclude_none=True),
        #                                      headers={"Content-Type": "application/json", "X-Auth-Token": token})
        #        logger.error(response)
        #        logger.error(f"{await response.json()}")
        #    logger.error(f"Error {e}, webhook: {webhook}")
        traceback.print_exc()
        logger.error(e)
        # raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(status_code=400, content=e.args[0])


@app.post("/api/scrape/hybrid", response_model=IndexingResult, tags=["Scrape"])
async def create_scrape_item_hybrid(
    item: ItemSingle, redis_client: Redis = Depends(get_redis_client)
):
    """
    Add item to namespace
    :param item:
    :param redis_client:
    :return: PineconeIndexingResult
    """
    webhook = ""
    token = ""

    try:
        logger.debug(item)
        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing started", status_code=2
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        logger.debug(f"Start {add_to_queue}")

        raw_webhook = item.webhook
        if "?" in raw_webhook:
            webhook, raw_token = raw_webhook.split("?")

            if raw_token.startswith("token="):
                _, token = raw_token.split("=")
        else:
            webhook = raw_webhook

        logger.info(f"webhook: {webhook}, token: {token}")

        _idx_t0 = time.monotonic()
        _idx_error: str | None = None
        pc_result = None
        try:
            pc_result = await add_item_hybrid(item)
        except Exception as _idx_exc:
            _idx_error = str(_idx_exc)
            raise
        finally:
            _idx_duration_ms = int((time.monotonic() - _idx_t0) * 1000)
            _et, _pl = analytics.events.content_indexed(
                kb_id=item.namespace,
                kb_name=item.namespace,
                embedding_model=analytics.events.get_embedding_model_name(
                    item.embedding
                ),
                engine=analytics.events.get_engine_value(item.engine),
                duration_ms=_idx_duration_ms,
                success=_idx_error is None,
                source_url=item.source,
                source_type=item.type,
                chunks_indexed=(pc_result.chunks or 0)
                if _idx_error is None and pc_result is not None
                else 0,
                error_message=_idx_error,
                request_id=item.request_id,
            )
            analytics.publish_nowait(_et, item.id_project, _pl)

        scrape_status_response = ScrapeStatusResponse(
            status_message="Indexing finish", status_code=3
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        return JSONResponse(
            content=pc_result.model_dump(exclude_none=True)
        )  # {"message": f"Item {item.id} created successfully"})

    except Exception as e:
        scrape_status_response = ScrapeStatusResponse(
            status_message="Error", status_code=4
        )
        add_to_queue = await redis_client.set(
            f"{item.namespace}/{item.id}",
            scrape_status_response.model_dump_json(),
            ex=expiration_in_seconds,
        )

        logger.error(f"Error {add_to_queue}")
        import traceback

        traceback.print_exc()
        logger.error(e)
        raise HTTPException(status_code=400, detail=repr(e))


@app.post(
    "/api/qa",
    response_model=Union[RetrievalResult, RetrievalChunksResult],
    tags=["Question & Answer"],
)
async def post_ask_with_memory_main(question_answer: QuestionAnswer):
    """
    Query and Answer with chat history.
    Pass use_cache=true to enable semantic cache (L1 exact + L2 cosine similarity).
    """
    logger.debug(question_answer)

    if question_answer.chunks_only:
        _chunks_t0 = time.monotonic()
        _chunks_result = await ask_for_chunks(question_answer)
        _chunks_latency_ms = int((time.monotonic() - _chunks_t0) * 1000)
        # Analytics: emit kb.query_executed for chunks_only (fire-and-forget)
        from tilellm.analytics import events as an_events
        _reranker_model = an_events.get_reranker_model(question_answer)
        _et, _pl = an_events.kb_query(
            kb_id=question_answer.namespace,
            kb_name=question_answer.namespace,
            query_text=question_answer.question
            if isinstance(question_answer.question, str)
            else str(question_answer.question),
            chunks_retrieved=len(_chunks_result.chunks or [])
            if hasattr(_chunks_result, "chunks")
            else 0,
            reranking_applied=bool(question_answer.reranking),
            reranker_model=_reranker_model,
            latency_ms=_chunks_latency_ms,
            request_id=question_answer.request_id,
        )
        analytics.publish_nowait(_et, question_answer.id_project, _pl)
        return _chunks_result

    # Semantic cache lookup (only when use_cache=True)
    query_embedding = None
    if question_answer.use_cache:
        from tilellm.shared.cache import SemanticCache
        from tilellm.shared.embedding_factory import create_embedding_instance

        try:
            embedding_model, _ = await create_embedding_instance(question_answer)
            query_text = question_answer.retrieval_query or question_answer.question
            query_embedding = await embedding_model.aembed_query(query_text)
            cached = await SemanticCache.lookup(
                namespace=question_answer.namespace,
                question=question_answer.question,
                embedding=query_embedding,
            )
            if cached is not None:
                result = RetrievalResult(
                    **{k: v for k, v in cached.items() if not k.startswith("_")}
                )
                logger.info(
                    f"/api/qa cache hit (level={cached.get('_cache_level')}, cosine={cached.get('_cache_similarity', 1.0):.4f})"
                )
                # Analytics: emit kb.query_executed for cache hit (fire-and-forget)
                from tilellm.analytics import events as an_events
                _reranker_model = an_events.get_reranker_model(question_answer)
                _et, _pl = an_events.kb_query(
                    kb_id=question_answer.namespace,
                    kb_name=question_answer.namespace,
                    query_text=question_answer.question
                    if isinstance(question_answer.question, str)
                    else str(question_answer.question),
                    chunks_retrieved=len(result.content_chunks or [])
                    if hasattr(result, "content_chunks")
                    else 0,
                    reranking_applied=False,
                    reranker_model=_reranker_model,
                    latency_ms=0,
                    request_id=question_answer.request_id,
                )
                analytics.publish_nowait(_et, question_answer.id_project, _pl)
                return result
        except Exception as e:
            logger.warning(
                f"/api/qa cache lookup failed ({e}), proceeding without cache"
            )

    if question_answer.search_type == "hybrid":
        _qa_t0 = time.monotonic()
        result = await ask_hybrid_with_memory(question_answer)
    else:
        _qa_t0 = time.monotonic()
        result = await ask_with_memory(question_answer)
    _qa_latency_ms = int((time.monotonic() - _qa_t0) * 1000)

    # Store in cache only on successful results, reusing the embedding computed at lookup
    if question_answer.use_cache and query_embedding is not None and getattr(result, "success", True):
        from tilellm.shared.cache import SemanticCache
        try:
            body = result.model_dump() if hasattr(result, "model_dump") else {}
            await SemanticCache.store(
                namespace=question_answer.namespace,
                question=question_answer.question,
                embedding=query_embedding,
                body=body,
            )
        except Exception as e:
            logger.warning(f"/api/qa cache store failed ({e})")

    logger.debug(result)

    # Analytics: emit kb.query_executed (fire-and-forget)
    from tilellm.analytics import events as an_events

    _reranker_model = an_events.get_reranker_model(question_answer)
    _et, _pl = an_events.kb_query(
        kb_id=question_answer.namespace,
        kb_name=question_answer.namespace,
        query_text=question_answer.question
        if isinstance(question_answer.question, str)
        else str(question_answer.question),
        chunks_retrieved=len(result.content_chunks or [])
        if hasattr(result, "content_chunks")
        else 0,
        reranking_applied=bool(question_answer.reranking),
        reranker_model=_reranker_model,
        latency_ms=_qa_latency_ms,
        request_id=question_answer.request_id,
    )
    analytics.publish_nowait(_et, question_answer.id_project, _pl)

    return result


@app.post("/api/ask", response_model=SimpleAnswer, tags=["Question & Answer"])
async def post_ask_to_llm_main(question: QuestionToLLM):
    """
    Query and Answer with a LLM.

    Routing logic:
    - Se non ci sono MCP servers -> usa ask_to_llm (semplice)
    - Se ci sono MCP servers:
      - Se question è una stringa semplice -> usa ask_mcp_agent_llm_simple
      - Se question è complessa (lista/multimodale) -> usa ask_mcp_agent_llm (complesso)

    :param question:
    :return: SimpleAnswer
    """
    logger.info(question)

    if not (question.servers or question.tools):
        # Nessun MCP server -> usa l'LLM semplice
        return await ask_to_llm(question=question)
    else:
        # Ci sono MCP servers -> verifica la complessità
        is_simple_string = isinstance(question.question, str)

        # Se è una stringa, prova a verificare se è un JSON/lista
        if is_simple_string:
            try:
                parsed = json.loads(question.question)
                # Se il parsing riesce ed è una lista, è complesso
                if isinstance(parsed, list):
                    is_simple_string = False
            except (json.JSONDecodeError, ValueError):
                # Non è JSON, rimane stringa semplice
                pass

        if is_simple_string:
            # Stringa semplice -> usa la versione semplice MCP
            logger.info("Using ask_mcp_agent_llm_simple (simple string input)")
            # Per ora usa la versione complessa
            return await ask_mcp_agent_llm_simple(question=question)
        else:
            # Input complesso (lista o multimodale) -> usa la versione complessa MCP
            logger.info("Using ask_mcp_agent_llm (complex/multimodal input)")
            return await ask_mcp_agent_llm(question=question)


@app.post("/api/thinking", response_model=SimpleAnswer, tags=["Question & Answer"])
async def post_ask_to_llm_reason_main(question: QuestionToLLM):
    """
    Query and Answer with a LLM
    :param question:
    :return: RetrievalResult
    """
    logger.info(question)

    return await ask_reason_llm(question=question)


@app.post("/api/scrape/status", response_model=ScrapeStatusResponse, tags=["Scrape"])
async def scrape_status_main(
    scrape_status_req: ScrapeStatusReq, redis_client: Redis = Depends(get_redis_client)
):
    """
    Check status of indexing
    :param scrape_status_req:
    :param redis_client:
    :return:
    """
    try:
        retrieved_data = await redis_client.get(
            f"{scrape_status_req.namespace}/{scrape_status_req.id}"
        )
        if retrieved_data:
            logger.debug(retrieved_data)
            scrape_status_response = ScrapeStatusResponse.model_validate(
                json.loads(retrieved_data.decode("utf-8"))
            )
            return JSONResponse(content=scrape_status_response.model_dump())
        else:
            try:
                repository_engine = RepositoryEngine(engine=scrape_status_req.engine)
                print(repository_engine.engine)
                retrieved_pinecone_data = await get_ids_namespace(
                    repository_engine,
                    metadata_id=scrape_status_req.id,
                    namespace=scrape_status_req.namespace,
                )

                if retrieved_pinecone_data.matches:
                    logger.debug(retrieved_pinecone_data.matches[0].date)
                    date_from_metadata = retrieved_pinecone_data.matches[0].date
                    scrape_status_response = ScrapeStatusResponse(
                        status_message=f"Indexing finished - verified in Pinecone metadata - date:{date_from_metadata}",
                        status_code=3,
                        queue_order=-1,
                    )
                    return JSONResponse(content=scrape_status_response.model_dump())

                else:
                    raise Exception("Pinecone data not found")
            except Exception as int_ex:
                raise Exception(
                    f"{repr(int_ex)}, id: {scrape_status_req.id}, namespace: {scrape_status_req.namespace}"
                )

    except Exception as ex:
        raise HTTPException(status_code=400, detail=repr(ex))


@app.post(
    "/api/delete/id",
    deprecated=True,
    description="This endpoint is deprecated and  is no longer supported. "
    "Use method DELETE /api/id/{id}/namespace/{namespace}",
    tags=["Namespace"],
)
async def delete_item_id_namespace_post(item_to_delete: RepositoryItem):
    """
    Delete items from namespace given document id via POST.
    :param item_to_delete:
    :return:
    """
    try:
        metadata_id = item_to_delete.id
        namespace = item_to_delete.namespace

        logger.info(f"delete of id {metadata_id} dal namespace {namespace}")
        await delete_id_from_namespace(item_to_delete, metadata_id, namespace)

        return JSONResponse(
            content={
                "success": True,
                "message": f"ids {metadata_id} in Namespace {namespace} deleted",
            }
        )
    except Exception as ex:
        return JSONResponse(
            content={
                "success": False,
                "message": f"ids {metadata_id} in Namespace {namespace} not deleted due to {repr(ex)}",
            }
        )
        # raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/delete/namespace", tags=["Namespace"])
async def delete_namespace_main_post(namespace_to_delete: RepositoryNamespace):
    """
    Delete Pinecone namespace by namespace_id
    :param namespace_to_delete:
    :return:
    """
    try:
        await delete_namespace(namespace_to_delete)
        return JSONResponse(
            content={
                "success": "true",
                "message": f"{namespace_to_delete.namespace} is deleted from database",
            }
        )
    except Exception as ex:
        return JSONResponse(
            content={
                "success": "false",
                "message": f"namespace {namespace_to_delete.namespace} is not deleted. {repr(ex)}",
            }
        )


@app.get(
    "/api/list/namespace/{token}",
    response_model=RepositoryNamespaceResult,
    tags=["Namespace"],
)
async def list_namespace_main(token: str):
    """
    Get all namespaces with id and vector count
    :return: list of namespace
    """
    try:
        engine_dec = decode_jwt(token)
        # print(type(engine_dec))
        logger.debug("All Namespaces ")
        repository_engine = RepositoryEngine(**engine_dec)
        result = await get_list_namespace(repository_engine)
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get(
    "/api/id/{metadata_id}/namespace/{namespace}/{token}",
    response_model=RepositoryItems,
    tags=["Namespace"],
)
async def get_items_id_namespace_main(token: str, metadata_id: str, namespace: str):
    """
    Get all items from namespace given id of document
    :param token
    :param metadata_id:
    :param namespace:
    :return:
    """
    try:
        logger.info(f"retrieve id {metadata_id} dal namespace {namespace}")
        engine_dec = decode_jwt(token)
        repository_engine = RepositoryEngine(**engine_dec)
        result = await get_ids_namespace(repository_engine, metadata_id, namespace)

        return JSONResponse(content=result.model_dump())
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get(
    "/api/desc/namespace/{namespace}/{token}",
    response_model=RepositoryDescNamespaceResult,
    tags=["Namespace"],
)
async def list_namespace_items_desc_main(token: str, namespace: str):
    """
    Get description for given namespace
    :param token
    :param namespace: namespace_id
    :return: description of namespace
    """
    try:
        logger.info(f"retrieve description for namespace {namespace}")
        engine_dec = decode_jwt(token)
        repository_engine = RepositoryEngine(**engine_dec)

        result = await get_desc_namespace(repository_engine, namespace)

        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get(
    "/api/listitems/namespace/{namespace}/{token}",
    response_model=RepositoryItems,
    tags=["Namespace"],
)
async def list_namespace_items_main(token: str, namespace: str):
    """
    Get all item with given namespace
    :param token
    :param namespace: namespace_id
    :return: list of all item into namespace
    """
    try:
        logger.info(f"retrieve namespace {namespace}")
        engine_dec = decode_jwt(token)
        repository_engine = RepositoryEngine(**engine_dec)
        result = await get_listitems_namespace(repository_engine, namespace)

        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get(
    "/api/listcompleteitems/namespace/{namespace}/all",
    response_model=RepositoryItems,
    tags=["Namespace"],
)
async def list_namespace_items_with_text(
    namespace: str, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get all items with given namespace
    :param credentials: Bearer token
    :param namespace: namespace_id
    :return: list of all items in namespace
    """
    try:
        token = credentials.credentials  # estrae il token dall'header
        logger.info(f"retrieve namespace {namespace}  Raw token: %s {token}")
        engine_dec = decode_jwt(token)
        logger.info(f"asd {engine_dec}")
        repository_engine = RepositoryEngine(**engine_dec)
        result = await get_listitems_namespace(repository_engine, namespace, True)
        return result  # FastAPI serializzerà il modello
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get(
    "/api/items", response_model=RepositoryItems, tags=["Namespace"]
)  # ?source={source}&namespace={namespace}&token={token}
async def get_items_source_namespace_main(source: str, namespace: str, token: str):
    """
    Get all item given the source and namespace
    :param source: source of document
    :param namespace: namespace id
    :param token:
    :return: list of all item
    """
    try:
        logger.info(f"retrieve source: {source}, namespace: {namespace}")
        engine_dec = decode_jwt(token)
        repository_engine = RepositoryEngine(**engine_dec)
        from urllib.parse import unquote

        source = unquote(source)
        result = await get_sources_namespace(repository_engine, source, namespace)

        return JSONResponse(content=result.model_dump())
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.delete("/api/namespace/{namespace}/{token}", tags=["Namespace"])
async def delete_namespace_main(token: str, namespace: str):
    """
    Delete namespace from index
    :param token
    :param namespace:
    :return:
    """
    try:
        engine_dec = decode_jwt(token)
        engine = Engine(**engine_dec["engine"])

        namespace_to_delete = RepositoryNamespace(namespace=namespace, engine=engine)

        await delete_namespace(namespace_to_delete)
        # Invalidate cache for this namespace (strategy B: full namespace invalidation)
        from tilellm.shared.cache import SemanticCache

        await SemanticCache.invalidate_namespace(namespace)
        return JSONResponse(content={"message": f"Namespace {namespace} deleted"})
    except Exception as ex:
        return JSONResponse(
            content={
                "success": "false",
                "message": f"namespace is not deleted. {repr(ex)}",
            }
        )
        # raise HTTPException(status_code=400, detail=repr(ex))


@app.delete("/api/chunk/{chunk_id}/namespace/{namespace}/{token}", tags=["Namespace"])
async def delete_item_chunk_id_namespace_main(
    token: str, chunk_id: str, namespace: str
):
    """
    Delete items from namespace identified by id and namespace
    :param token
    :param chunk_id:
    :param namespace:
    :return:
    """
    try:
        logger.info(f"delete id {chunk_id} dal namespace {namespace}")
        engine_dec = decode_jwt(token)
        repository_engine = RepositoryEngine(**engine_dec)

        await delete_chunk_id_from_namespace(repository_engine, chunk_id, namespace)

        return JSONResponse(
            content={"message": f"ids {chunk_id} in Namespace {namespace} deleted"}
        )
    except Exception as ex:
        print(repr(ex))
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.delete("/api/id/{metadata_id}/namespace/{namespace}/{token}", tags=["Namespace"])
async def delete_item_id_namespace_main(token: str, metadata_id: str, namespace: str):
    """
    Delete items from namespace identified by id and namespace
    :param token
    :param metadata_id:
    :param namespace:
    :return:
    """
    try:
        logger.info(f"delete id {metadata_id} dal namespace {namespace}")
        engine_dec = decode_jwt(token)
        engine = Engine(**engine_dec["engine"])
        item_to_delete = RepositoryItem(
            id=metadata_id, namespace=namespace, engine=engine
        )
        # repository_engine = RepositoryEngine(**engine_dec)
        await delete_id_from_namespace(item_to_delete, metadata_id, namespace)

        return JSONResponse(
            content={"message": f"ids {metadata_id} in Namespace {namespace} deleted"}
        )
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get("/metrics", tags=["Observability"], include_in_schema=False)
async def get_prometheus_metrics():
    """Prometheus scrape endpoint — exposes cache hit/miss/latency counters."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"error": "prometheus_client not installed"}, status_code=501
        )


@app.get("/api/cache/stats", tags=["Cache"])
async def get_cache_stats(namespace: Optional[str] = None):
    """Return semantic cache stats (entry count per namespace or all)."""
    from tilellm.shared.cache import SemanticCache

    return await SemanticCache.stats(namespace=namespace)


@app.delete("/api/cache/namespace/{namespace}", tags=["Cache"])
async def delete_cache_namespace(namespace: str):
    """Manually invalidate all cache entries for a namespace."""
    from tilellm.shared.cache import SemanticCache

    deleted = await SemanticCache.invalidate_namespace(namespace)
    return {"deleted": deleted, "namespace": namespace}


@app.get("/")
async def get_root_endpoint():
    return "Hello from Tiledesk LLM python server!!"


def register_feature_routers(_app: FastAPI, base_package_dir: str):
    """
    Scansiona una directory per trovare e registrare i router delle funzionalità.
    Se la directory non esiste, continua senza errori.
    """
    import importlib
    from fastapi import APIRouter
    from tilellm.shared.utility import get_service_config

    base_path = Path(base_package_dir)

    # --- MODIFICA CHIAVE ---
    # Controlla se la directory dei moduli esiste.
    if not base_path.is_dir():
        # Se non esiste, stampa un messaggio informativo e termina la funzione.
        logger.warning(
            f"Directory dei moduli '{base_package_dir}' non trovata. Nessun router dinamico sarà caricato."
        )
        return
    # -----------------------

    # Carica la configurazione dei servizi
    service_config = get_service_config()
    services_enabled = service_config.get("services", {})

    # Mappatura directory -> chiave di configurazione
    module_config_mapping = {
        "task_executor": "task_executor",
        "knowledge_graph": "graphrag",
        "knowledge_graph_falkor": "graphrag_falkor",
        "raptor": "raptor",
        "compliance_checker": "compliance",
        "pdf_ocr": "pdf_ocr",
        "conversion": "conversion",
        "tools_registry": "tools_registry",
        "api_v2": "api_v2",
        "ingestion": "ingestion",
    }

    # Se la configurazione non esiste o è vuota, abilita tutti i moduli (compatibilità all'indietro)
    enable_all = len(services_enabled) == 0

    package_name = str(base_path).replace(os.path.sep, ".")

    logger.info(f"🔍 Sto cercando i servizi nella directory: '{base_path}'...")
    if not enable_all:
        logger.info(f"📋 Configurazione servizi: {services_enabled}")

    found_routers = False
    for service_dir in base_path.iterdir():
        if service_dir.is_dir():
            module_name = service_dir.name
            config_key = module_config_mapping.get(module_name)

            # Determina se il modulo deve essere caricato
            should_load = False
            if module_name == "ingestion":
                should_load = True
            elif enable_all:
                should_load = True
            elif config_key:
                should_load = services_enabled.get(config_key, False)
            else:
                # Modulo non nella mappatura, caricamento di default (disabilitato per sicurezza)
                should_load = False
                logger.info(
                    f"⚠️  Modulo '{module_name}' non mappato, verrà saltato. Aggiungilo a module_config_mapping per abilitarlo."
                )

            if not should_load:
                logger.info(
                    f"⏭️  Modulo '{module_name}' disabilitato in configurazione, salto."
                )
                continue

            controller_file = service_dir / "controllers.py"

            if controller_file.exists():
                module_path = f"{package_name}.{module_name}.controllers"

                try:
                    module = importlib.import_module(module_path)

                    if hasattr(module, "router") and isinstance(
                        module.router, APIRouter
                    ):
                        logger.info(
                            f"✅ Trovato e registrato il router da: '{module_path}'"
                        )
                        _app.include_router(module.router)
                        found_routers = True
                    else:
                        logger.info(
                            f"⚠️  Nel modulo '{module_path}' non è stato trovato un APIRouter chiamato 'router'."
                        )

                except Exception as e:
                    logger.info(
                        f"❌ Errore durante il caricamento di '{module_path}': {e}"
                    )
            else:
                logger.info(f"📁 Modulo '{module_name}' non ha controllers.py, salto.")

    if not found_routers:
        logger.info("ℹ️  Nessun router valido trovato nei moduli.")


register_feature_routers(app, "tilellm/modules")


def main():
    import uvicorn

    uvicorn.run(
        "tilellm.__main__:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )  # , log_config=args.log_path


if __name__ == "__main__":
    main()
