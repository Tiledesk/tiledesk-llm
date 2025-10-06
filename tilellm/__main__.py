import base64
import os
from binascii import Error
from contextlib import asynccontextmanager
from typing import Union, List

from fastapi import (FastAPI,
                     Depends,
                     HTTPException)
from fastapi_cprofile.profiler import CProfileMiddleware
from fastapi.responses import JSONResponse

import asyncio
from redis.asyncio import Redis, from_url
import aiohttp
import json
from dotenv import load_dotenv

from tilellm.controller.conversion_controller import process_xlsx_to_csv, process_pdf_to_text
from tilellm.models.convertion import ConvertedFile, ConversionRequest, ConversionType
from tilellm.shared.const import populate_constant
from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.utility import decode_jwt
from tilellm.models import (ItemSingle,
                            QuestionToAgent,
                            Engine,
                            QuestionToLLM,
                            QuestionAnswer)
from tilellm.models.schemas import (RepositoryItem,
                                    RepositoryNamespace,
                                    ScrapeStatusReq,
                                    ScrapeStatusResponse,
                                    IndexingResult, RetrievalResult, RepositoryNamespaceResult,
                                    RepositoryDescNamespaceResult, RepositoryItems, SimpleAnswer,
                                    RepositoryEngine, RetrievalChunksResult)


from tilellm.store.redis_repository import redis_xgroup_create
from tilellm.controller.controller import (ask_with_memory,
                                           ask_hybrid_with_memory,
                                           ask_for_chunks,
                                           ask_with_sequence,
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
                                           ask_to_agent,
                                           ask_reason_llm, ask_mcp_agent_llm)

import logging


ENVIRONMENTS = {
    'serverless': '.environ',
    'pod': '.environ.prod',
}

expiration_in_seconds = 48 * 60 * 60

logger = logging.getLogger(__name__)


environment = os.environ.get("PINECONE_TYPE", "serverless")
# environment = "serverless"
load_dotenv(ENVIRONMENTS.get(environment) or '.environ')


# print(os.environ.get("PINECONE_API_KEY"))
# os.environ.__setitem__("ENVIRON", environment)

redis_url = os.environ.get("REDIS_URL")
tilellm_role = os.environ.get("TILELLM_ROLE")


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
                    streams={const.STREAM_NAME: '>'},
                    count=1,
                    block=0  # Set block to 0 for non-blocking
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
                        scrape_status_response = ScrapeStatusResponse(status_message="Indexing started",
                                                                      status_code=2
                                                                      )
                        add_to_queue = await channel.set(f"{item.get('namespace')}/{item.get('id')}",
                                                         scrape_status_response.model_dump_json(),
                                                         ex=expiration_in_seconds)

                        logger.debug(f"Start {add_to_queue}")

                        raw_webhook = item.get('webhook', "")
                        if '?' in raw_webhook:
                            webhook, raw_token = raw_webhook.split('?')

                            if raw_token.startswith('token='):
                                _, token = raw_token.split('=')
                        else:
                            webhook = raw_webhook

                        logger.info(f"webhook: {webhook}, token: {token}")

                        if webhook:
                            res = IndexingResult(id=item.get('id'), status=200)
                            try:
                                async with aiohttp.ClientSession() as session:
                                    res = await session.post(webhook,
                                                             json=res.model_dump(exclude_none=True),
                                                             headers={"Content-Type": "application/json",
                                                                      "X-Auth-Token": token})
                                    logger.info(f"200 {await res.json()}")
                            except Exception as ewh:
                                logger.error(ewh)
                                pass

                        pc_result = await add_item(item_single)
                        # import datetime
                        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")

                        # pc_result["date"]= current_time
                        # pc_result["status"] = current_time

                        # A POST request to the API

                        scrape_status_response = ScrapeStatusResponse(status_message="Indexing finish",
                                                                      status_code=3
                                                                      )
                        add_to_queue = await channel.set(f"{item.get('namespace')}/{item.get('id')}",
                                                         scrape_status_response.model_dump_json(),
                                                         ex=expiration_in_seconds)

                        logger.debug(f"End {add_to_queue}")
                        if webhook:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    res = await session.post(webhook,
                                                             json=pc_result.model_dump(exclude_none=True),
                                                             headers={"Content-Type": "application/json",
                                                                      "X-Auth-Token": token})
                                    logger.info(f"300 {await res.json()}")
                            except Exception as ewh:
                                logger.error(ewh)
                                pass

                        await channel.xack(
                            const.STREAM_NAME,
                            const.STREAM_CONSUMER_GROUP,
                            message_id)
                        logger.info(f"xack to message_id: {message_id}")

            except Exception as e:
                scrape_status_response = ScrapeStatusResponse(status_message="Error",
                                                              status_code=4
                                                              )
                add_to_queue = await channel.set(f"{item.get('namespace')}/{item.get('id')}",
                                                 scrape_status_response.model_dump_json(),
                                                 ex=expiration_in_seconds)

                logger.error(f"Error {add_to_queue}")
                import traceback
                if webhook:
                    res = IndexingResult(id=item.get('id'), status=400, error=repr(e))
                    async with aiohttp.ClientSession() as session:
                        response = await session.post(webhook,  json=res.model_dump(exclude_none=True),
                                                      headers={"Content-Type": "application/json", "X-Auth-Token": token})
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
    redis_client = from_url(redis_url)
    await redis_xgroup_create(redis_client)
    asyncio.create_task(reader(redis_client)) 
 
    yield 
    
    await redis_client.close()
    await TimedCache.async_clear_cache("vector_store_wrapper")



populate_constant()
app = FastAPI(lifespan=redis_consumer)


# Leggi la variabile d'ambiente per la profilazione
ENABLE_PROFILER = os.getenv("ENABLE_PROFILER", "False").lower() == "true"

if ENABLE_PROFILER:
    app.add_middleware(CProfileMiddleware, enable=True, print_each_request=True)

# Leggi la variabile d'ambiente per il livello di log
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.getLogger().setLevel(LOG_LEVEL)




@app.post("/api/scrape/enqueue" , tags=["Scrape"])
async def enqueue_scrape_item_main(item: ItemSingle, redis_client: Redis = Depends(get_redis_client)):
    """
    enqueue item to redis. Consumer read message and add it to namespace
    :param item:
    :param redis_client:
    :return: PineconeIndexingResult
    """
    from tilellm.shared import const
    logger.debug(item) 
    res = await redis_client.xadd(const.STREAM_NAME, {"single": item.model_dump_json()}, id="*")
    scrape_status_response = ScrapeStatusResponse(status_message="Document added to queue",
                                                  status_code=0
                                                  )
    addtoqueue = await redis_client.set(f"{item.namespace}/{item.id}",
                                        scrape_status_response.model_dump_json(),
                                        ex=expiration_in_seconds)

    logger.debug(res)

    return {"message": f"Item {item.id} created successfully, more {res}"}


@app.post("/api/scrape/single", response_model=IndexingResult, tags=["Scrape"])
async def create_scrape_item_single(item: ItemSingle, redis_client: Redis = Depends(get_redis_client)):
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
        scrape_status_response = ScrapeStatusResponse(status_message="Indexing started",
                                                      status_code=2
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(),
                                              ex=expiration_in_seconds)

        logger.debug(f"Start {add_to_queue}")

        raw_webhook = item.webhook
        if '?' in raw_webhook:
            webhook, raw_token = raw_webhook.split('?')

            if raw_token.startswith('token='):
                _, token = raw_token.split('=')
        else:
            webhook = raw_webhook

        logger.info(f"webhook: {webhook}, token: {token}")

        if item.hybrid:
            pc_result = await add_item_hybrid(item)
        else:
            pc_result = await add_item(item)

        scrape_status_response = ScrapeStatusResponse(status_message="Indexing finish",
                                                      status_code=3
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(),
                                              ex=expiration_in_seconds)

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
        scrape_status_response = ScrapeStatusResponse(status_message="Error",
                                                      status_code=4
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(by_alias=True),
                                              ex=expiration_in_seconds)

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
        raise HTTPException(status_code=400, detail=repr(e))

@app.post("/api/scrape/hybrid", response_model=IndexingResult, tags=["Scrape"])
async def create_scrape_item_hybrid(item: ItemSingle, redis_client: Redis = Depends(get_redis_client)):
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
        scrape_status_response = ScrapeStatusResponse(status_message="Indexing started",
                                                      status_code=2
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(),
                                              ex=expiration_in_seconds)

        logger.debug(f"Start {add_to_queue}")

        raw_webhook = item.webhook
        if '?' in raw_webhook:
            webhook, raw_token = raw_webhook.split('?')

            if raw_token.startswith('token='):
                _, token = raw_token.split('=')
        else:
            webhook = raw_webhook

        logger.info(f"webhook: {webhook}, token: {token}")

        pc_result = await add_item_hybrid(item)

        scrape_status_response = ScrapeStatusResponse(status_message="Indexing finish",
                                                      status_code=3
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(),
                                              ex=expiration_in_seconds)

        return JSONResponse(content=pc_result.model_dump(exclude_none=True)) # {"message": f"Item {item.id} created successfully"})

    except Exception as e:
        scrape_status_response = ScrapeStatusResponse(status_message="Error",
                                                      status_code=4
                                                      )
        add_to_queue = await redis_client.set(f"{item.namespace}/{item.id}",
                                              scrape_status_response.model_dump_json(),
                                              ex=expiration_in_seconds)

        logger.error(f"Error {add_to_queue}")
        import traceback

        traceback.print_exc()
        logger.error(e)
        raise HTTPException(status_code=400, detail=repr(e))


@app.post("/api/qa", response_model=Union[RetrievalResult, RetrievalChunksResult], tags=["Question & Answer"])
async def post_ask_with_memory_main(question_answer: QuestionAnswer):
    """
    Query and Aswer with chat history
    :param question_answer:
    :return: RetrievalResult
    """
    logger.debug(question_answer)

    if question_answer.chunks_only:
        result = await ask_for_chunks(question_answer)
    else:
        if question_answer.search_type == 'hybrid':
            result= await ask_hybrid_with_memory(question_answer)
        else:
            result = await ask_with_memory(question_answer)

    logger.debug(result)
    return result


@app.post("/api/agent", response_model=SimpleAnswer, tags=["Question & Answer"])
async def post_ask_to_agent_main(question_to_agent: QuestionToAgent):
    """
    Query and Aswer with chat history
    :param question_to_agent:
    :return: SimpleAnswer
    """
    print(question_to_agent)
    logger.debug(question_to_agent)

    result = await ask_to_agent(question_to_agent)

    logger.debug(result)
    return JSONResponse(content=result.model_dump())



@app.post("/api/ask", response_model=SimpleAnswer, tags=["Question & Answer"])
async def post_ask_to_llm_main(question: QuestionToLLM):
    """
    Query and Answer with a LLM
    :param question:
    :return: RetrievalResult
    """
    logger.info(question)
    if not question.servers:
        return await ask_to_llm(question=question)
    else:
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


@app.post("/api/qachain", response_model=RetrievalResult, tags=["Question & Answer"])
async def post_ask_with_memory_chain_main(question_answer: QuestionAnswer):
    logger.debug(question_answer)
    result = await ask_with_sequence(question_answer)
    logger.debug(result)
    return JSONResponse(content=result.model_dump())
    # return result


@app.post("/api/convert", response_model=List[ConvertedFile], tags=["Conversion"])
async def convert_file(request: ConversionRequest):
    """
    Converte un file fornito come stringa Base64.

    - **xlsx_to_csv**: Converte ogni foglio di un file XLSX in un file CSV separato.
    - **pdf_to_text**: Estrae il contenuto testuale da un file PDF.
    """
    try:
        # Decodifica la stringa Base64 per ottenere i byte del file
        file_bytes = base64.b64decode(request.file_content)
    except (Error, ValueError):
        raise HTTPException(status_code=400, detail="Contenuto del file non valido: la stringa Base64 non è corretta.")

    if request.conversion_type == ConversionType.XLSX_TO_CSV:
        return process_xlsx_to_csv(request.file_name, file_bytes)

    elif request.conversion_type == ConversionType.PDF_TO_TEXT:
        return process_pdf_to_text(request.file_name, file_bytes)

    # Questo non dovrebbe mai accadere grazie alla validazione Pydantic, ma è una sicurezza in più.
    raise HTTPException(status_code=400, detail="Tipo di conversione non supportato.")



@app.post("/api/scrape/status", response_model=
          ScrapeStatusResponse, tags=["Scrape"])
async def scrape_status_main(scrape_status_req: ScrapeStatusReq,
                             redis_client: Redis = Depends(get_redis_client)):
    """
    Check status of indexing
    :param scrape_status_req:
    :param redis_client:
    :return:
    """
    try:
        retrieved_data = await redis_client.get(f"{scrape_status_req.namespace}/{scrape_status_req.id}")
        if retrieved_data:
            logger.debug(retrieved_data)
            scrape_status_response = ScrapeStatusResponse.model_validate(json.loads(retrieved_data.decode('utf-8')))
            return JSONResponse(content=scrape_status_response.model_dump())
        else:
            try:
                repository_engine = RepositoryEngine(engine=scrape_status_req.engine)
                print(repository_engine.engine)
                retrieved_pinecone_data = await get_ids_namespace(repository_engine,
                                                                  metadata_id=scrape_status_req.id,
                                                                  namespace=scrape_status_req.namespace)


                if retrieved_pinecone_data.matches:
                    logger.debug(retrieved_pinecone_data.matches[0].date)
                    date_from_metadata = retrieved_pinecone_data.matches[0].date
                    scrape_status_response = ScrapeStatusResponse(
                        status_message=f"Indexing finished - verified in Pinecone metadata - date:{date_from_metadata}",
                        status_code=3,
                        queue_order=-1)
                    return JSONResponse(content=scrape_status_response.model_dump())

                else:
                    raise Exception("Pinecone data not found")
            except Exception as int_ex:
                raise Exception(f"{repr(int_ex)}, id: {scrape_status_req.id}, namespace: {scrape_status_req.namespace}")

    except Exception as ex:
        raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/delete/id", deprecated=True,
          description="This endpoint is deprecated and  is no longer supported. "
                      "Use method DELETE /api/id/{id}/namespace/{namespace}", tags=["Namespace"])
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
        result = await delete_id_from_namespace(item_to_delete, metadata_id, namespace)

        return JSONResponse(content={"success": True, "message": f"ids {metadata_id} in Namespace {namespace} deleted"})
    except Exception as ex:
        return JSONResponse(content={"success": False, "message": f"ids {metadata_id} in Namespace {namespace} not deleted due to {repr(ex)}"})
        # raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/delete/namespace", tags=["Namespace"])
async def delete_namespace_main(namespace_to_delete: RepositoryNamespace):
    """
    Delete Pinecone namespace by namespace_id
    :param namespace_to_delete:
    :return:
    """
    try:
        result = await delete_namespace(namespace_to_delete)
        return JSONResponse(content={"success": "true", "message": f"{namespace_to_delete.namespace} is deleted from database"})
    except Exception as ex:
        return JSONResponse(content={"success": "false", "message": f"namespace {namespace_to_delete.namespace} is not deleted. {repr(ex)}"})
        #raise HTTPException(status_code=400, detail={"success": "false", "message": repr(ex)})


@app.get("/api/list/namespace/{token}", response_model=RepositoryNamespaceResult, tags=["Namespace"])
async def list_namespace_main(token: str):
    """
    Get all namespaces with id and vector count
    :return: list of namespace
    """
    try:
        engine_dec = decode_jwt(token)
        #print(type(engine_dec))
        logger.debug(f"All Namespaces ")
        repository_engine = RepositoryEngine(**engine_dec)
        result = await get_list_namespace(repository_engine)
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get("/api/id/{metadata_id}/namespace/{namespace}/{token}", response_model=RepositoryItems, tags=["Namespace"])
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


@app.get("/api/desc/namespace/{namespace}/{token}", response_model=RepositoryDescNamespaceResult, tags=["Namespace"])
async def list_namespace_items_main(token: str, namespace: str):
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


@app.get("/api/listitems/namespace/{namespace}/{token}", response_model=RepositoryItems, tags=["Namespace"])
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


@app.get("/api/items", response_model=RepositoryItems, tags=["Namespace"])#?source={source}&namespace={namespace}&token={token}
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
        raise HTTPException(status_code=400, detail=repr(ex) )


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

        result = await delete_namespace(namespace_to_delete)
        return JSONResponse(content={"message": f"Namespace {namespace} deleted"})
    except Exception as ex:
        return JSONResponse(content={"success": "false",
                                     "message": f"namespace is not deleted. {repr(ex)}"})
        #raise HTTPException(status_code=400, detail=repr(ex))


@app.delete("/api/chunk/{chunk_id}/namespace/{namespace}/{token}", tags=["Namespace"])
async def delete_item_chunk_id_namespace_main(token: str, chunk_id: str, namespace: str):
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

        result = await delete_chunk_id_from_namespace(repository_engine, chunk_id, namespace)

        return JSONResponse(content={"message": f"ids {chunk_id} in Namespace {namespace} deleted"})
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
        item_to_delete = RepositoryItem(id=metadata_id,
                                        namespace=namespace,
                                        engine=engine
        )
        #repository_engine = RepositoryEngine(**engine_dec)
        result = await delete_id_from_namespace(item_to_delete, metadata_id, namespace)

        return JSONResponse(content={"message": f"ids {metadata_id} in Namespace {namespace} deleted"})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get("/")
async def get_root_endpoint():
    return "Hello from Tiledesk LLM python server!!"


def main():
    logger.debug(f"Environment: {environment}")
    import uvicorn
    uvicorn.run("tilellm.__main__:app", host="0.0.0.0", port=8000, reload=True, log_level="info")#, log_config=args.log_path


if __name__ == "__main__":

    main()
