import os
from contextlib import asynccontextmanager
from fastapi import (FastAPI, 
                     Depends,
                     HTTPException)
from fastapi.responses import JSONResponse

# import argparse

import aioredis
import asyncio
from aioredis import from_url
import aiohttp
import json
from dotenv import load_dotenv

from tilellm.shared.const import populate_constant
from tilellm.models.item_model import (ItemSingle,
                                       QuestionAnswer,
                                       PineconeItemToDelete,
                                       PineconeNamespaceToDelete,
                                       ScrapeStatusReq,
                                       ScrapeStatusResponse,
                                       PineconeIndexingResult)

from tilellm.store.redis_repository import redis_xgroup_create
from tilellm.controller.openai_controller import (ask_with_memory,
                                                  ask_with_sequence,
                                                  add_pc_item,
                                                  delete_namespace,
                                                  delete_id_from_namespace,
                                                  get_ids_namespace,
                                                  get_listitems_namespace,
                                                  get_desc_namespace,
                                                  get_list_namespace,
                                                  get_sources_namespace)

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
    try:
        redis_client = await from_url(redis_url)
        yield redis_client
    finally:
        await redis_client.close()


async def reader(channel: aioredis.client.Redis):
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
                            res = PineconeIndexingResult(id=item.get('id'), status=200)
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

                        pc_result = await add_pc_item(item_single)
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
                    res = PineconeIndexingResult(id=item.get('id'), status=400, error=repr(e))
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


populate_constant()
app = FastAPI(lifespan=redis_consumer)


@app.post("/api/scrape/enqueue")
async def enqueue_scrape_item_main(item: ItemSingle, redis_client: aioredis.client.Redis = Depends(get_redis_client)):
    """
    enqueue item to redis. Consumer read message and add it to namespace
    :param item:
    :param redis_client:
    :return:
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

@app.post("/api/scrape/single")
async def create_scrape_item_single(item: ItemSingle, redis_client: aioredis.client.Redis = Depends(get_redis_client)):
    """
    Add item to namespace
    :param item:
    :param redis_client:
    :return:
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

        # if webhook:
        #    res = PineconeIndexingResult(id=item.id, status=200)
        #    try:
        #        async with aiohttp.ClientSession() as session:
        #            res = await session.post(webhook,
        #                                     json=res.model_dump(exclude_none=True),
        #                                     headers={"Content-Type": "application/json",
        #                                              "X-Auth-Token": token})
        #            logger.info(f"200 {await res.json()}")
        #    except Exception as ewh:
        #        logger.error(ewh)
        #        pass

        pc_result = await add_pc_item(item)
        # import datetime
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")

        # pc_result["date"]= current_time
        # pc_result["status"] = current_time

        # A POST request to the API

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

@app.post("/api/qa")
async def post_ask_with_memory_main(question_answer: QuestionAnswer):
    logger.debug(question_answer)

    result = await ask_with_memory(question_answer)

    logger.debug(result)
    return JSONResponse(content=result)


@app.post("/api/qachain")
async def post_ask_with_memory_chain_main(question_answer: QuestionAnswer):
    print(question_answer)
    logger.debug(question_answer)
    result = ask_with_sequence(question_answer)
    logger.debug(result)
    return JSONResponse(content=result)
    # return result


@app.get("/api/list/namespace")
async def list_namespace_main():
    """
    Get all namespaces with id and vector count
    :return: list of namespace
    """
    try:
        logger.debug(f"All Namespaces ")
        result = await get_list_namespace()
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))

@app.get("/api/desc/namespace/{namespace}")
async def list_namespace_items_main(namespace: str):
    """
    Get description for given namespace
    :param namespace: namespace_id
    :return: description of namespace
    """
    try:
        logger.info(f"retrieve description for namespace {namespace}")
        result = await get_desc_namespace(namespace)

        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))

@app.get("/api/listitems/namespace/{namespace}")
async def list_namespace_items_main(namespace: str):
    """
    Get all item with given namespace
    :param namespace: namespace_id
    :return: list of all item into namespace
    """
    try:
        logger.info(f"retrieve namespace {namespace}")
        result = await get_listitems_namespace(namespace)

        return JSONResponse(content=result.model_dump(exclude_none=True))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/scrape/status")
async def scrape_status_main(scrape_status_req: ScrapeStatusReq,
                             redis_client: aioredis.client.Redis = Depends(get_redis_client)):
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
                retrieved_pinecone_data = await get_ids_namespace(metadata_id=scrape_status_req.id,
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


@app.delete("/api/namespace/{namespace}")
async def delete_namespace_main(namespace: str):
    """
    Delete namespace from index
    :param namespace:
    :return:
    """
    try:
        result = await delete_namespace(namespace)
        return JSONResponse(content={"message": f"Namespace {namespace} deleted"})
    except Exception as ex:
        raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/delete/namespace")
async def delete_namespace_main(namespace_to_delete: PineconeNamespaceToDelete):
    """
    Delete Pinecone namespace by namespace_id
    :param namespace_to_delete:
    :return:
    """
    try:
        result = await delete_namespace(namespace_to_delete.namespace)
        return JSONResponse(content={"success": "true", "message": f"{namespace_to_delete.namespace} is deleted from database"})
    except Exception as ex:
        raise HTTPException(status_code=400, detail={"success": "false", "message": repr(ex)})


@app.delete("/api/id/{metadata_id}/namespace/{namespace}")
async def delete_item_id_namespace_main(metadata_id: str, namespace: str):
    """
    Delete items from namespace identified by id and namespace
    :param metadata_id:
    :param namespace:
    :return:
    """
    try:
        logger.info(f"delete id {metadata_id} dal namespace {namespace}")
        result = await delete_id_from_namespace(metadata_id, namespace)

        return JSONResponse(content={"message": f"ids {metadata_id} in Namespace {namespace} deleted"})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.post("/api/delete/id", deprecated=True,
          description="This endpoint is deprecated and  is no longer supported. "
                      "Use method DELETE /api/id/{id}/namespace/{namespace}")
async def delete_item_id_namespace_post(item_to_delete: PineconeItemToDelete):
    """
    Delete items from namespace given document id via POST.
    :param item_to_delete:
    :return:
    """
    try:
        metadata_id = item_to_delete.id
        namespace = item_to_delete.namespace
        logger.info(f"delete of id {metadata_id} dal namespace {namespace}")
        result = await delete_id_from_namespace(metadata_id, namespace)

        return JSONResponse(content={"success": True, "message": f"ids {metadata_id} in Namespace {namespace} deleted"})
    except Exception as ex:
        return JSONResponse(content={"success": True, "message": f"ids {metadata_id} in Namespace {namespace} not deleted due to {repr(ex)}"})
        # raise HTTPException(status_code=400, detail=repr(ex))


@app.get("/api/id/{metadata_id}/namespace/{namespace}")
async def get_items_id_namespace_main(metadata_id: str, namespace: str):
    """
    Get all items from namespace given id of document
    :param metadata_id:
    :param namespace:
    :return:
    """
    try:
        logger.info(f"retrieve id {metadata_id} dal namespace {namespace}")
        result = await get_ids_namespace(metadata_id,namespace)

        return JSONResponse(content=result.model_dump())
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex))


@app.get("/api/items")#?source={source}&namespace={namespace}
async def get_items_source_namespace_main(source: str, namespace: str ):
    """
    Get all item given the source and namespace
    :param source: source of document
    :param namespace: namespace id
    :return: list of all item
    """
    try:
        logger.info(f"retrieve source: {source}, namespace: {namespace}")

        from urllib.parse import unquote
        source = unquote(source)
        result = await get_sources_namespace(source,namespace)

        return JSONResponse(content=result.model_dump())
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=400, detail=repr(ex) )


@app.get("/")
async def get_root_endpoint():
    return "Hello from Tiledesk LLM python server!!"


def main():
    logger.debug(f"Environment: {environment}")
    import uvicorn
    uvicorn.run("tilellm.__main__:app", host="0.0.0.0", port=8000, reload=True, log_level="info")#, log_config=args.log_path


if __name__ == "__main__":

    main()
