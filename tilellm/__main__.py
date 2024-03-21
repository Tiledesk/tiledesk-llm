from contextlib import asynccontextmanager
from fastapi import (FastAPI, 
  Depends, HTTPException)
from fastapi.responses import JSONResponse

import argparse

import redis
import aioredis
import asyncio
from aioredis import from_url
import aiohttp

from tilellm.models.item_model import ItemSingle, QuestionAnswer
from tilellm.store.redis_repository import redis_xgroup_create
from tilellm.controller.openai_controller import (ask_with_memory,
                                                  add_pc_item,
                                                  delete_namespace,
                                                  delete_id_from_namespace,
                                                  get_ids_namespace)

import logging



parser = argparse.ArgumentParser(description="Tiledesk: llms integration")
parser.add_argument("--host", default="localhost", help="Hostname for FastAPI")
parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI")
parser.add_argument("--redis_url", default="redis://localhost:6379/0", help="Redis url. Default redis://localhost:6379/0")
parser.add_argument("--log_path", default="log_conf.yaml", help="Log configuration file path. Default log_conf.yaml")
args = parser.parse_args()


logger = logging.getLogger(__name__)

async def get_redis_client():
    try:
        redis_client= await from_url(args.redis_url)
        yield redis_client
    finally:
        await redis_client.close()

async def reader(channel: aioredis.client.Redis):  
    
    from tilellm.shared import const  
    webhook = ""
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
                for message in message_data:
                    message_id, message_values = message
                    #print(f"Received message {message_id}: {message_values}")
                    import ast
                     
                    byte_str = message_values[b"single"]
                    dict_str = byte_str.decode("UTF-8")
                    item = ast.literal_eval(dict_str)
                    itemSingle = ItemSingle(**item)
                    webhook = item.get('webhook',"")

                    pc_result = add_pc_item(itemSingle)
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")

                    pc_result["date"]=current_time

                    # A POST request to the API
                    logger.info(f"webhook {webhook}")  
                    if webhook:     
                        async with aiohttp.ClientSession() as session:
                            response = await session.post(webhook,  json = pc_result,  headers={"Content-Type": "application/json"})
                           
                            
                    await channel.xack(
                        const.STREAM_NAME, 
                        const.STREAM_CONSUMER_GROUP, 
                        message_id)    
                                 

            
        except Exception as e:
            import traceback 
            if webhook:     
                async with aiohttp.ClientSession() as session:
                    response = await session.post(webhook,  json = repr(e),  headers={"Content-Type": "application/json"})
                    
                print(f"ERRORE {e}, webhook: {webhook}")
            traceback.print_exc() 
            logger.error(e)
           

@asynccontextmanager
async def redis_consumer(app: FastAPI):
    
    redis_client= from_url(args.redis_url)
    
    await redis_xgroup_create(redis_client)
 
    asyncio.create_task(reader(redis_client)) 
 
    yield 
    
    await redis_client.close()   



app = FastAPI(lifespan=redis_consumer)



@app.post("/api/scrape/single")
async def create_scrape_item_main(item: ItemSingle, redis_client: aioredis.client.Redis = Depends(get_redis_client)):
    from tilellm.shared import const
    logger.debug(item) 
    res = await redis_client.xadd(const.STREAM_NAME, {"single":item.model_dump_json()} , id="*")
    logger.debug(res)

    return {"message": f"Item {item.id} created successfully, more {res}"}

@app.post("/api/qa")
async def post_ask_with_memory_main(question_answer:QuestionAnswer ):
    logger.debug(question_answer) 
    result = ask_with_memory(question_answer)
    logger.debug(result)
    return JSONResponse(content=result)
    #return result

@app.post("/api/list/namespace")
async def list_namespace_items_main(namespace:str ):
    return {"message":"not implemented yet"}


@app.delete("/api/namespace/{namespace}")
async def delete_namespace_main(namespace:str ):
    try:
        result = delete_namespace(namespace)
        return JSONResponse(content={"message":f"Namespace {namespace} deleted"})
    except Exception as ex:
        import json
        #from pinecone.core.client.exceptions import NotFoundException
        #a = NotFoundException()
        #a.body
        print(ex.body)
        raise HTTPException(status_code=ex.status, detail=json.loads(ex.body) )

@app.delete("/api/id/{id}/namespace/{namespace}")
async def delete_item_id_namespace_main(id:str, namespace:str ):
    try:
        logger.info(f"cancellazione id {id} dal namespace {namespace}")
        result = delete_id_from_namespace(id,namespace)

        return JSONResponse(content={"message":f"ids {id} in Namespace {namespace} deleted"})
    except Exception as ex:
        import json
        #from pinecone.core.client.exceptions import NotFoundException
        #a = NotFoundException()
        #a.body
        logger.error(ex.body)
        raise HTTPException(status_code=ex.status, detail=json.loads(ex.body) )

@app.get("/api/id/{id}/namespace/{namespace}")
async def get_items_id_namespace_main(id:str, namespace:str ):
    try:
        logger.info(f"retrieve id {id} dal namespace {namespace}")
        result = get_ids_namespace(id,namespace)

        return JSONResponse(content=result.model_dump())
    except Exception as ex:
        import json
        #from pinecone.core.client.exceptions import NotFoundException
        #a = NotFoundException()
        #a.body
        logger.error(ex.body)
        raise HTTPException(status_code=ex.status, detail=json.loads(ex.body) )


def main(): 
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv('.environ')
    
    uvicorn.run("tilellm.__main__:app", host=args.host, port=args.port, log_config=args.log_path, reload=True)#log_config=args.log_path,

if __name__ == "__main__":
   main()
