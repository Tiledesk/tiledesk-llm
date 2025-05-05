from redis import Redis, ResponseError

from tilellm.shared import const

   
async def redis_xgroup_create(redis_client: Redis):
    try:
        await redis_client.xgroup_create(
            const.STREAM_NAME, 
            const.STREAM_CONSUMER_GROUP, 
            id="0", 
            mkstream=True)
        
    except ResponseError as e:
        if "BUSYGROUP Consumer Group name already exists" not in str(e):
            raise
        
