
# Install
```commandline
pip install .
```
or for development environment:
```commandline
pip install -e .
```

# Launch


```commandline
export REDIS_URL="redis://localhost:6379/0"
export PINECONE_TYPE="serverless|pod"
export PINECONE_API_KEY="pinecone api key"
export PINECONE_TEXT_KEY="pinecone field for text - default text in pod content"
export PINECONE_INDEX="pinecone index name"
export TILELLM_ROLE="role in pod. Train enable all the APIs, qa do not consume redis queue only Q&A"
export WORKERS=INT number of workers 2*CPU+1
export TIMEOUT=INT seconds of timeout default=180
export MAXREQUESTS=INT The maximum number of requests a worker will process before restarting. deafult=1200
export MAXRJITTER=INT The maximum jitter to add to the max_requests setting default=5
export GRACEFULTIMEOUT=INT Timeout for graceful workers restart default=30 
tilellm 
```

# Docker

```
sudo docker build -t tilellm .
```


```
sudo docker run -d -p 8000:8000 --env environment="dev|prod" \
--env PINECONE_API_KEY="yourapikey" \
--env PINECONE_TEXT_KEY="text|content" \
--env PINECONE_INDEX="index_name" \
--env TILELLM_ROLE="train|qa" \
--env WORKERS=3 \
--env TIMEOUT=180 \
--env MAXREQUESTS=1200 \
--env MAXRJITTER=5 \
--env GRACEFULTIMEOUT=30 \
--env REDIS_URL="redis://redis:6379/0" \
--name tilellm --link test-redis:redis tilellm

 
```

test-redis is a redis containar 

# Pinecone index type

## Serverless 
New version

```
pc.create_index(const.PINECONE_INDEX, 
                dimension=emb_dimension, 
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                                cloud="aws",
                                region="us-west-2")
                )
```


## Pod
old one

```
pc.create_index(const.PINECONE_INDEX, 
                dimension=emb_dimension, 
                metric='cosine', 
                spec=pinecone.PodSpec(
                                pod_type="p1",
                                pods=1,
                                environment="us-west4-gpc",
                                'replicas': 1,
                                'shards': 1,
                                'source_collection': ''
                               )
                )
```

## Models
Models for /api/ask

### OpenAI - engine: openai
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo
- got-4o

### Cohere - engine: cohere
- command-r
- command-r-plus

### Google - engine: google
- gemini-pro

### Anthropic - engine: anthropic
- claude-3-5-sonnet-20240620

### Groq - engine: groq
- Llama3-70b-8192
- Llama3-8b-8192
- Mixtral-8x7b-32768
- Gemma-7b-It