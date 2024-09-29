
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
- got-4o-mini

### Cohere - engine: cohere
- command-r
- command-r-plus

### Google - engine: google
- gemini-pro

### Anthropic - engine: anthropic
- claude-3-5-sonnet-20240620

### Groq - engine: groq
- llama3-70b-8192
- llama3-8b-8192
- llama-3.1-8b-instant
- llama-3.1-70b-versatile
- Mixtral-8x7b-32768
- Gemma-7b-It

## Semantic chunk

```json
{
  
  "semantic_chunk": true,
  "breakpoint_threshold_type": "percentile"
  
  
}

```
### percentile
The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.

### standard_deviation
In this method, any difference greater than X standard deviations is split.

### interquartile
In this method, the interquartile distance is used to split chunks.

### gradient
In this method, the gradient of distance is used to split chunks along with the percentile method. This method is useful when chunks are highly correlated with each other or specific to a domain e.g. legal or medical. The idea is to apply anomaly detection on gradient array so that the distribution become wider and easy to identify boundaries in highly semantic data.


## Hybrid Search

### /api/scrape/single

```json
{
 ...
 "embedding":"huggingface",
  "hybrid":true,
  "sparse_encoder":"splade|bge-m3",
  ... 
  "engine":
   {
    "name": "",
    "type": "",
    "apikey" : "",
    "vector_size": 1024,
    "index_name": "" 
   }  
}
```

### /api/qa

```json
{
  "question": "question",
  "namespace": "",
  "debug":true,
  "citations":true,
  "llm": "anthropic|groq",
  "gptkey": "api-key of llm",
  "model": "es. claude-3-5-sonnet-20240620 | llama-3.1-70b-versatile",
  "temperature": 0.9,
  "max_tokens":2048,
  "embedding":"huggingfacce",
  "sparse_encoder":"splade|bge-m3",
  "search_type":"hybrid",
  "alpha": 0.2,
  "similarity_threshold":0.95,
  "system_context":"",
  "top_k": 6,
  "engine":
  {
    "name": "",
    "type": "",
    "apikey" : "",
    "vector_size": 1024,
    "index_name": "" 
  }
}
```
