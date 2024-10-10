
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
export JWT_SECRET_KEY="yourkey-256-bit"
export TOKENIZERS_PARALLELISM=false
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
sudo docker run -d -p 8000:8000 \
--env JWT_SECRET_KEY = "yourkey-256-bit"
--env TOKENIZERS_PARALLELISM=false
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
  "id": "content id",
  "source": "name or url of document",
  "type": "text|txt|url|pdf|docx",
  "content": "content of document",
  "hybrid": true,
  "sparse_encoder": "splade|bge-m3",
  "gptkey": "llm key; openai|anthropic|groq|cohere|gemini|ollama, ",
  "scrape_type": 0,
  "embedding": "name of embedding; huggingface|ollama|openai...|bge-m3",
  "model": {
    "name": "optional, used only with ollama",
    "url": "ollama base url",
    "dimension": 3072
  },
  "namespace": "vector store namespace",
  "webhook": "string",
  "semantic_chunk": false,
  "breakpoint_threshold_type": "percentile",
  "chunk_size": 1000,
  "chunk_overlap": 100,
  "parameters_scrape_type_4": {
    "unwanted_tags": [
      "string"
    ],
    "tags_to_extract": [
      "string"
    ],
    "unwanted_classnames": [
      "string"
    ],
    "desired_classnames": [
      "string"
    ],
    "remove_lines": true,
    "remove_comments": true,
    "time_sleep": 2
  },
  "engine": {
    "name": "pinecone",
    "type": "serverless",
    "apikey": "string",
    "vector_size": 1536,
    "index_name": "index name",
    "text_key": "text for serverless; content for pod",
    "metric": "cosine|dotproduct for hybrid"
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
  "embedding":"huggingface",
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
