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
tilellm 
    --host      | default 0.0.0.0
    --port      | default=8000
    --redis_url | default="redis://localhost:6379/0"
    --env       | default="dev" | dev|prod
    --log_path  | default="log_conf.yaml"
```
create a .environ file with this variable:
```commandline
PINECONE_API_KEY="pinecone api key"
PINECONE_TEXT_KEY="pinecone field for text - default text in pod content"
PINECONE_INDEX = "pinecone index name"
```

# Docker

```
sudo docker build -t tilellm .
```


```
 sudo docker run -d -p 8000:8000 --env environment="dev|prod" --env PINECONE_API_KEY="yourapikey" --env PINECONE_TEXT_KEY="text|content" --env PINECONE_INDEX="index_name" --env REDIS_URL="redis://redis:6379/0" --name tilellm --link test-redis:redis tilellm

 
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
