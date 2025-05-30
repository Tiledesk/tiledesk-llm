# **TILEDESK LLM - Changelog**

### **Authors**: 
    *Gianluca Lorenzo*
    *Andrea Sponziello* 
### **Copyrigth**: *Tiledesk SRL*

## [2025-05-16]
### 0.4.3-rc2
- upgrade torch library

## [2025-05-16]
### 0.4.3-rc1
- add: /api/qa chunks parameters
- add: support to qdrant vector store

## [2025-05-16]
### 0.4.2-rc4
- fix: /api/qa

## [2025-05-16]
### 0.4.2-rc3
- fix: async connection

## [2025-05-16]
### 0.4.2-rc2
- fix: scrape error

## [2025-05-07]
### 0.4.2-rc1
- upgrade: python version to 3.12
- upgrade: langchain version to 0.3.25
- add: stream support for /api/ask, /api/thinking and /api/qa

## [2025-03-08]
### 0.4.1-rc2
- add: stream support for /api/thinking


## [2025-02-16]
### 0.4.1-rc1
- fix: parse of claude-3.7 response if thinking is enabled

## [2025-02-16]
### 0.4.0-rc3
- add: /api/thinking for o1 and claude-3.7

## [2025-02-16]
### 0.4.0-rc2
- minor fix

## [2025-02-16]
### 0.4.0-rc1
- add stream support

## [2024-10-10]
### 0.3.2-rc2
- fix: /api/id/{id}/namespace/{namespace}/{token}
- add sentence embedding with bge-m3
- add: hybrid search with bg3-m3
- modify: deleted env variable for vector store 

## [2024-09-23]
### 0.3.0
- add: hybrid search
- add: indexing based on spade 
- minor fix


## [2024-09-17]
### 0.2.20
- upgrade: worker to 0.0.27

## [2024-09-14]
### 0.2.19
- upgrade: worker to 0.0.25
- 
## [2024-09-14]
### 0.2.18
- upgrade: worker
- modify: default value for scrape type: 4

## [2024-09-05]
### 0.2.17
- fix: nltk download on Dockerfile   

## [2024-09-04]
### 0.2.16
- fix: max_tokens=1024 if citations=True  

## [2024-09-04]
### 0.2.15
- fix: citations without quote

## [2024-09-04]
### 0.2.14
- modify: citations without quote
- 
## [2024-09-04]
### 0.2.13
- modify: source on qa

## [2024-08-31]
### 0.2.12
- add: citations

## [2024-07-31]
### 0.2.11
- fix: log

## [2024-07-31]
### 0.2.10
- fix: write log
- updated: version of libs


## [2024-07-29]
### 0.2.9
- add: n_messages on /api/ask to set the maximum number of messages to include 

## [2024-07-27]
### 0.2.8
- add: history on /api/ask 


## [2024-07-26]
### 0.2.7
- add: scrape_type=3|4 
- add: to /api/qa "similarity_threshold" 

## [2024-07-09]
### 0.2.6
- add: DELETE /api/chunk/<chunk_id>/namespace/<namespace>
- add: search_type parameter similarity|mmr 

## [2024-07-01]
### 0.2.5
- fix: user-agent for scrape

## [2024-07-01]
### 0.2.4
- fix: scrape_type=0
- added: /api/ask to ask to llm 


## [2024-06-21]
### 0.2.3
- fix: delete chunks from namespace by metadata id
- added: /api/desc/namespace/{ns} for namespace description 

## [2024-06-15]
### 0.2.2
- fix: indexing of txt documents



## [2024-06-15]
### 0.2.1
- update: langchain v. 0.1.16
- modified: prompt for q&A

## [2024-06-08]
### 0.2.0
- refactor: refactor repository in order to manage pod and serverless

## [2024-06-07]
### 0.1.21
- added: support for pdf, docx and txt

## [2024-06-06]
### 0.1.20
- added: log_conf.json

## [2024-06-06]

### 0.1.19
- minor fix: return 400 if url is not correct

## [2024-05-20]

### 0.1.18
- added: scrape_type =0|1
- added: trainer_worker as a node application

## [2024-05-20]

### 0.1.17
- added: PIENCONE_TYPE = "serverless|pod"

## [2024-05-18]

### 0.1.16
- added: /api/scrape/single without redis queue
- added: /api/scrape/enqueue to enqueue item into redis queue 

## [2024-05-14]

### 0.1.15
- minor fix: Dockerfile


## [2024-05-07]

### 0.1.14
- added parameter to entrypoint.sh

## [2024-05-06]

### 0.1.13
- fixed: delete ids from namespace. top_k max 10k

## [2024-05-03]

### 0.1.12
- fixed: send status 200 to webhook and id for status 400
- added: DELETE of namespace by POST
- fixed: /api/scrape/status check not only on redis but on Pinecone too

## [2024-05-02]

### 0.1.11
- fixed: log_conf.json 

## [2024-05-02]

### 0.1.10
- fixed: any fields of metadata cannot be None.
- added: TILELLM_ROLE=qa|train in order to manage qa and train 

## [2024-05-01]

### 0.1.9
- modified: log_conf.json to INFO Level

## [2024-04-30]

### 0.1.8
- added: log_conf.json to Dockerfile

## [2024-04-24]

### 0.1.7
- fixed: logging

## [2024-04-22]

### 0.1.6
- fixed: json response of /api/delete/id

## [2024-04-20]

### 0.1.5
- added: list items by namespace with GET /api/listitems/namespace/{namespace}
- added: list all namespaces with POST /api/list/namespace
- fixed some async function 

## [2024-04-19]

### 0.1.4
- update webhook with status=300|400

### 0.1.3
- added: delete chunk by id/namespace with POST method
- added: expiration time to redis cache