# **TILEDESK LLM - Changelog**

### **Authors**: 
    *Gianluca Lorenzo*
    *Andrea Sponziello* 
### **Copyrigth**: *Tiledesk SRL*

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