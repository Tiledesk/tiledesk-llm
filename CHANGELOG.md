# **TILEDESK LLM - Changelog**

### **Authors**: 
    *Gianluca Lorenzo*
    *Andrea Sponziello* 
### **Copyrigth**: *Tiledesk SRL*


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