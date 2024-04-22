# **TILEDESK LLM - Changelog**

### **Authors**: 
    *Gianluca Lorenzo*
    *Andrea Sponziello* 
### **Copyrigth**: *Tiledesk SRL*

## [2024-04-22]

### 0.1.6
- fixed: response of /api/delete/id

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