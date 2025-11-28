# Tiledesk LLM - API Documentation

Complete REST API documentation for Tiledesk LLM server.

**Base URL**: `http|https://<host>:<port>`

**Interactive Docs**:
- Swagger UI: `http|https://<host>:<port>`
- ReDoc: `http|https://<host>:<port>`

---

## Table of Contents

- [Scraping & Indexing APIs](#scraping--indexing-apis)
- [Question & Answer APIs](#question--answer-apis)
- [Namespace Management APIs](#namespace-management-apis)
- [Conversion APIs](#conversion-apis)
- [Tools Registry APIs](#tools-registry-apis)
- [Knowledge Graph APIs](#knowledge-graph-apis)
- [Authentication](#authentication)
- [Data Models](#data-models)

---

## Scraping & Indexing APIs

### POST `/api/scrape/single`
Indexes a single document.

**Request Body** (`ItemSingle`):
```json
{
  "id": "string",
  "source": "string",
  "content": "string",
  "namespace": "string",
  "webhook": "string (optional)",
  "hybrid": false,
  "semantic_chunk": false,
  "chunk_size": 1000,
  "chunk_overlap": 400,
  "embedding": "text-embedding-ada-002",
  "engine": {
    "name": "pinecone",
    "type": "pod!serverless",
    "apikey": "string",
    "vector_size": 1536,
    "index_name": "yourindexname"
  }
}
```

**Response**:
```json
{
  "message": "Item {id} created successfully"
}
```

**Status Codes**:
- `200`: Success
- `300`: Queued
- `400`: Error
---

### POST `/api/scrape/hybrid`
Indexes using hybrid search (vector + keyword).

**Request Body**: Same as `/api/scrape/single` with `"hybrid": true`

**Response**: Same as `/api/scrape/single`

**Hybrid Parameters**:
- `hybrid_batch_size`: Batch size for hybrid indexing (default: 10)
- `doc_batch_size`: Batch size for generate embeddings (default: 100)
- `sparse_encoder`: Encoder type - `"splade"` or `"bge-m3"` (default: `"splade"`)

---

### POST `/api/scrape/status`
Checks indexing operation status.

**Request Body** (`ScrapeStatusReq`):
```json
{
  "id": "string",
  "namespace": "string",
  "engine": { /* Engine object */ }
}
```

**Response** (`ScrapeStatusResponse`):
```json
{
  "status_message": "Indexing finish",
  "status_code": 3,
  "queue_order": -1
}
```

**Status Codes**:
- `0`: Document added to queue
- `2`: Indexing started
- `3`: Indexing finished
- `4`: Error

---

## Question & Answer APIs

### POST `/api/qa`
Query with conversation history on knowledge base.

**Request Body** (`QuestionAnswer`):
```json
{
  "question": "string",
  "namespace": "string",
  "gptkey": "string",
  "llm": "openai",
  "model": "gpt-4o",
  "temperature": 0.0,
  "top_k": 5,
  "max_tokens": 512,
  "embedding": "text-embedding-ada-002",
  "search_type": "similarity",
  "chunks_only": false,
  "citations": false,
  "chat_history_dict": {},
  "engine": { /* Engine object */ }
}
```

**Response** (`RetrievalResult` or `RetrievalChunksResult`):
```json
{
  "answer": "string",
  "success": true,
  "namespace": "string",
  "sources": ["source1", "source2"],
  "citations": [
    {
      "source_id": 1,
      "source_name": "https://example.com/doc"
    }
  ],
  "duration": 2.5
}
```

**Parameters**:
- `search_type`: `"similarity"` | `"hybrid"` | `"mmr"`
- `chunks_only`: If true, returns only chunks without generating answer
- `citations`: If true, includes source citations (requires `max_tokens >= 1024`)

---

### POST `/api/ask`
Direct LLM query with optional MCP servers and tools support.

**Request Body** (`QuestionToLLM`):
```json
{
  "question": "string",
  "llm_key": "string",
  "llm": "openai",
  "model": "gpt-4o",
  "temperature": 0.0,
  "max_tokens": 128,
  "system_context": "You are a helpful AI bot...",
  "servers": {
    "server-name": {
      "transport": "sse",
      "url": "https://example.com/mcp"
    }
  },
  "tools": ["tool1", "tool2"],
  "chat_history_dict": {}
}
```

**Response** (`SimpleAnswer`):
```json
{
  "answer": "string",
  "tools_log": [
    {
      "tool": "tool_name",
      "input": {},
      "output": "result"
    }
  ],
  "chat_history_dict": {},
  "prompt_token_info": {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150
  }
}
```

**MCP Server Configuration**:
- `transport`: `"sse"` | `"stdio"` | `"streamable_http"`
- For SSE & Streamable Http: provide `url` and optional `api_key`
- For stdio: provide `command` and `args`

**Routing Logic**:
1. No MCP servers → Simple LLM call
2. With MCP servers:
   - Simple string input → `ask_mcp_agent_llm_simple`
   - Complex/multimodal input → `ask_mcp_agent_llm`

---

### POST `/api/thinking`
LLM query with advanced reasoning for complex problems.

**Request Body**: Same as `/api/ask` with optional `thinking` parameter:
```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  }
}
```

**Response**: Same as `/api/ask`

---
## Namespace Management APIs

### GET `/api/list/namespace/{token}`
Lists all namespaces with vector counts.

**Path Parameters**:
- `token`: JWT token containing engine configuration

**Response** (`RepositoryNamespaceResult`):
```json
{
  "namespaces": [
    {
      "namespace": "my-namespace",
      "vector_count": 1500
    }
  ]
}
```

---

### GET `/api/desc/namespace/{namespace}/{token}`
Gets description of a specific namespace.

**Response** (`RepositoryDescNamespaceResult`):
```json
{
  "namespace_desc": {
    "namespace": "my-namespace",
    "vector_count": 1500
  },
  "ids": [
    {
      "metadata_id": "doc-001",
      "source": "https://example.com",
      "chunks_count": 10
    }
  ]
}
```

---

### GET `/api/listitems/namespace/{namespace}/{token}`
Gets all items in a namespace.

**Response** (`RepositoryItems`):
```json
{
  "matches": [
    {
      "id": "chunk-001",
      "metadata_id": "doc-001",
      "metadata_source": "https://example.com",
      "metadata_type": "pdf",
      "date": "2025-11-08 16:30:00",
      "text": "Content..."
    }
  ]
}
```

---

### GET `/api/id/{metadata_id}/namespace/{namespace}/{token}`
Retrieves all chunks of a specific document.

**Response**: Same as `/api/listitems/namespace`

---

### GET `/api/items`
Gets items filtered by source and namespace.

**Query Parameters**:
- `source`: Document URL or source
- `namespace`: Namespace ID
- `token`: JWT token

**Response**: Same as `/api/listitems/namespace`

---

### DELETE `/api/namespace/{namespace}/{token}`
Deletes an entire namespace.

**Response**:
```json
{
  "message": "Namespace {namespace} deleted"
}
```

---

### DELETE `/api/id/{metadata_id}/namespace/{namespace}/{token}`
Deletes all chunks of a specific document.

**Response**:
```json
{
  "message": "ids {metadata_id} in Namespace {namespace} deleted"
}
```

---

### DELETE `/api/chunk/{chunk_id}/namespace/{namespace}/{token}`
Deletes a single specific chunk.

**Response**:
```json
{
  "message": "ids {chunk_id} in Namespace {namespace} deleted"
}
```

---

## Conversion APIs

### POST `/api/convert`
Converts files between different formats.

**Request Body** (`ConversionRequest`):
```json
{
  "file_name": "document.pdf",
  "file_content": "base64_string or https://url.com/file.pdf",
  "conversion_type": "pdf_to_text"
}
```

**Conversion Types**:
- `xlsx_to_csv`: Converts each Excel sheet to separate CSV
- `pdf_to_text`: Extracts text from PDF
- `pdf_to_images`: Converts each PDF page to PNG image

**Response** (`List[ConvertedFile]`):
```json
[
  {
    "FileName": "output.txt",
    "FileExt": "txt",
    "FileSize": 1024,
    "File": "base64_encoded_content",
    "FileContent": "Plain text content..."
  }
]
```

---

## Tools Registry APIs

### GET `/api/tools`
Lists all available tools in the system.

**Response**:
```json
[
  {
    "name": "tool_name",
    "description": "Tool description",
    "parameters": {
      "param1": {
        "type": "string",
        "description": "Parameter description"
      }
    }
  }
]
```

---

## Authentication

Many endpoints require a JWT token containing engine configuration.

**Token Format**: Passed as path parameter `{token}` in protected endpoints.

**Token Contents**: Engine configuration including:
- Vector store credentials (Pinecone/Qdrant)
- Index name
- Vector dimensions

---

## Data Models

### Engine
Vector store configuration.

```python
{
  "name": "pinecone" | "qdrant",
  "type": "serverless" | "pod",  # Pinecone only
  "apikey": "string",
  "vector_size": 1536,
  "index_name": "index or collection name",
  "text_key": "text",
  "metric": "cosine",
  "host": "localhost",  # Qdrant only
  "port": 6333,  # Qdrant only
  "deployment": "local" | "cloud"  # Qdrant only
}
```

---

### LlmEmbeddingModel
Embedding model configuration.

```python
{
  "provider": "openai" | "huggingface" | "ollama" | "google" | "cohere" | "voyage" | "vllm",
  "name": "text-embedding-ada-002",
  "api_key": "string (optional)",
  "url": "string (optional)",
  "dimension": 1024
}
```

---

### ServerConfig (MCP)
MCP server configuration.

```python
{
  "transport": "sse" | "stdio" | "streamable_http",
  "url": "string (for SSE)",
  "command": "string (for stdio)",
  "args": ["string"] (for stdio),
  "api_key": "string (optional)",
  "parameters": {}
}
```

---

### ChatEntry
Conversation history entry.

```python
{
  "question": "string",
  "answer": "string"
}
```

---

### Citation
Source citation for answers.

```python
{
  "source_id": 1,
  "source_name": "https://example.com/doc"
}
```

---

### PromptTokenInfo
Token usage information.

```python
{
  "input_tokens": 100,
  "output_tokens": 50,
  "total_tokens": 150
}
```

---

## Configuration

### Environment Variables

**Required**:
- `REDIS_URL`: Redis connection URL

**Optional**:
- `ENABLE_PROFILER`: Enable performance profiling (`True`/`False`)
- `LOG_LEVEL`: Logging level (`DEBUG` | `INFO` | `WARNING` | `ERROR`)

## Error Handling

All APIs return standardized errors:

```json
{
  "detail": "Descriptive error message"
}
```

**Common Status Codes**:
- `200`: Success
- `201`: Resource created
- `204`: Success, no content
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `422`: Validation error
- `500`: Internal server error
- `503`: Service unavailable

---

## Usage Examples

### Example 1: Index a Document

```bash
curl -X POST "http://localhost:8000/api/scrape/single" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc-001",
    "source": "https://example.com/doc.pdf",
    "content": "Document content to index...",
    "namespace": "my-docs",
    "embedding": "text-embedding-ada-002",
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "your-api-key",
      "vector_size": 1536,
      "index_name": "tilellm"
    }
  }'
```

---

### Example 2: Query Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "namespace": "my-docs",
    "gptkey": "your-openai-key",
    "model": "gpt-4o",
    "embedding": "text-embedding-ada-002",
    "engine": { /* engine config */ }
  }'
```

---

### Example 3: Convert PDF to Text

```bash
curl -X POST "http://localhost:8000/api/convert" \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "document.pdf",
    "file_content": "https://example.com/document.pdf",
    "conversion_type": "pdf_to_text"
  }'
```

---

### Example 5: Query LLM with MCP Tools

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze this code repository",
    "llm_key": "your-openai-key",
    "llm": "openai",
    "model": "gpt-4o",
    "servers": {
      "github": {
        "transport": "sse",
        "url": "https://mcp-server.com/github"
      }
    },
    "tools": ["code-analyzer"]
  }'
```

---

## Support

**Repository**: https://github.com/Tiledesk/tiledesk-llm

**Last updated**: 2025-11-08
