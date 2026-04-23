# Tiledesk LLM - API Documentation

Complete REST API documentation for Tiledesk LLM server.

**Base URL**: `http|https://<host>:<port>`

**Interactive Docs**:
- Swagger UI: `http|https://<host>:<port>`
- ReDoc: `http|https://<host>:<port>`

---

## Table of Contents

- [Unified Ingestion API](#unified-ingestion-api)
- [Scraping & Indexing APIs](#scraping--indexing-apis)
- [Question & Answer APIs](#question--answer-apis)
- [Namespace Management APIs](#namespace-management-apis)
- [Conversion APIs](#conversion-apis)
- [Tag Filtering](#tag-filtering)
- [Tools Registry APIs](#tools-registry-apis)
- [Knowledge Graph APIs (Neo4j)](#knowledge-graph-apis)
- [Knowledge Graph APIs (FalkorDB)](#knowledge-graph-apis-falkordb)
- [Authentication](#authentication)
- [Data Models](#data-models)

---

## Unified Ingestion API

### POST `/api/ingestion`

Single entry point for all document ingestion. Automatically routes to the appropriate pipeline based on document type and configuration. The old endpoints (`/api/scrape/single`, `/api/pdf/scrape`) remain active for backward compatibility.

**Type auto-detection**

When `type` is omitted or set to `"auto"`, the system resolves the document type automatically using a 6-level priority chain (no network calls, no LLM):

1. **Magic bytes** — reads the first bytes of `file_content` (base64): `%PDF` → `pdf`; OLE2 header → `xls`; ZIP header (DOCX/XLSX ambiguous) → falls through to extension.
2. **Extension from `file_name`** — e.g. `report.docx` → `docx`.
3. **Extension from `source` URL** — e.g. `https://example.com/data.xlsx` → `xlsx` (query string/fragment stripped).
4. **URL heuristic** — `http(s)` source with no known file extension (`.html`, `.htm`, empty path) → `url`.
5. **Direct text** — no source/file, but `content` field is populated → `text`.
6. **No signal** — leaves type unresolved; repository direct-text branch handles it.

**Routing logic (after type resolution):**

| Condition | Pipeline |
|-----------|----------|
| `type = "pdf"` AND `use_ocr = true` | Advanced OCR pipeline (Docling) |
| `type = "docx"` AND `use_ocr = true` | DOCX image-aware pipeline |
| `hybrid = true` | Hybrid ingestion (`add_item_hybrid`) |
| default | Standard ingestion (`add_item`) |

**Request Body** (`ItemSingle`):
```json
{
  "id": "doc-001",
  "source": "https://example.com/document.pdf",
  "type": "auto",
  "namespace": "my-docs",
  "gptkey": "sk-...",
  "embedding": "text-embedding-ada-002",
  "engine": {
    "name": "pinecone",
    "type": "serverless",
    "apikey": "your-pinecone-key",
    "index_name": "tilellm",
    "vector_size": 1536
  },
  "use_ocr": false,
  "hybrid": false,
  "situated_context": {
    "enable": true,
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-..."
  },
  "tags": ["finance", "2024"],
  "chunk_size": 1000,
  "chunk_overlap": 400
}
```

**OCR-specific fields** (only when `use_ocr = true`, passed through to the Docling pipeline):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_ocr` | bool | `false` | Route to Docling OCR pipeline |
| `situated_context` | object | `null` | Configuration for Contextual Retrieval (situated context). Prepends LLM-generated context to each chunk. |
| `situated_context.enable` | bool | `false` | Whether to enable situated context. |
| `situated_context.provider` | string | `openai` | LLM provider (`openai`\|`anthropic`\|`google`\|`groq`\|`vllm`\|`ollama`). |
| `situated_context.model` | string | `gpt-4o-mini` | LLM model for context generation. |
| `situated_context.api_key` | string | `null` | API key for the provider. |

**Supported document types** (via `type` field):

| `type` | Source | Loader / Path | Notes |
|--------|--------|---------------|-------|
| `auto` | any | Auto-detected (see above) | Omit `type` or pass `"auto"` |
| `url` | `source` URL | Trafilatura → Playwright fallback | 6 scraping strategies (`scrape_type` 0–5) |
| `pdf` | `source` URL | PyPDFLoader (basic) / Docling (`use_ocr=true`) | Use `use_ocr=true` for tables, images, formulas |
| `docx` | `source` URL | `StructuredDocxLoader` / image pipeline (`use_ocr=true`) | Preserves heading hierarchy and tables |
| `xlsx` / `xls` | `source` URL | `ExcelLoader` | Sheet-by-sheet, vertical chunking >100 rows |
| `csv` | `source` URL | `CSVLoader` | Vertical chunking >100 rows |
| `txt` | `source` URL | `TextLoader` | Plain text fetched from URL |
| `md` | `source` URL | `UnstructuredMarkdownLoader` | Headings partially parsed |
| `text` | `content` field | `process_auto_detected_text` | **Direct inline content** — no source URL needed. Sub-format (plain / Markdown / tabular) detected at chunk time. |
| `regex_custom` | `source` URL | Playwright + user regex | Custom split pattern via `chunk_regex` |

**Response**: same as `/api/scrape/single`
```json
{ "message": "Item doc-001 created successfully" }
```

**Status Codes**:
- `200`: Processed synchronously
- `300`: Queued (if `webhook` is provided)
- `400`: Validation error (e.g., `use_ocr=true` but missing `source`)

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
  "tags": ["string"] (optional),
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
- `hybrid_batch_size`: Batch size for hybrid indexing (default: 10). For **local** sparse encoders (`splade`, `bge-m3`) this is treated as a hard cap on items per batch; the effective batch size is additionally bounded by a token-aware budget (`batch_size * max_seq_len ≤ token_budget_per_batch`, default 2048). See README → [Local Inference Performance](../README.md#local-inference-performance-crossencoder--sparse-encoders) for the full list of tunables. TEI sparse encoder and TEI reranker are unaffected — they use the fixed value.
- `doc_batch_size`: Batch size for generate embeddings (default: 100)
- `sparse_encoder`: Encoder type - `"splade"` or `"bge-m3"` (default: `"splade"`)

**Note on local GPU inference**: When `sparse_encoder` is `"splade"` or `"bge-m3"` and no TEI endpoint is configured (`TEI_SPARSE_ENCODER_URL` unset), the encoder is loaded and run locally. Local encoders automatically apply fp16 (Splade), warmup on load, a process-wide GPU lock shared with the reranker, token-budget batching, and CUDA OOM recovery with budget halving. These behaviors are not per-request — they are class-level defaults in `tilellm/tools/sparse_encoders.py`.

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
  "tags": ["string"] (optional),
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
  "use_hyde": false,
  "use_raptor": false,
  "use_cache": false,
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
- `use_hyde`: If true, generates a hypothetical document before retrieval to improve embedding quality
- `use_raptor`: If true, uses RAPTOR hierarchical tree retrieval instead of standard RAG
- `use_cache`: If true, enables semantic cache (L1 exact match + L2 cosine similarity). Default: `false`. On hit, returns cached response without calling the vector store or LLM. Cache is automatically invalidated when documents are added or the namespace is deleted.
- `tags`: Optional tag filter. Can be a list of strings (`["python", "api"]`) or a boolean expression (`"(python|javascript)&!legacy"`). See [Tag Filtering](#tag-filtering) for details.
- `reranking`: Enable reranking (bool or object, see README → Reranker). When using a **local** cross-encoder (no `provider` field), the model is loaded on GPU with fp16, warmup, process-wide GPU lock, token-budget batching, and CUDA OOM recovery. These optimizations are automatic and do not require any request-level configuration. Tunables (`token_budget_per_batch`, `max_pairs_per_batch`, `max_oom_retries`, `_use_fp16`, `_warmup_on_load`) are class-level defaults in `tilellm/tools/reranker.py::TileReranker` — see README → [Local Inference Performance](../README.md#local-inference-performance-crossencoder--sparse-encoders).

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
- For SSE & Streamable Http: provide `url` and optional `headers`
- For stdio: provide `command` and `args`
- `headers`: optional HTTP headers sent on every request (e.g. API keys, user identity)
- `enabled_tools`: list of tool names to expose, or `["all"]` (default)

**Routing Logic**:
1. No MCP servers → Simple LLM call
2. With MCP servers:
   - Simple string input → `ask_mcp_agent_llm_simple`
   - Complex/multimodal input → `ask_mcp_agent_llm`

---

### POST `/api/thinking`
LLM query with advanced reasoning for complex problems. Supports GPT-5, Claude 4/4.5, Gemini 2.5/3.0, and DeepSeek.

**Request Body**: Same as `/api/ask` with `thinking` configuration object (`ReasoningConfig`):

**Common Parameters**:
- `show_thinking_stream` (boolean, default: `true`): Show thinking content in stream. If `false`, thinking is included only in final response.

**Provider-Specific Parameters**:

**OpenAI GPT-5**:
```json
{
  "question": "Your question",
  "llm": "openai",
  "model": "gpt-5-nano",
  "llm_key": "sk-...",
  "stream": true,
  "thinking": {
    "show_thinking_stream": true,
    "reasoning_effort": "high",
    "reasoning_summary": "auto"
  }
}
```
- `reasoning_effort`: `"low"` | `"medium"` | `"high"`
- `reasoning_summary`: `"auto"` | `"always"` | `"never"`

**Anthropic Claude 4/4.5**:
```json
{
  "question": "Your question",
  "llm": "anthropic",
  "model": "claude-sonnet-4.5-20250514",
  "llm_key": "sk-ant-...",
  "stream": true,
  "thinking": {
    "show_thinking_stream": true,
    "type": "enabled",
    "budget_tokens": 10000
  }
}
```
- `type`: `"enabled"` | `"disabled"`
- `budget_tokens`: 0-100000

**Google Gemini 2.5 Pro**:
```json
{
  "question": "Your question",
  "llm": "google",
  "model": "gemini-2.5-pro",
  "llm_key": "...",
  "stream": true,
  "thinking": {
    "show_thinking_stream": true,
    "thinkingBudget": -1
  }
}
```
- `thinkingBudget`: `-1` (dynamic), `0` (disabled), or positive number ≤ 32000

**Google Gemini 3.0 Pro**:
```json
{
  "question": "Your question",
  "llm": "google",
  "model": "gemini-3.0-pro",
  "llm_key": "...",
  "stream": true,
  "thinking": {
    "show_thinking_stream": true,
    "thinkingLevel": "high"
  }
}
```
- `thinkingLevel`: `"low"` | `"medium"` | `"high"`

**DeepSeek Reasoner**:
```json
{
  "question": "Your question",
  "llm": "deepseek",
  "model": "deepseek-reasoner",
  "llm_key": "...",
  "stream": true,
  "thinking": {
    "show_thinking_stream": true
  }
}
```
No specific parameters needed - reasoning is automatic.

**Response** (`ReasoningAnswer`):
```json
{
  "answer": "The final answer",
  "reasoning_content": "Complete reasoning process",
  "chat_history_dict": {...},
  "prompt_token_info": {
    "input_tokens": 100,
    "output_tokens": 500,
    "total_tokens": 600
  }
}
```

**Streaming Response** (SSE):
- `event: metadata` - Start/end metadata
- `event: chunk` - Content chunks with `reasoning_content` or `content`
- `event: done` - Final complete response

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
### GET `/api/listcompleteitems/namespace/{namespace}/all`
Gets all items in a namespace with full text content. Uses Bearer token authentication (Authorization header).

**Path Parameters**:
- `namespace`: Namespace ID

**Authentication**: Bearer token (via `Authorization: Bearer <token>` header)

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

## Knowledge Graph APIs (Neo4j)

The Knowledge Graph module provides GraphRAG (Graph-based Retrieval Augmented Generation) capabilities using Neo4j and MinIO.

### Utility Endpoints

#### `GET /api/kg/health`
Check Neo4j connection health.

**Response**:
```json
{
  "status": "healthy",
  "neo4j_version": "5.x.x",
  "database": "neo4j"
}
```

#### `GET /api/kg/stats`
Get database statistics.

**Response**:
```json
{
  "node_count": 1500,
  "relationship_count": 3000,
  "labels": ["Document", "Person", "Organization"],
  "relationship_types": ["REFERENCES", "RELATES_TO"]
}
```

### Node Management

#### `POST /api/kg/nodes`
Create a new node in the knowledge graph.

**Request Body** (`Node`):
```json
{
  "label": "Document",
  "properties": {
    "title": "Introduction to RAG",
    "content": "RAG stands for Retrieval Augmented Generation...",
    "embedding": [0.1, 0.2, 0.3]
  }
}
```

**Response**: Same as request with generated `id` field.

#### `GET /api/kg/nodes/{node_id}`
Retrieve a node by ID.

#### `GET /api/kg/nodes?label={label}&limit={limit}`
List nodes by label.

#### `DELETE /api/kg/nodes/{node_id}?detach={true|false}`
Delete a node (optionally delete connected relationships).

### Relationship Management

#### `POST /api/kg/relationships`
Create a relationship between two nodes.

**Request Body** (`Relationship`):
```json
{
  "source_id": "node-123",
  "target_id": "node-456",
  "type": "REFERENCES",
  "properties": {
    "weight": 0.8,
    "context": "citation"
  }
}
```

#### `GET /api/kg/nodes/{node_id}/relationships?direction={incoming|outgoing|both}`
Get all relationships connected to a node.

### Graph Operations

#### `POST /api/kg/create`
Create/import a knowledge graph from documents in a vector store namespace. It retrieves chunks from the vector store and uses an LLM to extract entities and relationships (GraphRAG).

**Request Body** (`GraphCreateRequest`):
```json
{
  "namespace": "my-documents",
  "index_name": "tilellm",
  "engine": {
    "name": "pinecone",
    "type": "serverless",
    "apikey": "your-api-key",
    "vector_size": 1536,
    "index_name": "tilellm"
  },
  "limit": 100,
  "overwrite": false
}
```

**Response**:
```json
{
  "namespace": "my-documents",
  "chunks_processed": 100,
  "nodes_created": 150,
  "relationships_created": 300,
  "status": "success"
}
```

#### `POST /api/kg/hierarchical`
Perform Hierarchical Clustering (Levels 0, 1, 2) using the Leiden algorithm. Generates community reports at multiple levels of granularity, enabling global search.

**Request Body** (`GraphClusterRequest`):
```json
{
  "namespace": "my-documents",
  "engine": { /* engine config */ },
  "overwrite": true
}
```

**Response** (`GraphClusterResponse`):
```json
{
  "status": "success",
  "communities_detected": 15,
  "reports_created": 15,
  "message": "Hierarchical clustering completed"
}
```

#### `POST /api/kg/louvain-cluster`
Perform Louvain clustering and generate community reports.

**Request Body**: Same as `/hierarchical`

#### `POST /api/kg/leiden-cluster`
Perform Leiden clustering (flat) and generate community reports.

**Request Body**: Same as `/hierarchical`

#### `POST /api/kg/add-document`
Add a single document to an existing knowledge graph and update community reports.

**Request Body** (`AddDocumentRequest`):
```json
{
  "metadata_id": "doc_12345_uuid",
  "namespace": "my-documents",
  "engine": {
    "name": "pinecone",
    "type": "serverless",
    "apikey": "your-api-key",
    "index_name": "tilellm"
  },
  "deduplicate_entities": true,
  "sparse_encoder": "splade",
  "llm_key": "my-llm-key",
  "model": "gpt-4"
}
```

**Response** (`AddDocumentResponse`):
```json
{
  "metadata_id": "doc_12345_uuid",
  "chunks_processed": 15,
  "entities_extracted": 25,
  "entities_new": 20,
  "entities_reused": 5,
  "relationships_created": 30,
  "status": "success",
  "community_reports_updated": true,
  "report_stats": {
    "communities_detected": 12,
    "reports_created": 36,
    "status": "success"
  }
}
```

**Note**: This endpoint performs incremental graph updates by:
1. Retrieving all chunks of the document from vector store using `metadata_id`
2. Extracting entities and relationships using GraphRAG
3. Adding new nodes and relationships to Neo4j (with optional deduplication)
4. **Automatically regenerating community reports** to keep summaries up-to-date

### Search & QA Endpoints

#### `POST /api/kg/qa`
**METHOD 1: Community/Global Search**
Performs global search ONLY on community reports. Efficient for high-level questions ("What are the main themes?").

**Request Body** (`GraphQARequest`):
```json
{
  "question": "Summarize the main topics",
  "namespace": "my-documents",
  "engine": { /* engine config */ },
  "max_results": 10
}
```

**Response** (`CommunityQAResponse`):
```json
{
  "answer": "The main topics are...",
  "reports_used": 5,
  "chat_history_dict": null
}
```

#### `POST /api/kg/hybrid`
**METHOD 2: Integrated Hybrid Search**
Unified pipeline: Global Search (Community Reports) + Parallel Retrieval (Vector + Keyword from Vector Store) + RRF + Graph Expansion (from Neo4j) + Reranking.
Best for complex queries requiring specific details and broader context.

**Request Body** (`GraphQAAdvancedRequest`):
```json
{
  "question": "What is the relationship between AI and machine learning?",
  "namespace": "my-documents",
  "engine": { /* engine config */ },
  "retrieval_strategy": "integrated_hybrid",
  "top_k": 10,
  "vector_weight": 1.0,
  "keyword_weight": 1.0,
  "graph_weight": 1.0,
  "query_type": "exploratory"
}
```

**Response** (`GraphQAAdvancedResponse`):
```json
{
  "answer": "AI is a broader field that encompasses machine learning...",
  "entities": [
    {
      "id": "node-123",
      "label": "Concept",
      "properties": {"name": "Artificial Intelligence"}
    }
  ],
  "relationships": [
    {
      "id": "rel-456",
      "type": "CONTAINS",
      "source_id": "node-123",
      "target_id": "node-789"
    }
  ],
  "retrieval_strategy": "integrated_hybrid",
  "scores": {
    "vector_weight": 1.0,
    "graph_weight": 1.0
  },
  "expanded_nodes": [],
  "expanded_relationships": []
}
```

#### `GET /api/kg/network`
Get the graph network (nodes + relationships) for visualization.

**Query Parameters**:
- `namespace`: Filter by namespace
- `index_name`: Filter by index name
- `node_limit`: Max nodes (default 1000)
- `relationship_limit`: Max relationships (default 5000)
- `node_labels`: Filter by labels (e.g. `PERSON`)
- `community`: If `true`, returns the community graph (`BELONGS_TO_COMMUNITY` relationships) instead of entity graph.

---

## Tag Filtering

Tag filtering allows you to filter documents by tags during indexing and querying. Tags can be assigned when indexing documents and used as filters during retrieval using boolean expressions or simple lists.

### Tag Grammar

- Single tag: `"python"`
- OR operator: `"python|api"` (matches documents tagged with either "python" OR "api")
- AND operator: `"python&api"` (matches documents tagged with both "python" AND "api")
- NOT operator: `"!legacy"` (excludes documents tagged with "legacy")
- Parentheses: `"(python|javascript)&(api|rest)&!legacy"` (complex nested expressions)
- List syntax: `["python", "api"]` (equivalent to `"python&api"`)

### Usage Examples

**Indexing with tags** (in `/api/scrape/single` or `/api/scrape/hybrid`):
```json
{
  "id": "doc-001",
  "source": "https://example.com",
  "content": "Document content...",
  "namespace": "my-docs",
  "tags": ["python", "api", "latest"]
}
```

**Querying with tag filter** (in `/api/qa`):
```json
{
  "question": "How to use the API?",
  "namespace": "my-docs",
  "tags": "(python|javascript)&api&!legacy"
}
```

**Supported Vector Stores**: Pinecone (Serverless and Pod) and Qdrant fully support tag filtering. Redis vector store is not affected (used only for caching/streaming).

**Implementation Details**: Tags are stored in vector store metadata under the `"tags"` field. Filter conversion happens automatically for different vector stores (Pinecone native filters, Qdrant via `build_filter()`). Works with all search types (`similarity`, `hybrid`, `mmr`) and reranking.

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

## Knowledge Graph APIs (FalkorDB)

The Knowledge Graph module provides advanced graph-based retrieval using FalkorDB (Redis-based graph database). All endpoints are prefixed with `/api/kg-falkor/`.

### Utility Endpoints
- `GET /api/kg-falkor/health` - Check FalkorDB connection health
- `GET /api/kg-falkor/stats` - Get database statistics (node count, relationship count, etc.)
- `GET /api/kg-falkor/tasks/{task_id}` - Check status of an asynchronous task

### Node Management
- `POST /api/kg-falkor/nodes` - Create a new node
- `GET /api/kg-falkor/nodes/{node_id}` - Read node by ID
- `GET /api/kg-falkor/nodes` - List nodes (optional filters: label, namespace, index_name)
- `GET /api/kg-falkor/nodes/search` - Search nodes by text across searchable properties
- `PUT /api/kg-falkor/nodes/{node_id}` - Update node
- `PATCH /api/kg-falkor/nodes/{node_id}` - Partially update node
- `DELETE /api/kg-falkor/nodes/{node_id}` - Delete node

### Relationship Management
- `POST /api/kg-falkor/relationships` - Create relationship between nodes
- `GET /api/kg-falkor/relationships/{relationship_id}` - Read relationship by ID
- `GET /api/kg-falkor/nodes/{node_id}/relationships` - List relationships for a node (direction: incoming/outgoing/both)
- `PUT /api/kg-falkor/relationships/{relationship_id}` - Update relationship
- `PATCH /api/kg-falkor/relationships/{relationship_id}` - Partially update relationship
- `DELETE /api/kg-falkor/relationships/{relationship_id}` - Delete relationship

### Graph Operations
- `POST /api/kg-falkor/create` - Create/import knowledge graph from vector store namespace (async)
- `POST /api/kg-falkor/add-document` - Add a single document to existing knowledge graph and update community reports (async)
- `POST /api/kg-falkor/louvain-cluster` - Perform Louvain clustering with MinIO storage (async)
- `POST /api/kg-falkor/leiden-cluster` - Perform Leiden clustering (async)
- `POST /api/kg-falkor/hierarchical` - Perform Hierarchical Clustering (async)
- `POST /api/kg-falkor/community-analysis` - Perform community analysis (async)

### Search & QA
- `POST /api/kg-falkor/qa` - Community/Global search on community reports
- `POST /api/kg-falkor/hybrid` - Integrated hybrid search (Global + Local + Graph expansion)
- `POST /api/kg-falkor/advancedqa` - Advanced QA with context fusion, reranking, and adaptive expansion
- `POST /api/kg-falkor/agenticqa` - Agentic QA with iterative reasoning and tool usage
- `GET /api/kg-falkor/network` - Get network visualization data (nodes and relationships)
- `POST /api/kg-falkor/multimodal-search` - Multimodal search combining text and image embeddings

For detailed request/response schemas and examples, refer to the [Knowledge Graph FalkorDB README](tilellm/modules/knowledge_graph_falkor/README.md).

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

```json
{
  "transport": "sse | stdio | streamable_http",
  "url": "string (required for SSE and streamable_http)",
  "command": "string (required for stdio)",
  "args": ["string"],
  "headers": {"header-name": "value"},
  "enabled_tools": ["all"],
  "parameters": {}
}
```

**`headers`** — arbitrary HTTP headers forwarded on every MCP request. Use this for:
- Provider API keys (`"x-api-key": "<key>"`)
- User-scoped identity (see Composio example below)

#### Composio integration

Composio requires the **user_id as a query parameter** in the URL and the **API key as an HTTP header**:

```json
"servers": {
  "composio": {
    "transport": "streamable_http",
    "url": "https://backend.composio.dev/v3/mcp/<composio-token>/mcp?user_id=<entity-id>",
    "headers": {
      "x-api-key": "<composio-api-key>"
    }
  }
}
```

| Parameter | Where | Description |
|---|---|---|
| `<composio-token>` | URL path | Composio MCP token (project-level) |
| `user_id` | URL query param | Composio entity/user ID for account lookup |
| `x-api-key` | HTTP header | Composio API key |

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
    "tags": ["python", "api"],
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
    "tags": "python",
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

### Example 6: Query LLM with Composio MCP (Gmail, etc.)

Composio requires the `user_id` as a URL query parameter and the API key as `x-api-key` header.

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "List my unread emails",
    "llm_key": "your-openai-key",
    "llm": "openai",
    "model": "gpt-4o",
    "servers": {
      "composio": {
        "transport": "streamable_http",
        "url": "https://backend.composio.dev/v3/mcp/<composio-token>/mcp?user_id=<entity-id>",
        "headers": {
          "x-api-key": "<composio-api-key>"
        }
      }
    }
  }'
```

---

## Semantic Cache APIs

The semantic cache reduces latency and LLM costs for recurring queries. Enable it per-request with `use_cache: true` in any `QuestionAnswer` payload. The cache uses L1 exact match and L2 cosine similarity (threshold: 0.90 by default, configurable via `CACHE_SIMILARITY_THRESHOLD` env var).

Cache is automatically invalidated when:
- A document is added/updated via `POST /api/scrape/single`, `POST /api/ingestion`, or `POST /api/pdf/scrape`
- A namespace is deleted via `DELETE /api/namespace/{namespace}/{token}`

### GET `/api/cache/stats`
Returns entry counts for the semantic cache.

**Query params**:
- `namespace` (optional): filter stats to a specific namespace

**Response**:
```json
{
  "available": true,
  "namespace": "my-bot",
  "semantic_entries": 42
}
```

---

### DELETE `/api/cache/namespace/{namespace}`
Manually invalidate all cache entries for a namespace.

**Response**:
```json
{
  "deleted": 15,
  "namespace": "my-bot"
}
```

---

**Environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_SIMILARITY_THRESHOLD` | `0.90` | Cosine similarity threshold for L2 semantic match |
| `CACHE_TTL_EXACT` | `86400` | TTL in seconds for exact-match entries (24h) |
| `CACHE_TTL_SEMANTIC` | `21600` | TTL in seconds for semantic entries (6h) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (shared with TaskIQ) |

---

## Support

**Repository**: https://github.com/Tiledesk/tiledesk-llm

**Last updated**: 2026-04-07
