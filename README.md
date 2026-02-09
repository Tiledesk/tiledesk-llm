# Tiledesk LLM

Tiledesk LLM is a powerful backend service designed for Retrieval-Augmented Generation (RAG). It provides a comprehensive suite of REST APIs to handle document scraping, indexing, question answering, and interaction with various Large Language Models (LLMs).

**Interactive API Docs**:
- **Swagger UI**: Access at the root URL (e.g., `http://localhost:8000/docs`)
- **ReDoc**: Access at `/redoc` (e.g., `http://localhost:8000/redoc`)

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running with Docker](#running-with-docker)
- [Modular Architecture](#modular-architecture)
- [Configuration-Based Module Loading](#configuration-based-module-loading)
- [Docker Compose Profiles](#docker-compose-profiles)
- [API Documentation](#api-documentation)
  - [Scraping & Indexing APIs](#scraping--indexing-apis)
  - [Question & Answer APIs](#question--answer-apis)
  - [Namespace Management APIs](#namespace-management-apis)
  - [File Conversion](#conversion-apis)
  - [Tools Registry APIs](#tools-registry-apis)
  - [Authentication](#authentication)
- [Advanced Features](#advanced-features)
  - [Hybrid Search](#hybrid-search)
  - [Semantic Chunks](#semantic-chunks)
  - [Reranker](#reranker)
  - [Structured Output](#structured-output)
  - [Tag Filtering](#tag-filtering)
  - [MCP (Model Context Protocol) Integration](#mcp-model-context-protocol-integration)

## Supported Models

### Large Language Models (LLMs)

| Provider (`llm`) | Models (`model`)                                                              
|------------------|-------------------------------------------------------------------------------|
| **OpenAI**       | `gpt-5.2`, `gpt-5-mini`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`                   
| **Anthropic**    | `claude-opus-4.5`, `claude-sonnet-4.5`, `claude-haiku-4.5`, `claude-opus-4.1` 
| **Google**       | `gemini-3-pro`, `gemini-2.5-pro`, `gemini-2.5-flash`                          
| **Groq**         | `mixtral-8x22b`, `llama-4-scout`, `llama3-70b`, `llama3-8b`, `mixtral-8x7b`   
| **Deepseek**     | `deepseek-chat`                                                               |
| **Ollama**       | `llama3.x` ...                                                                |
| **vLLM**         | `qwen2.5` ...                                                                 |

### Embedding Models

The system supports multiple embedding providers through a flexible factory pattern. Embeddings are used for vector search and can be configured in scraping and QA APIs.

| Provider | Supported Models | Default Dimensions | Notes |
|----------|------------------|-------------------|-------|
| **OpenAI** | `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large` | 1536-3072 | Default provider for most operations |
| **HuggingFace** | `BAAI/bge-m3`, `all-MiniLM-L6-v2` (and any HF model) | 384-1024 | Local models with GPU support |
| **Ollama** | Any Ollama embedding model | 4096 | Local Ollama instances |
| **Google** | Google Generative AI models | 768 | Google AI Studio embeddings |
| **Cohere** | Cohere embedding models | 1024 | Cohere API embeddings |
| **VoyageAI** | `voyage-multilingual-2` | 1024 | Voyage AI multilingual embeddings |
| **vLLM** | vLLM OpenAI-compatible models | 3072 | vLLM server embeddings |
| **TEI** | Any TEI-compatible model (e.g., `intfloat/multilingual-e5-large-instruct`) | Varies | Text Embeddings Inference server |

### TEI (Text Embeddings Inference) Models

For sparse encoding, reranking, and dense embeddings, you can use models served by [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) from Hugging Face. TEI provides an efficient API for inference of embedding, sparse encoding, and reranking models.

- **Dense Embeddings**: Any TEI-compatible embedding model (e.g., `intfloat/multilingual-e5-large-instruct`) can be served via TEI and configured using environment variables (e.g., `TEI_EMBEDDING_URL`, `TEI_EMBEDDING_MODEL`).
- **Sparse Encoder**: The `splade` model (e.g., `naver/efficient-splade-VI-BT-large-query`) can be served via TEI and configured using environment variables (e.g., `TEI_SPARSE_ENCODER_URL`, `TEI_SPARSE_ENCODER_MODEL`).
- **Reranker**: The `bge-reranker` model (e.g., `BAAI/bge-reranker-large`) can be served via TEI and configured using environment variables (e.g., `TEI_RERANKER_URL`, `TEI_RERANKER_MODEL`).

To use them, ensure you have started TEI services (e.g., via the Docker Compose `tei` profile) and configured the correct URLs via environment variables.

Example environment variables in `.env.example`:

```yaml
tei:
  embedding:
    url: "http://localhost:7580"
    model: "intfloat/multilingual-e5-large-instruct"
  sparse_encoder:
    url: "http://localhost:7380"
    model: "naver/efficient-splade-VI-BT-large-query"
  reranker:
    url: "http://localhost:7480"
    model: "BAAI/bge-reranker-large"
```

#### Sparse Encoders (Hybrid Search)
- **SPLADE** (`splade`): Sparse lexical encoder via Pinecone Text
- **BGE-M3** (`bge-m3`): Sparse+dense encoder from FlagEmbedding

Both SPLADE and BGE-M3 sparse encoders can also be served via TEI (Text Embeddings Inference) for improved performance (see [TEI Models](#tei-text-embeddings-inference-models) section).

#### Configuration
Embedding models can be specified in API requests via the `embedding` parameter. The system automatically handles:
- HTTP session pooling for network-based providers
- Caching of embedding instances
- Automatic error recovery and reconnection
- Device management for local models (CPU/GPU auto-detection)

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:** If you don't have it, follow the [official instructions](https://python-poetry.org/docs/#installation).
2.  **Install Dependencies:** From the project root directory, run:
    ```bash
    poetry install
    ```

---

## Configuration

The application is configured via environment variables.

| Variable               | Description                                                                 | Default     |
| ---------------------- | --------------------------------------------------------------------------- | ----------- |
| `JWT_SECRET_KEY`       | **Required.** A 256-bit secret key for signing JWTs.                        | `None`      |
| `REDIS_URL`            | **Required.** The connection URL for your Redis instance.                   | `None`      |
| `WORKERS`              | The number of worker processes. A good starting point is `2 * CPU cores + 1`. | `None`      |
| `TIMEOUT`              | Worker timeout in seconds.                                                  | `180`       |
| `MAXREQUESTS`          | The maximum number of requests a worker will process before restarting.     | `1200`      |
| `MAXRJITTER`           | The maximum jitter to add to `MAXREQUESTS`.                                 | `5`         |
| `GRACEFULTIMEOUT`      | Timeout for a graceful worker restart.                                      | `30`        |
| `TOKENIZERS_PARALLELISM` | Set to `false` to avoid warnings with HuggingFace tokenizers.               | `None`      |
| `ENABLE_PROFILER`      | Set to `True` to enable the `fastapi-cprofile` profiler.                    | `False`     |
| `LOG_LEVEL`            | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                        | `INFO`      |

---

## Running the Application

Once you have installed the dependencies and set up the environment variables, you can start the server using the following command:

```bash
poetry run tilellm
```

The server will be available at `http://localhost:8000` by default.

---

## Running with Docker

### Build the Image

To build the base Docker image, run the following command from the project root:

```bash
docker build -t tilellm .
```

For modular builds, use the Dockerfiles in the `docker/` directory:

```bash
# Build specific module configurations
docker build -f docker/Dockerfile.base -t tilellm-base .
docker build -f docker/Dockerfile.graphrag -t tilellm-graph .
docker build -f docker/Dockerfile.pdfocr -t tilellm-ocr .
docker build -f docker/Dockerfile.all -t tilellm-all .
```

### Run the Container

You can run the application as a container, linking it to a Redis container for storage.

```bash 
  docker run -d -p 8000:8000 \
  --name tilellm \
  --add-host=host.docker.internal:host-gateway \
  --network your-docker-network \
  -e JWT_SECRET_KEY="your-key-256" \
  -e REDIS_URL="redis://your-redis-container:6379/0" \
  -e TOKENIZERS_PARALLELISM=false \
  -e WORKERS=3 \
  -e ENABLE_TASKIQ=True|False \
  -e ENABLE_GRAPHRAG=True|False \
  -e ENABLE_PDF_OCR=True|False \
  -e ENABLE_CONVERSION=True|False \
  -e ENABLE_TOOLS_REGISTRY=True|False \
  -e ENABLE_API_V2=True|False \
  tilellm
```
*Replace `your-redis-container` with the name of your running Redis container.*

### Docker Compose

The project uses a modular Docker Compose setup with profiles for different module combinations. All Docker Compose configuration is located in the `docker/` directory.

#### Environment Configuration

1. Navigate to the `docker/` directory:
   ```bash
   cd docker
   ```

2. Copy the environment template and customize:
   ```bash
   cp docker/.env.example .env
   # Edit .env to configure ports, images, and optional features
   ```

3. Start services using profiles (from within the `docker/` directory):
   ```bash
   # Base application
   docker-compose --profile app-base up --build

   # GraphRAG enabled
   docker-compose --profile app-graph up --build

   # PDF OCR enabled
   docker-compose --profile app-ocr up --build

   # All modules
   docker-compose --profile app-all up --build

   # With TEI services (local ML inference)
   docker-compose --profile tei --profile app-all up --build

   # With Qdrant vector store
   docker-compose --profile qdrant --profile app-base up --build
   ```

#### Available Profiles

| Profile | Description |
|---------|-------------|
| `app-base` | Base application (core RAG) |
| `app-graph` | Base + Knowledge Graph module |
| `app-ocr` | Base + PDF OCR module |
| `app-all` | All modules enabled |
| `tei` | Text Embeddings Inference services (GPU) |
| `qdrant` | Qdrant vector store |

For other setups (e.g., `docker-compose-vllm.yml`), use:
```bash
docker-compose -f <compose-file-name>.yml up --build -d
```

### Qdrant Vector Store

Qdrant can be run either as a standalone container or via the Docker Compose `qdrant` profile.

#### Standalone Container
```bash
docker run -p 6333:6333 -p 6334:6334 \
    --name qdrant \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

#### Docker Compose Profile
Using the `qdrant` profile with Docker Compose (recommended for integration):

1. Ensure Qdrant is enabled in your `.env` file (default ports: 6333/6334)
2. Start with any application profile:
   ```bash
   cd docker
   docker-compose --profile qdrant --profile app-base up --build
   ```

The Qdrant service will be available at `http://localhost:6333` (or your configured port).

### Milvus Vector Store

Milvus is a cloud-native vector database that supports both dense and sparse vectors, enabling hybrid search capabilities. It can be deployed locally or using managed cloud services.

#### Standalone Container

Run Milvus standalone using Docker:

```bash
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v /path/to/milvus/data:/var/lib/milvus \
  milvusdb/milvus:v2.6.7
```

The Milvus service will be available at `http://localhost:19530` (gRPC) and `http://localhost:9091` (HTTP).

#### Configuration

To use Milvus as your vector store, configure the engine in your JWT token with the following parameters:

- `name`: `"milvus"`
- `deployment`: `"local"` (for local deployment) or `"cloud"` (for Zilliz Cloud)
- `host`: `"localhost"` (or your Milvus server host)
- `port`: `19530` (default gRPC port) or `9091` (HTTP port)
- `index_name`: Your collection name
- `apikey`: Optional API key for cloud deployments
- `database`: Optional database name (default: "default")
- `metric`: Optional distance metric (`"L2"`, `"IP"`, `"COSINE"`), defaults to `"L2"`

Example engine configuration for local Milvus:

```json
{
  "name": "milvus",
  "deployment": "local",
  "host": "localhost",
  "port": 19530,
  "index_name": "my-collection",
  "metric": "COSINE"
}
```

#### Features

- **Hybrid Search**: Supports both dense and sparse vectors (requires Milvus 2.4+)
- **Namespace Support**: Uses metadata filtering for namespace isolation
- **Tag Filtering**: Full support for tag filtering expressions
- **Scalability**: Horizontal scaling for large-scale vector search

### FalkorDB Graph Database

FalkorDB is a Redis-based graph database that brings native graph capabilities to Redis. It's the recommended choice for the Knowledge Graph (GraphRAG) module, offering superior performance and scalability.

#### Standalone Container

Run FalkorDB using Docker:

```bash
docker run -d \
  --name falkordb \
  -p 6380:6379 \
  -v /path/to/falkordb/data:/data \
  falkordb/falkordb:latest
```

The FalkorDB service will be available at `redis://localhost:6380`.

#### Configuration

To use FalkorDB for GraphRAG, configure these environment variables:

```bash
# Enable FalkorDB implementation
export ENABLE_GRAPHRAG_FALKOR=true

# FalkorDB connection URI (defaults to REDIS_URL if not specified)
export FALKORDB_URI="redis://localhost:6380"

# Optional: Use same Redis instance as application
# export FALKORDB_URI=$REDIS_URL

# Optional: Configure connection pool
export FALKORDB_MAX_CONNECTIONS=50
export FALKORDB_SOCKET_TIMEOUT=30
```

#### Features

- **Native Graph Operations**: Cypher query language support via openCypher
- **Redis Integration**: Leverages Redis infrastructure (persistence, replication, clustering)
- **High Performance**: In-memory graph processing with disk persistence
- **Multi-Tenancy**: Namespace-per-graph isolation for secure multi-user environments
- **Async/Await**: Full async implementation with connection pooling
- **Production Ready**: Battle-tested in high-scale deployments

#### Docker Compose Integration

FalkorDB is included in the `docker-compose-graph.yml` configuration. To start with GraphRAG:

```bash
cd docker
docker-compose -f docker-compose-graph.yml up -d
```

This starts:
- FalkorDB (port 6380)
- MinIO (object storage for community reports)
- Tiledesk LLM with GraphRAG enabled

#### GraphRAG Workflow with FalkorDB

1. **Ingestion**: Documents → Vector Store → GraphRAG extraction
2. **Graph Construction**: Entities & relationships → FalkorDB graph
3. **Community Detection**: Leiden clustering (3 hierarchical levels)
4. **Report Generation**: LLM-based community reports with synthetic QA
5. **Vector Indexing**: Reports embedded and indexed for semantic search
6. **Query Time**: Context fusion (global + local + graph expansion)

See [FalkorDB GraphRAG Documentation](tilellm/modules/knowledge_graph_falkor/README.md) for detailed API usage and examples.

## Modular Architecture

Tiledesk LLM now features a modular architecture that allows you to enable or disable specific features based on your needs. This reduces deployment footprint and resource usage by loading only the modules you require.

### Available Modules

| Module | Description | Optional Dependencies | Docker Profile |
|--------|-------------|----------------------|----------------|
| **Base** | Core RAG functionality (scraping, QA, namespace management) | None | `app-base` |
| **Knowledge Graph (Neo4j)** | Graph-based retrieval and reasoning with Neo4j and MinIO | `graph` (neo4j, minio, langchain-aws) | `app-graph` |
| **Knowledge Graph (FalkorDB)** | **[NEW]** Production-ready GraphRAG with FalkorDB (Redis-based graph), Leiden clustering, adaptive expansion | `graph` (falkordb, minio, langchain-aws) | `app-graph` (enable with `ENABLE_GRAPHRAG_FALKOR=true`) |
| **PDF OCR** | Optical Character Recognition for PDF documents | `ocr` (pdf2image, paddleocr, unstructured) | `app-ocr` |
| **Conversion** | File format conversion (XLSX↔CSV, PDF→text/images) | Built-in | All profiles |
| **Tools Registry** | Tool management for LLM interactions | Built-in | All profiles |

**Recent enhancements to Knowledge Graph module:**
- **Incremental document addition**: Add single documents to existing graphs via `/api/kg/add-document`
- **Automatic community report updates**: Community reports are automatically regenerated after document addition
- **Sparse encoder support**: Full support for hybrid search with SPLADE and BGE-M3 sparse encoders

**FalkorDB Implementation (NEW - Production Ready):**
- **FalkorDB Integration**: Redis-based graph database with native graph operations, async/await implementation
- **Hierarchical Leiden Clustering**: Multi-level community detection (3 levels: fine/medium/coarse) with configurable resolution
- **Adaptive Graph Expansion**: Query-type aware expansion (technical: 1-hop, exploratory: 2-hop, relational: 3-hop)
- **Synthetic QA Generation**: Automatic question generation for community reports with context enhancement
- **Cross-Encoder Reranking**: Advanced relevance scoring with TEI and Pinecone Inference API support
- **Context Fusion Search**: Ultimate hybrid method combining global (community), local (vector+keyword), and graph expansion
- **RRF (Reciprocal Rank Fusion)**: Intelligent fusion of dense and sparse retrieval results
- **Cleanup Management**: Automatic cleanup of stale community reports before regeneration (prevents duplicates)
- **Multi-Tenancy**: Namespace-per-graph isolation for secure multi-user environments
- **Query Type Detection**: LLM-based detection (exploratory/technical/relational) with adaptive weight adjustment

**Architecture Features:**
- **Async-first**: Fully async/await implementation with connection pooling
- **Modular Design**: Clean separation of services (extraction, clustering, search, synthesis)
- **Production Optimizations**: Semaphore-based rate limiting, error handling, graceful degradation
- **MinIO Integration**: Efficient storage of community reports, entities, and relationships in Parquet format
- **DuckDB**: Fast analytical queries on exported graph data

Note: For more in-depth information regarding the architecture and module specifics, check the KG [README](tilellm/modules/knowledge_graph/README.md), [FalkorDB README](tilellm/modules/knowledge_graph_falkor/README.md), and the [Architecture Report](tilellm/modules/knowledge_graph/REPORT.md). For improvement suggestions based on latest GraphRAG research (2024-2026), see [IMPROVEMENTS.md](tilellm/modules/knowledge_graph_falkor/IMPROVEMENTS.md).

## Configuration-Based Module Loading

Modules can be enabled/disabled via environment variables (e.g., `TILELLM_PROFILE` or individual `ENABLE_*` flags):

1. Copy the template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to enable desired modules:
   ```bash
   # Use a profile to enable a predefined set of modules
   TILELLM_PROFILE="app-graph"
   
   # Or enable modules individually
   ENABLE_GRAPHRAG="true"
   ENABLE_PDF_OCR="false"
   ENABLE_CONVERSION="true"
   ENABLE_TOOLS_REGISTRY="true"
   ```

3. The application automatically loads only enabled modules at startup.

### Optional Dependencies

Install optional dependencies via Poetry extras:
```bash
# Install all modules
poetry install --extras "all"

# Install specific modules
poetry install --extras "graph"     # Knowledge Graph
poetry install --extras "ocr"       # PDF OCR
```

## Docker Compose Profiles

The project includes a comprehensive Docker Compose setup with profiles for different module combinations:

### Available Profiles

| Profile | Description | Modules Enabled |
|---------|-------------|-----------------|
| `app-base` | Base application (core RAG) | Base, Conversion, Tools Registry |
| `app-graph` | GraphRAG enabled | Base + Knowledge Graph |
| `app-ocr` | PDF OCR enabled | Base + PDF OCR |
| `app-all` | All modules | All features |
| `tei` | Text Embeddings Inference services (GPU) | SPLADE, Reranker, Embedding models |
| `qdrant` | Qdrant vector store | Vector database for RAG |

**Note**: The Knowledge Graph module supports both Neo4j and FalkorDB (Redis-based) graph databases.

**FalkorDB (Recommended for Production):**
- Use environment variable `ENABLE_GRAPHRAG_FALKOR=true` to enable the FalkorDB implementation
- Configure connection via `FALKORDB_URI` environment variable (defaults to `REDIS_URL`)
- Fully async implementation with superior performance and scalability
- Native Redis integration for caching and connection pooling
- Supports all advanced features: hierarchical clustering, adaptive expansion, context fusion

**Neo4j (Legacy):**
- Default when `ENABLE_GRAPHRAG_FALKOR` is not set
- Full-featured Neo4j driver with Cypher query language
- Best for existing Neo4j deployments

Configuration example for FalkorDB:
```bash
export ENABLE_GRAPHRAG_FALKOR=true
export FALKORDB_URI="redis://localhost:6380"  # FalkorDB default port
# Or use same Redis instance as application:
export FALKORDB_URI=$REDIS_URL
```

### Usage Examples

```bash
# Start base application
docker-compose --profile app-base up --build

# Start with Knowledge Graph module
docker-compose --profile app-graph up --build

# Start with PDF OCR module
docker-compose --profile app-ocr up --build

# Start all modules
docker-compose --profile app-all up --build

# Combine with TEI services
docker-compose --profile tei --profile app-graph up --build

# With Qdrant vector store
docker-compose --profile qdrant --profile app-base up --build
```

### Dockerfile Structure

The modular architecture uses separate Dockerfiles in the `docker/` directory:
- `Dockerfile.base` - Base application
- `Dockerfile.graphrag` - Knowledge Graph module
- `Dockerfile.pdfocr` - PDF OCR module
- `Dockerfile.all` - All modules

Each Dockerfile installs only the required dependencies for its profile.

---

## API Documentation

This section provides a detailed overview of the available API endpoints.

### Scraping & Indexing APIs

#### `POST /api/scrape/single`
Indexes a single document. Can be configured for dense or hybrid indexing.

- **Request Body**: `ItemSingle`
- **Response**: `{"message": "Item {id} created successfully"}`
- **Behavior**: If a `webhook` is provided, the operation is queued and a `300` status is returned. Otherwise, it's processed synchronously.

#### `POST /api/scrape/hybrid`
A dedicated endpoint for hybrid indexing (vector + keyword).

- **Request Body**: `ItemSingle` (must include `"hybrid": true`)
- **Key Parameters**: `hybrid_batch_size`, `doc_batch_size`, `sparse_encoder` (`"splade"` or `"bge-m3"`).

#### `POST /api/scrape/status`
Checks the status of a queued indexing operation.

- **Request Body**: `ScrapeStatusReq`
- **Response**: `ScrapeStatusResponse` with status codes (`0`: Queued, `2`: Started, `3`: Finished, `4`: Error).
### Question & Answer APIs

#### `POST /api/qa`
Performs a query against the knowledge base, optionally using conversation history.

- **Request Body**: `QuestionAnswer`
- **Response**: `RetrievalResult` (with answer) or `RetrievalChunksResult` (if `chunks_only` is true).
- **Features**: Supports different search types (`similarity`, `hybrid`, `mmr`), citation generation, and returns sources.

#### `POST /api/ask`
Sends a direct query to an LLM. Supports advanced features like routing to MCP (Multi-Computation Platform) servers and using external tools.

- **Request Body**: `QuestionToLLM`
- **Response**: `SimpleAnswer` containing the LLM's response, tool execution logs, and token usage info.

#### `POST /api/thinking`
Performs a query with advanced reasoning capabilities for complex problems. Supports GPT-5, Claude 4/4.5, Gemini 2.5/3.0, and DeepSeek reasoner models.

- **Request Body**: Same as `/api/ask`, with `thinking` configuration object
- **Response**: `ReasoningAnswer` containing both `answer` and `reasoning_content`

**Supported Models**:
- **OpenAI**: `gpt-5-nano`, `gpt-5-mini`, `gpt-5` (parameters: `reasoning_effort`, `reasoning_summary`)
- **Anthropic**: `claude-sonnet-4.5`, `claude-opus-4.5` (parameters: `type`, `budget_tokens`)
- **Google**: `gemini-2.5-pro` (parameter: `thinkingBudget`), `gemini-3.0-pro` (parameter: `thinkingLevel`)
- **DeepSeek**: `deepseek-reasoner` (automatic reasoning, no config needed)

**Common Parameters**:
- `show_thinking_stream` (default: `true`): Controls thinking visibility in stream

**Example** (Claude with thinking):
```json
{
  "question": "Solve this complex problem...",
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

See `API_DOCUMENTATION.md` for detailed examples for each provider.

### Namespace Management APIs

These endpoints manage vector store namespaces. A valid JWT `{token}` containing engine configuration is required in the path.

- **`GET /api/list/namespace/{token}`**: Lists all namespaces and their vector counts.
- **`GET /api/desc/namespace/{namespace}/{token}`**: Describes a specific namespace, listing document IDs.
- **`GET /api/listitems/namespace/{namespace}/{token}`**: Lists all text chunks within a namespace.
- **`GET /api/listcompleteitems/namespace/{namespace}/all`**: Lists all text chunks within a namespace.
- **`GET /api/id/{metadata_id}/namespace/{namespace}/{token}`**: Retrieves all chunks for a specific document ID.
- **`GET /api/items?source=...&namespace=...&token=...`**: Gets items filtered by source and namespace.
- **`DELETE /api/namespace/{namespace}/{token}`**: Deletes an entire namespace.
- **`DELETE /api/id/{metadata_id}/namespace/{namespace}/{token}`**: Deletes a document and its associated chunks.
- **`DELETE /api/chunk/{chunk_id}/namespace/{namespace}/{token}`**: Deletes a single, specific chunk.

### Conversion APIs

#### `POST /api/convert`
Converts files between formats. The file can be provided as a base64 string or a URL.

- **Request Body**: `ConversionRequest`
- **Conversion Types**:
  - `xlsx_to_csv`: Converts each Excel sheet into a separate CSV file.
  - `pdf_to_text`: Extracts text from a PDF.
  - `pdf_to_images`: Converts each PDF page into a PNG image.
- **Response**: `List[ConvertedFile]` containing the converted file(s).

### Tools Registry APIs

#### `GET /api/tools`
Lists all tools available in the system that can be used with the `/api/ask` endpoint.

---

### Authentication

Many endpoints are protected and require a JWT token passed as a path parameter (`/{token}`). This token must contain the `engine` configuration object, which specifies the vector store details (e.g., Pinecone or Qdrant credentials, index name).

---

## Advanced Features

### Hybrid Search
Hybrid search combines semantic (vector) search with traditional keyword (sparse) search to improve relevance.

- **Indexing**: To enable hybrid search, set `"hybrid": true` when calling `/api/scrape/single`. You can also specify a `sparse_encoder` (`splade` or `bge-m3`).
- **Querying**: In the `/api/qa` endpoint, set `search_type` to `"hybrid"`. You can tune the balance between semantic and keyword search with the `alpha` parameter (0.0 for pure keyword, 1.0 for pure vector).

### Semantic Chunks
Instead of splitting text by a fixed size, semantic chunking splits text based on semantic similarity, keeping related sentences together.

- **Enable**: Set `"semantic_chunk": true` in the `/api/scrape/single` request.
- **Methods** (`breakpoint_threshold_type`):
  - `percentile` (default): Splits where the semantic distance between sentences exceeds a certain percentile.
  - `standard_deviation`: Splits based on standard deviation of distances.
  - `interquartile`: Uses the interquartile range to find split points.
  - `gradient`: Uses gradient of distance along with percentile, useful for highly specialized domains (e.g., legal, medical).

### Reranker
Reranking improves search relevance by reordering retrieved documents based on their relevance to the query using cross-encoder models.

- **Enable**: Set `"reranking": true` in the `/api/qa` request.
- **Configuration**:
  - `reranking_multiplier` (default: `3`): Retrieve `top_k * multiplier` documents, then rerank to select best `top_k`.
  - `reranker_model` (default: `"cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2"`): Hugging Face cross‑encoder model for reranking.
- **Supported Models**:
  - **Cross‑encoder models** from Hugging Face (e.g., Microsoft MARCO models, BGE rerankers)
  - **TEI‑served reranker models** (e.g., `BAAI/bge-reranker-large`) by configuring the TEI service (see [TEI Models](#tei-text-embeddings-inference-models) section)
  - **Pinecone Inference API** (`bge-reranker-v2-m3` and other models) - managed reranking service
- **Integration**: Works with both hybrid search (`search_type="hybrid"`) and similarity/MMR search (`search_type="similarity"` or `"mmr"`).

#### Configuration Examples

**1. Default Cross‑encoder (local/transformers):**
```json
{
  "question": "Your question",
  "namespace": "your-namespace",
  "top_k": 5,
  "reranking": true,
  "reranking_multiplier": 3,
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "search_type": "hybrid"
}
```

**2. TEI‑served reranker:**
```json
{
  "question": "Your question",
  "namespace": "your-namespace",
  "top_k": 5,
  "reranking": {
    "provider": "tei",
    "name": "bge-reranker-large",
    "url": "http://localhost:7480"
  },
  "reranking_multiplier": 3,
  "search_type": "hybrid"
}
```

**3. Pinecone Inference API:**
```json
{
  "question": "Your question",
  "namespace": "your-namespace",
  "top_k": 5,
  "reranking": {
    "provider": "pinecone",
    "api_key": "your-pinecone-api-key",
    "model": "bge-reranker-v2-m3"
  },
  "reranking_multiplier": 3,
  "search_type": "hybrid"
}
```

**Note**: When using Pinecone reranker, the `reranker_model` parameter is ignored. The Pinecone Inference API automatically handles token limits and provides managed infrastructure for reranking operations.

#### Pinecone Model Specifications & Adaptive Features

The Pinecone reranker integration includes adaptive optimizations based on model capabilities:

| Model | Max Tokens per Pair | Max Documents | Max Rank Fields | Supports `truncate` | Requires Max-P Strategy | Notes |
|-------|---------------------|---------------|-----------------|-------------------|------------------------|-------|
| `cohere-rerank-3.5` | 40,000 | 200 | Unlimited | ❌ No | ❌ Disabled | High token limit, supports multiple rank fields, `truncate` parameter filtered out |
| `bge-reranker-v2-m3` | 1,024 | 100 | 1 | ✅ Yes | ✅ Enabled | Single rank field only, `truncate: "END"` default |
| `pinecone-rerank-v0` | 512 | 100 | 1 | ✅ Yes | ✅ Enabled | Single rank field only, `truncate: "END"` default |

**Adaptive Features**:
- **Max-P Strategy**: Automatically enabled for models with low token limits (`bge-reranker-v2-m3`, `pinecone-rerank-v0`) to chunk long documents while preserving relevance. Disabled for `cohere-rerank-3.5` due to high token limit.
- **Parameter Filtering**: Unsupported parameters are automatically filtered (e.g., `truncate` is removed for `cohere-rerank-3.5`).
- **Rank Field Validation**: Models limited to single rank field automatically use the first specified field.
- **Batch Processing**: Documents exceeding `max_documents` limit are automatically split into batches.

**Example with multiple rank fields (cohere model)**:
```json
{
  "reranking": {
    "provider": "pinecone",
    "api_key": "your-pinecone-api-key",
    "model": "cohere-rerank-3.5",
    "rank_fields": ["chunk_text", "title", "summary"],
    "parameters": {
      "return_scores": true
      // Note: 'truncate' parameter is automatically filtered for cohere model
    }
  }
}
```

### Structured Output

The `/api/ask`, `/api/thinking`, and `/api/qa` endpoints support structured output generation, which forces the LLM to return responses in a specific JSON format defined by a JSON schema. This is useful for extracting structured data from text, building APIs with guaranteed response shapes, or integrating with type‑safe systems.

- **Enable**: Set `"structured_output": true` in the request.
- **Schema Definition**: Provide an `output_schema` field containing the JSON Schema of the desired output structure.
- **Supported Endpoints**: Works with `/api/ask` (direct LLM queries), `/api/thinking` (reasoning models), and `/api/qa` (RAG‑based answers).

**Example** (Structured output with a custom schema):

First, define your JSON Schema:

```json
{
    "name": "courses",
    "strict": true,
    "schema": 

{
    "type": "object",
    "properties": {
      "name": {"type": "string","description": " your description" },
      "age": {"type": "integer", "description": "your description"},
      "is_student": {"type": "boolean", "description":  "your description"},
      "courses": {"type": "array", "description":  "list of courses", "items": {"type": "string"}}
      }
    },
    "additionalProperties": false,
    "required": ["name", "age", "is_student", "courses"],
  }
```

Then, make a request to `/api/ask` with the schema:

```json
{
  "question": "Extract information about John Doe, a 31-year-old student studying Math and Science.",
  "llm": "openai",
  "model": "gpt-4o",
  "llm_key": "sk-...",
  "structured_output": true,
  "output_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"},
      "is_student": {"type": "boolean"},
      "courses": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age", "is_student", "courses"],
    "additionalProperties": false
  }
}
```

**Response**:
```json
{
  "answer": {
    "name": "John Doe",
    "age": 31,
    "is_student": true,
    "courses": ["Math", "Science"]
  },
  "chat_history_dict": { ... },
  "prompt_token_info": { ... }
}
```

**Notes**:
- The LLM's raw output is automatically parsed and validated against the provided schema.
- If the LLM fails to produce a valid JSON object conforming to the schema, an error is returned.
- Structured output works with both streaming and non‑streaming requests.
- For `/api/thinking` and `/api/qa`, the structured answer is returned in the `answer` field, while reasoning content (if any) is placed in `reasoning_content`.
- Requires LLM providers that support structured output (OpenAI, Anthropic, Google Gemini, and other compatible providers).

### Tag Filtering
Tag filtering allows you to filter documents by tags during indexing and querying. You can assign tags to documents when indexing and use boolean expressions or simple lists to filter results during retrieval.

- **Indexing**: Add a `tags` field to the `ItemSingle` model when calling `/api/scrape/single` or `/api/scrape/hybrid`. Tags can be a list of strings or a single string.
- **Querying**: Add a `tags` field to the `QuestionAnswer` model when calling `/api/qa`. The tags field can be:
  - A list of strings: `["python", "api"]` (treated as AND condition)
  - A boolean expression: `"(python|javascript)&!legacy"` (supports `&` (AND), `|` (OR), `!` (NOT), parentheses)
- **Supported Vector Stores**: Pinecone (Serverless and Pod), Qdrant, and Milvus fully support tag filtering. Redis vector store is not affected (used only for caching/streaming).

**Tag Grammar**:
- Single tag: `"python"`
- OR operator: `"python|api"` (matches documents tagged with either "python" OR "api")
- AND operator: `"python&api"` (matches documents tagged with both "python" AND "api")
- NOT operator: `"!legacy"` (excludes documents tagged with "legacy")
- Parentheses: `"(python|javascript)&(api|rest)&!legacy"` (complex nested expressions)
- List syntax: `["python", "api"]` (equivalent to `"python&api"`)

**Examples**:

1. **Indexing with tags**:
```json
{
  "id": "doc-001",
  "source": "https://example.com",
  "content": "Document content...",
  "namespace": "my-docs",
  "tags": ["python", "api", "latest"],
  "engine": { ... }
}
```

2. **Querying with simple tag filter**:
```json
{
  "question": "How to use the API?",
  "namespace": "my-docs",
  "tags": "python",
  "engine": { ... }
}
```

3. **Querying with boolean expression**:
```json
{
  "question": "Find documentation about Python or JavaScript APIs",
  "namespace": "my-docs",
  "tags": "(python|javascript)&api&!legacy",
  "engine": { ... }
}
```

4. **Querying with list (AND condition)**:
```json
{
  "question": "Find latest Python API documentation",
  "namespace": "my-docs",
  "tags": ["python", "api", "latest"],
  "engine": { ... }
}
```

**Implementation Details**:
- Tags are stored in vector store metadata under the `"tags"` field
- Filter conversion happens automatically for different vector stores (Pinecone native filters, Qdrant via `build_filter()`)
- Works with all search types (`similarity`, `hybrid`, `mmr`) and reranking
- Compatible with hybrid search sparse encoders (SPLADE, BGE-M3)

### MCP (Model Context Protocol) Integration
The `/api/ask` endpoint supports integration with MCP servers, enabling the LLM to use external tools and data sources.

- **Server Configuration**: Configure MCP servers in the `servers` field with transport options (`sse`, `stdio`, `streamable_http`).
- **Tool Filtering**: Use the `enabled_tools` parameter on each server configuration to specify which tools from that server are available (default: `["all"]`).
- **Internal Tools**: Specify internal tools from the tool registry using the `tools` field in the request body.

**Example** (MCP integration):
```json
{
  "question": "Analyze the repository",
  "llm": "openai",
  "model": "gpt-4o",
  "llm_key": "sk-...",
  "servers": {
    "github": {
      "transport": "sse",
      "url": "https://mcp-server.com/github",
      "enabled_tools": ["search_repo", "read_file"]
    },
    "filesystem": {
      "transport": "stdio",
      "command": "mcp-server-fs",
      "args": ["--root", "/home/user/docs"],
      "enabled_tools": ["list_directory", "read_file"]
    }
  },
  "tools": ["pdf_extractor", "web_search"]
}
```

---

