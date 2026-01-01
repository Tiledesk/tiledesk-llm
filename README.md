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

#### Sparse Encoders (Hybrid Search)
- **SPLADE** (`splade`): Sparse lexical encoder via Pinecone Text
- **BGE-M3** (`bge-m3`): Sparse+dense encoder from FlagEmbedding

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
  --link your-redis-container:redis \
  -e JWT_SECRET_KEY="yourkey-256-bit" \
  -e REDIS_URL="redis://redis:6379/0" \
  -e TOKENIZERS_PARALLELISM=false \
  -e WORKERS=3 \
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
   cp .env.example .env
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

## Modular Architecture

Tiledesk LLM now features a modular architecture that allows you to enable or disable specific features based on your needs. This reduces deployment footprint and resource usage by loading only the modules you require.

### Available Modules

| Module | Description | Optional Dependencies | Docker Profile |
|--------|-------------|----------------------|----------------|
| **Base** | Core RAG functionality (scraping, QA, namespace management) | None | `app-base` |
| **Knowledge Graph (GraphRAG)** | Graph-based retrieval and reasoning with Neo4j and MinIO | `graph` (neo4j, minio, langchain-aws) | `app-graph` |
| **PDF OCR** | Optical Character Recognition for PDF documents | `ocr` (pdf2image, paddleocr, unstructured) | `app-ocr` |
| **Conversion** | File format conversion (XLSX↔CSV, PDF→text/images) | Built-in | All profiles |
| **Tools Registry** | Tool management for LLM interactions | Built-in | All profiles |

**Recent enhancements to Knowledge Graph module:**
- **Incremental document addition**: Add single documents to existing graphs via `/api/kg/add-document`
- **Automatic community report updates**: Community reports are automatically regenerated after document addition
- **Sparse encoder support**: Full support for hybrid search with SPLADE and BGE-M3 sparse encoders

## Configuration-Based Module Loading

Modules can be enabled/disabled via the `service_conf.yaml` configuration file:

1. Copy the template:
   ```bash
   cp service_conf.yaml.template service_conf.yaml
   ```

2. Edit `service_conf.yaml` to enable desired modules:
   ```yaml
   services:
     graphrag: true      # Enable Knowledge Graph module
     pdf_ocr: false      # Disable PDF OCR module
     conversion: true    # Enable conversion module
     tools_registry: true
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
- **Supported Models**: Cross‑encoder models from Hugging Face (e.g., Microsoft MARCO models, BGE rerankers).
- **Integration**: Works with both hybrid search (`search_type="hybrid"`) and similarity/MMR search (`search_type="similarity"` or `"mmr"`).

**Example** (QA with reranking):
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

