# Knowledge Graph (GraphRAG) Module - FalkorDB Implementation

A production-ready modular component for Graph-based Retrieval Augmented Generation (GraphRAG) using **FalkorDB** (Redis-based graph database) and MinIO object storage.

## Features

### Core Capabilities
- **Graph-based Retrieval**: Leverage graph structures for enhanced RAG with entity and relationship awareness
- **FalkorDB Integration**: Redis-based graph database with native graph operations, async/await implementation
- **MinIO Storage**: Store graph embeddings, community reports, and intermediate results in object storage (Parquet format)
- **Complete CRUD Operations**: Create, Read, Update, Delete for nodes and relationships
- **Multi-Tenancy**: Namespace-per-graph isolation for secure multi-user environments

### Advanced Search Methods
- **Context Fusion Search**: Ultimate hybrid method combining three retrieval streams:
  - **Global Search**: Semantic search on community reports (LLM-generated summaries)
  - **Local Search**: Vector + keyword (hybrid) search on document chunks with RRF fusion
  - **Graph Expansion**: Adaptive multi-hop expansion (query-type aware: 1/2/3 hops)
- **Cross-Encoder Reranking**: Advanced relevance scoring across all sources
- **Query Type Detection**: LLM-based (exploratory/technical/relational) with adaptive weight matrices

### Clustering & Community Detection
- **Hierarchical Leiden Clustering**: Multi-level community detection with 3 resolution levels:
  - Level 0 (res=1.2): Fine-grained communities (many small)
  - Level 1 (res=0.8): Medium communities
  - Level 2 (res=0.5): Coarse communities (fewer, larger)
- **Louvain Algorithm**: Alternative community detection method (NetworkX-based)
- **Community Report Generation**: LLM-based reports for each community with:
  - Title, summary, key findings
  - Rating (0-5) and rating explanation
  - Synthetic QA: 3-5 generated questions per report
  - Timestamp metadata for versioning

### Advanced Features
- **Synthetic QA Generation**: Automatic question generation for community reports with fallback mechanisms
- **Adaptive Graph Expansion**: Query-type aware multi-hop expansion with early stopping
- **RRF (Reciprocal Rank Fusion)**: Intelligent fusion of dense and sparse retrieval results
- **Cleanup Management**: Automatic cleanup of stale reports before regeneration (prevents duplicates)
- **Entity Resolution**: Deduplication with semantic similarity (name/type matching)
- **Relationship Weighting**: Strength-based ranking for prioritized expansion

### Technical Excellence
- **Fully Async**: Complete async/await implementation with semaphore-based rate limiting
- **Connection Pooling**: BlockingConnectionPool for FalkorDB with configurable limits
- **Type-safe**: Pydantic models with comprehensive validation
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Observability**: Comprehensive logging with structured output
- **RESTful API**: FastAPI endpoints with OpenAPI/Swagger documentation
- **Modular Architecture**: Clean separation of services (extraction, clustering, search, synthesis)

## Module Activation

The Knowledge Graph module is part of Tiledesk LLM's modular architecture. To enable it:

### 1. Configuration
Edit `service_conf.yaml`:
```yaml
services:
  graphrag: true  # Enable Knowledge Graph module

# Required dependencies configuration
minio:
  endpoint: "localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"
  secure: false

neo4j:
  uri: "neo4j://localhost:7687"
  user: "neo4j"
  password: "password"
  database: "neo4j"
```

### 2. Install Optional Dependencies
```bash
# Install with Poetry extras
poetry install --extras "graph"

# Or install all modules
poetry install --extras "all"
```

### 3. Docker Deployment
Use the GraphRAG Docker profile:
```bash
docker-compose --profile app-graph up --build
```

## Dependencies

### Required Services
- **Neo4j**: Graph database (version 5.x)
- **MinIO**: Object storage for embeddings and reports
- **Redis**: For caching and job queues (shared with main application)

### Python Dependencies
- `neo4j`: Neo4j Python driver
- `minio`: MinIO Python SDK
- `langchain-aws`: AWS integrations (for MinIO)
- `igraph`: Graph analysis library
- `pandas`: Data processing for community reports

## Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_GRAPHRAG_FALKOR` | Enable FalkorDB implementation | `false` |
| `FALKORDB_URI` | FalkorDB connection URI (Redis protocol) | Uses `REDIS_URL` |
| `FALKORDB_MAX_CONNECTIONS` | Max connections in pool | `50` |
| `FALKORDB_SOCKET_TIMEOUT` | Socket timeout (seconds) | `30` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin` |
| `MINIO_SECURE` | Use HTTPS | `false` |

**FalkorDB Connection Examples:**
```bash
# Option 1: Dedicated FalkorDB instance
export FALKORDB_URI="redis://localhost:6380"

# Option 2: Use same Redis as application (graph stored in separate keyspace)
export FALKORDB_URI=$REDIS_URL

# Option 3: With authentication
export FALKORDB_URI="redis://:password@localhost:6380"

# Option 4: Redis Cluster
export FALKORDB_URI="redis://node1:6380,node2:6380,node3:6380"
```

### Service Configuration
Configuration is centralized in `service_conf.yaml`. See `service_conf.yaml.template` for complete options.

## API Endpoints

### Utility
- `GET /api/kg/health` - Check Neo4j connection health
- `GET /api/kg/stats` - Get database statistics (node count, relationship count, etc.)

### Node Management
- `POST /api/kg/nodes` - Create a new node
- `GET /api/kg/nodes/{node_id}` - Read node by ID
- `GET /api/kg/nodes?label=...` - List nodes by label
- `GET /api/kg/nodes/search?label=...&property_key=...&property_value=...` - Search nodes by property
- `PUT /api/kg/nodes/{node_id}` - Update node
- `PATCH /api/kg/nodes/{node_id}` - Partially update node
- `DELETE /api/kg/nodes/{node_id}` - Delete node

### Relationship Management
- `POST /api/kg/relationships` - Create relationship between nodes
- `GET /api/kg/relationships/{relationship_id}` - Read relationship by ID
- `GET /api/kg/nodes/{node_id}/relationships?direction=...` - List relationships for a node
- `PUT /api/kg/relationships/{relationship_id}` - Update relationship
- `PATCH /api/kg/relationships/{relationship_id}` - Partially update relationship
- `DELETE /api/kg/relationships/{relationship_id}` - Delete relationship

### Graph Operations
- `POST /api/kg/create` - Create/import knowledge graph from vector store namespace
- `POST /api/kg/add-document` - Add a single document to existing knowledge graph and update community reports
- `POST /api/kg/louvein-cluster` - Perform Louvain clustering with MinIO storage
- `POST /api/kg/leiden-cluster` - Perform Leiden clustering
- `POST /api/kg/hierarchical` - Perform Hierarchical Clustering

### Search & QA
- `POST /api/kg/hybrid` - **Primary endpoint**: Integrated hybrid search (Global + Parallel Retrieval + RRF + Expansion + Reranking)
- `POST /api/kg/qa` - Community/Global search on community reports

## Usage Examples

### 1. Health Check
```bash
curl http://localhost:8000/api/kg/health
```

### 2. Create a Node
```bash
curl -X POST http://localhost:8000/api/kg/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "label": "Document",
    "properties": {
      "title": "Introduction to RAG",
      "content": "RAG stands for Retrieval Augmented Generation...",
      "embedding": [0.1, 0.2, 0.3]
    }
  }'
```

### 3. Create a Relationship
```bash
curl -X POST http://localhost:8000/api/kg/relationships \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "123",
    "target_id": "456",
    "type": "REFERENCES",
    "properties": {
      "weight": 0.8,
      "context": "citation"
    }
  }'
```

### 4. Search Nodes
```bash
curl "http://localhost:8000/api/kg/nodes?label=Document&limit=10"
```

### 5. Create Graph from Vector Store
```bash
curl -X POST http://localhost:8000/api/kg/create \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "my-documents",
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "your-api-key",
      "vector_size": 1536,
      "index_name": "tilellm"
    }
  }'
```

### 6. Add Document to Graph (Incremental Update)
```bash
curl -X POST http://localhost:8000/api/kg/add-document \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 7. Context Fusion Search (Ultimate Hybrid)

**NEW:** The primary search endpoint combining global, local, and graph retrieval.

```bash
curl -X POST http://localhost:8000/api/kg/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between AI and machine learning?",
    "namespace": "my-documents",
    "search_type": "hybrid",
    "sparse_encoder": "splade",
    "reranking": {
      "provider": "tei",
      "name": "bge-reranker-large",
      "url": "http://localhost:7480"
    },
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "your-api-key",
      "vector_size": 1536,
      "index_name": "tilellm"
    },
    "llm_key": "your-llm-key",
    "model": "gpt-4"
  }'
```

**Response:**
```json
{
  "answer": "AI and machine learning are...",
  "entities": [
    {"id": "123", "label": "TECHNOLOGY", "properties": {...}},
    {"id": "456", "label": "FIELD", "properties": {...}}
  ],
  "relationships": [
    {"id": "789", "type": "SUBSET_OF", "source_id": "456", "target_id": "123"}
  ],
  "retrieval_strategy": "integrated_hybrid_exploratory",
  "scores": {
    "global_reports": 3,
    "local_chunks": 15,
    "graph_nodes": 8,
    "query_type": "exploratory"
  },
  "expanded_nodes": [...],
  "expanded_relationships": [...]
}
```

**Features:**
- Parallel execution of three retrieval streams (asyncio.gather)
- Query type detection with adaptive weight adjustment
- RRF fusion for dense + sparse results
- Cross-encoder reranking on unified results
- Chat history support for multi-turn conversations

### 8. Cleanup Before Regeneration (NEW)

Automatically clean up stale community reports before regenerating clusters:

```bash
curl -X POST http://localhost:8000/api/kg/create \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "my-documents",
    "cleanup_before": true,
    "engine": {...}
  }'
```

**What gets cleaned up:**
1. Vector store namespace `{namespace}-reports` (all embeddings)
2. BELONGS_TO_COMMUNITY relationships (graph edges)
3. CommunityReport nodes (graph nodes)

**What is preserved:**
- Entity nodes
- RELATED_TO relationships
- All original document data

**Benefits:**
- Prevents duplicate community reports
- Ensures consistency between vector store and graph
- Enables clean regeneration after adding/removing documents

### 9. Hybrid Search (Legacy)
```bash
curl -X POST http://localhost:8000/api/kg/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between AI and machine learning?",
    "namespace": "my-documents",
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "your-api-key",
      "vector_size": 1536,
      "index_name": "tilellm"
    }
  }'
```

## Data Models

### Node
```python
{
  "id": "string",           # Auto-generated by Neo4j
  "label": "string",        # Node type (e.g., Document, Person)
  "properties": {           # Custom properties
    "key": "value"
  }
}
```

### Relationship
```python
{
  "id": "string",           # Auto-generated by Neo4j
  "source_id": "string",    # Source node ID
  "target_id": "string",    # Target node ID
  "type": "string",         # Relationship type (e.g., REFERENCES)
  "properties": {           # Custom properties
    "key": "value"
  }
}
```

## Neo4j Conventions

- **Node Labels**: Use PascalCase (e.g., `Document`, `Person`, `Organization`)
- **Relationship Types**: Use UPPER_SNAKE_CASE (e.g., `RELATES_TO`, `REFERENCES`, `CITES`)
- **Properties**: Use snake_case (e.g., `created_at`, `document_id`, `embedding_vector`)

## Connection Pooling

The module uses a connection pool with the following settings:
- **Max pool size**: 50 connections (configurable)
- **Acquisition timeout**: 60 seconds
- **Connection lifetime**: 1 hour

The pool is initialized once and reused for all requests, ensuring optimal performance.

## MinIO Storage Structure

GraphRAG uses MinIO for storing:
- **Community reports** (Parquet format): `community-reports/`
- **Graph embeddings**: `embeddings/`
- **Intermediate processing results**: `intermediate/`

Bucket naming follows the pattern: `graphrag-{namespace}`.

## Error Handling

The module handles the following errors:
- **400 Bad Request**: Validation failed, invalid data
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Database or internal errors
- **503 Service Unavailable**: Neo4j or MinIO unavailable

## Testing

Access interactive API documentation:
```
http://localhost:8000/docs
```

All endpoints are documented with interactive examples.

## Best Practices for GraphRAG

### 1. Store Documents with Embeddings
```python
{
  "label": "Document",
  "properties": {
    "content": "...",
    "embedding": [...],  # Vector embedding
    "metadata": {...}
  }
}
```

### 2. Create Semantic Relationships
```python
{
  "source_id": "doc1",
  "target_id": "doc2",
  "type": "SIMILAR_TO",
  "properties": {
    "similarity_score": 0.85
  }
}
```

### 3. Model Knowledge Hierarchy
```python
# Document -> Sections -> Paragraphs
doc = create_node(label="Document", properties={...})
section = create_node(label="Section", properties={...})
create_relationship(doc.id, section.id, "CONTAINS")
```

### 4. Community Detection
Use clustering endpoints (`/api/kg/hierarchical`) to automatically detect and organize related content into communities.

## Integration with Main Application

The Knowledge Graph module integrates seamlessly with Tiledesk LLM:
- **Authentication**: Uses the same JWT token system
- **Vector Stores**: Compatible with Pinecone and Qdrant
- **Configuration**: Centralized via `service_conf.yaml`
- **Docker**: Available via `app-graph` profile

## Performance Optimization

### Connection Pooling
- Max pool size: 50 connections (configurable via `FALKORDB_MAX_CONNECTIONS`)
- Acquisition timeout: 60 seconds
- Socket timeout: 30 seconds (configurable)
- Automatic reconnection on connection failures

### Clustering Optimization
- Parallel community report generation (max 10 concurrent LLM calls)
- Semaphore-based rate limiting to prevent API overload
- Minimum community size: 3 nodes (filters noise)
- Early stopping in graph expansion (< 3 new nodes per hop)

### Query Optimization
- Query type detection caches embeddings
- Shared PineconeAsyncio client for multiple searches
- DuckDB for efficient Parquet analytics
- RRF fusion reduces redundant retrievals

## Monitoring & Observability

### Logging
All operations are logged with structured output:
```python
logger.info(f"Graph expansion complete: {nodes} nodes, {rels} relationships in {hops} hops")
logger.debug(f"Query type: {query_type}, Weights: {weights}")
```

### Metrics to Track
- **Entity extraction**: entities/relationships per document, token usage
- **Clustering**: communities detected, avg size, modularity score
- **Search**: retrieval time, expansion depth, reranking scores
- **Cleanup**: nodes/relationships deleted, vector store size

### Health Checks
```bash
# Check FalkorDB connection
curl http://localhost:8000/api/kg/health

# Get graph statistics
curl http://localhost:8000/api/kg/stats
```

## Resources

### Documentation
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [openCypher Query Language](https://opencypher.org/)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Research Papers
- **GraphRAG** (Microsoft Research, 2024): Original hierarchical community detection paper
- **Leiden Algorithm** (Nature, 2019): Superior modularity optimization vs Louvain
- **RRF (Reciprocal Rank Fusion)** (SIGIR, 2009): Classic fusion algorithm
- **Synthetic QA** (ACL, 2023): Question generation for retrieval augmentation

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed suggestions based on latest GraphRAG research (2024-2026).

## Technical Documentation

For detailed technical documentation on how the Knowledge Graph module works, including:
- Creation process (`/api/kg/create`)
- Global Search (`/api/kg/qa`) 
- Integrated Hybrid Search (`/api/kg/hybrid`)
- Role of LLMs, embeddings, reranking, and adaptive graph expansion

See the following reports:
- [REPORT_it.md](REPORT_it.md) (Italian)
- [REPORT.md](REPORT.md) (English)

## Best Practices

### 1. Graph Construction
- **Use descriptive entity names**: "Apple Inc." instead of "Apple" (avoids ambiguity)
- **Normalize entity types**: Consistent casing (ORGANIZATION, not Organization/organization)
- **Add source tracking**: Always include `source_id` for provenance
- **Weight relationships**: Use meaningful weights (1.0-5.0) based on confidence/importance

### 2. Clustering Strategy
- **Start with hierarchical**: Use 3 levels (res=1.2/0.8/0.5) for comprehensive coverage
- **Tune resolution**: Lower resolution = larger communities (adjust based on graph density)
- **Monitor community size**: Aim for 5-20 nodes per community (adjust min threshold)
- **Cleanup before regeneration**: Always use `cleanup_before=true` to prevent duplicates

### 3. Search Optimization
- **Query type matters**: Exploratory queries benefit from higher graph weights
- **Use reranking**: Cross-encoder reranking significantly improves precision
- **Tune expansion limits**: Increase `max_nodes` for complex queries (default: 30)
- **Enable sparse encoders**: SPLADE for technical docs, BGE-M3 for multilingual

### 4. Production Deployment
- **Connection pooling**: Set `FALKORDB_MAX_CONNECTIONS` based on concurrency needs
- **Monitor memory**: FalkorDB is in-memory (size graph accordingly)
- **Use MinIO for artifacts**: Essential for large-scale deployments
- **Enable cleanup**: Automatic cleanup prevents unbounded growth
- **Set rate limits**: Use semaphores to control LLM API usage

### 5. Troubleshooting
| Issue | Solution |
|-------|----------|
| Slow clustering | Reduce graph size, increase semaphore limit |
| Low retrieval quality | Enable reranking, tune query weights |
| Duplicate entities | Implement entity resolution (see IMPROVEMENTS.md) |
| High LLM costs | Use cheaper models for extraction, premium for synthesis |
| Connection timeouts | Increase `FALKORDB_SOCKET_TIMEOUT` |

## Migration from Neo4j

If migrating from Neo4j to FalkorDB:

1. **Export Neo4j data**:
   ```cypher
   MATCH (n)-[r]->(m)
   RETURN n, r, m
   ```

2. **Transform to FalkorDB format**:
   - Convert `elementId()` → `id()` (FalkorDB uses integer IDs)
   - Update property access patterns
   - Adjust Cypher queries for openCypher compatibility

3. **Import to FalkorDB**:
   ```python
   # Use async FalkorDB repository
   from tilellm.modules.knowledge_graph_falkor.repository import AsyncFalkorGraphRepository

   repo = AsyncFalkorGraphRepository()
   await repo.create_node(node_data, namespace="my-graph")
   ```

4. **Update application code**:
   - Set `ENABLE_GRAPHRAG_FALKOR=true`
   - Remove Neo4j-specific code
   - Test clustering and search

## Support

For issues and questions:
- **Documentation**: [REPORT.md](REPORT.md) (English), [REPORT_it.md](REPORT_it.md) (Italian)
- **Improvements**: [IMPROVEMENTS.md](IMPROVEMENTS.md) (research-based suggestions)
- **Issues**: https://github.com/Tiledesk/tiledesk-llm/issues
- **Main Repo**: https://github.com/Tiledesk/tiledesk-llm

---

**Module Status**: Production Ready
**Implementation**: FalkorDB (Redis-based)
**Last Updated**: February 2026
**Version**: 2.0 (Async + Hierarchical + Context Fusion)