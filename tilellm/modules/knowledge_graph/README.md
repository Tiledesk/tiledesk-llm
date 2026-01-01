# Knowledge Graph (GraphRAG) Module

A modular component for Graph-based Retrieval Augmented Generation (GraphRAG) using Neo4j graph database and MinIO object storage.

## Features

- **Graph-based Retrieval**: Leverage graph structures for enhanced RAG with entity and relationship awareness
- **Neo4j Integration**: Efficient connection pooling with configurable connection management
- **MinIO Storage**: Store graph embeddings, community reports, and intermediate results in object storage
- **Complete CRUD Operations**: Create, Read, Update, Delete for nodes and relationships
- **Advanced Search Methods**:
  - Community/Global Search
  - Integrated Hybrid Search (Global + Parallel Retrieval + RRF + Expansion + Reranking)
  - Microsoft GraphRAG integration
- **Clustering Algorithms**: Louvain, Leiden, Hierarchical clustering for community detection
- **RESTful API**: FastAPI endpoints with OpenAPI/Swagger documentation
- **Modular Architecture**: Can be enabled/disabled via configuration
- **Type-safe**: Pydantic models for automatic validation
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes

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
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `NEO4J_DATABASE` | Neo4j database name | `neo4j` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin` |
| `MINIO_SECURE` | Use HTTPS | `false` |

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
- `POST /api/kg/add-document` - Create/import knowledge graph from vector store namespace
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

### 6. Hybrid Search
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
Use clustering endpoints (`/api/kg/cluster`, `/api/kg/clusterms`) to automatically detect and organize related content into communities.

## Integration with Main Application

The Knowledge Graph module integrates seamlessly with Tiledesk LLM:
- **Authentication**: Uses the same JWT token system
- **Vector Stores**: Compatible with Pinecone and Qdrant
- **Configuration**: Centralized via `service_conf.yaml`
- **Docker**: Available via `app-graph` profile

## Roadmap

- [ ] Semantic search with vector embeddings in graph queries
- [ ] Custom Cypher query endpoints
- [ ] Advanced graph traversal for RAG
- [ ] Caching for frequent queries
- [ ] Batch operations for large graphs
- [ ] Metrics and monitoring dashboard
- [ ] Real-time graph updates

## Resources

- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Neo4j Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)

## Support

For issues and questions, refer to the main project repository: https://github.com/Tiledesk/tiledesk-llm

---

**Module Status**: Active  
**Last Updated**: December 2025