# Knowledge Graph Module

Modulo per la gestione di un Knowledge Graph basato su Neo4j per RAG (Retrieval Augmented Generation).

## Caratteristiche

- **Pool di connessioni Neo4j**: Gestione efficiente delle connessioni con pool configurabile
- **Operazioni CRUD complete**: Create, Read, Update, Delete per nodi e relazioni
- **API RESTful**: Endpoint FastAPI completamente documentati con OpenAPI/Swagger
- **Architettura a layer**: Controller → Service → Repository per separazione delle responsabilità
- **Type-safe**: Modelli Pydantic per validazione automatica
- **Gestione errori**: Error handling completo con status code HTTP appropriati

## Configurazione

### 1. Installare dipendenze

```bash
poetry install
```

Le dipendenze includono `neo4j` driver per Python.

### 2. Configurare Neo4j

Modifica le credenziali in `modules/knowledge_graph/controllers.py`:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_POOL_SIZE = 50
```

**Raccomandazione**: Spostare queste configurazioni in variabili d'ambiente o file di configurazione.

### 3. Avviare Neo4j

Usando Docker:

```bash
sudo docker run -d --name neo4j \
  --publish 7474:7474 --publish 7687:7687 \
  --env NEO4J_AUTH=neo4j/password\
  --env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
  --volume=<directory>/neo4j_data:/data \
  neo4j:latest
```

## Struttura del Modulo

```
modules/knowledge_graph/
├── __init__.py           # Export del modulo
├── controllers.py        # Endpoint FastAPI
├── models/
│   ├── __init__.py
│   └── models.py         # Modelli Pydantic (Node, Relationship, etc.)
├── services/
│   ├── __init__.py
│   └── services.py       # Business logic
├── repository/
│   ├── __init__.py
│   └── repository.py     # Accesso dati Neo4j
└── README.md
```

## API Endpoints

### Utility

- `GET /api/kg/health` - Verifica connessione Neo4j
- `GET /api/kg/stats` - Statistiche database

### Nodi

- `POST /api/kg/nodes` - Crea nodo
- `GET /api/kg/nodes/{node_id}` - Leggi nodo
- `GET /api/kg/nodes?label=...` - Lista nodi per label
- `GET /api/kg/nodes/search?label=...&property_key=...&property_value=...` - Ricerca nodi
- `PUT /api/kg/nodes/{node_id}` - Aggiorna nodo
- `PATCH /api/kg/nodes/{node_id}` - Aggiorna parziale nodo
- `DELETE /api/kg/nodes/{node_id}` - Elimina nodo

### Relazioni

- `POST /api/kg/relationships` - Crea relazione
- `GET /api/kg/relationships/{relationship_id}` - Leggi relazione
- `GET /api/kg/nodes/{node_id}/relationships?direction=...` - Lista relazioni di un nodo
- `PUT /api/kg/relationships/{relationship_id}` - Aggiorna relazione
- `PATCH /api/kg/relationships/{relationship_id}` - Aggiorna parziale relazione
- `DELETE /api/kg/relationships/{relationship_id}` - Elimina relazione

## Esempi d'Uso

### 1. Health Check

```bash
curl http://localhost:8000/api/kg/health
```

### 2. Creare un Nodo

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

Risposta:
```json
{
  "id": "123",
  "label": "Document",
  "properties": {
    "title": "Introduction to RAG",
    "content": "RAG stands for...",
    "embedding": [0.1, 0.2, 0.3]
  }
}
```

### 3. Creare una Relazione

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

### 4. Cercare Nodi

```bash
curl "http://localhost:8000/api/kg/nodes?label=Document&limit=10"
```

### 5. Aggiornare un Nodo

```bash
curl -X PUT http://localhost:8000/api/kg/nodes/123 \
  -H "Content-Type: application/json" \
  -d '{
    "properties": {
      "title": "Updated Title",
      "last_modified": "2025-01-20"
    }
  }'
```

### 6. Ottenere le Relazioni di un Nodo

```bash
curl "http://localhost:8000/api/kg/nodes/123/relationships?direction=outgoing"
```

### 7. Eliminare un Nodo

```bash
curl -X DELETE "http://localhost:8000/api/kg/nodes/123?detach=true"
```

## Modelli Dati

### Node

```python
{
  "id": "string",           # Auto-generato da Neo4j
  "label": "string",        # Tipo di nodo (es: Document, Person)
  "properties": {           # Proprietà custom
    "key": "value"
  }
}
```

### Relationship

```python
{
  "id": "string",           # Auto-generato da Neo4j
  "source_id": "string",    # ID nodo sorgente
  "target_id": "string",    # ID nodo target
  "type": "string",         # Tipo relazione (es: REFERENCES)
  "properties": {           # Proprietà custom
    "key": "value"
  }
}
```

## Convenzioni Neo4j

- **Label dei nodi**: Usare PascalCase (es: `Document`, `Person`)
- **Tipi di relazioni**: Usare UPPER_SNAKE_CASE (es: `RELATES_TO`, `REFERENCES`)
- **Properties**: Usare snake_case (es: `created_at`, `document_id`)

## Pool di Connessioni

Il modulo utilizza un pool di connessioni con le seguenti impostazioni:

- **Max pool size**: 50 connessioni (configurabile)
- **Acquisition timeout**: 60 secondi
- **Connection lifetime**: 1 ora

Il pool viene inizializzato una sola volta e riutilizzato per tutte le richieste, garantendo performance ottimali.

## Error Handling

Il modulo gestisce i seguenti errori:

- **400 Bad Request**: Validazione fallita, dati non validi
- **404 Not Found**: Risorsa non trovata
- **500 Internal Server Error**: Errori del database o interni
- **503 Service Unavailable**: Neo4j non raggiungibile

## Testing

Per testare il modulo, accedere alla documentazione Swagger:

```
http://localhost:8000/docs
```

Qui troverai tutti gli endpoint documentati con esempi interattivi.

## Best Practices per RAG

### 1. Memorizzare Documenti con Embeddings

```python
# Creare un nodo documento con embedding
{
  "label": "Document",
  "properties": {
    "content": "...",
    "embedding": [...],  # Vector embedding
    "metadata": {...}
  }
}
```

### 2. Creare Relazioni Semantiche

```python
# Collegare documenti correlati
{
  "source_id": "doc1",
  "target_id": "doc2",
  "type": "SIMILAR_TO",
  "properties": {
    "similarity_score": 0.85
  }
}
```

### 3. Modellare Gerarchia di Conoscenza

```python
# Documento -> Sezioni -> Paragrafi
doc = create_node(label="Document", properties={...})
section = create_node(label="Section", properties={...})
create_relationship(doc.id, section.id, "CONTAINS")
```

## Roadmap

- [ ] Implementare ricerca semantica con vector embeddings
- [ ] Aggiungere query Cypher personalizzate
- [ ] Implementare graph traversal per RAG avanzato
- [ ] Aggiungere caching per query frequenti
- [ ] Supporto per batch operations
- [ ] Metriche e monitoring

## Risorse

- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Neo4j Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
