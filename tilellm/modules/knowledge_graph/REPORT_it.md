# REPORT - Knowledge Graph (GraphRAG) Module

## 1. Introduzione

Il modulo **Knowledge Graph** di Tiledesk LLM implementa un'architettura **GraphRAG** (Graph-based Retrieval Augmented Generation) che combina:
- **Knowledge Graph** (Neo4j) per memorizzare entità e relazioni estratte dai documenti
- **Vector Store** (Pinecone/Qdrant) per la ricerca semantica sui chunk e sui report
- **MinIO** per l'archiviazione di report Parquet
- **LLM** per l'estrazione di entità, la generazione di report e la sintesi delle risposte

Il sistema offre tre principali modalità di interrogazione:
1. **Global Search** (`/api/kg/qa`): ricerca solo sui report delle community
2. **Integrated Hybrid Search** (`/api/kg/hybrid`): pipeline completa che unisce global search, local search e graph expansion
3. **Creazione del grafo** (`/api/kg/create`): importazione dei documenti, estrazione GraphRAG, clustering e generazione report

Questo documento descrive nel dettaglio il funzionamento di ciascun componente e il flusso dati end-to-end.

## 2. Architettura del modulo

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector Store  │◄──►│  GraphRAG       │◄──►│   Neo4j         │
│   (Pinecone/    │    │  Service        │    │   Knowledge     │
│   Qdrant)       │    │                 │    │   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Community     │    │   Hybrid        │    │   Graph         │
│   Reports       │    │   Search        │    │   Expansion     │
│   (Parquet)     │    │   Service       │    │   (Adaptive)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────┬───────────┴───────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │   LLM Synthesis │
            │   & Reranking   │
            └─────────────────┘
```

**Componenti principali:**
- **`CommunityGraphService`** (`services/community_graph_service.py`): orchestratore della creazione dei report e delle ricerche globali/ibride
- **`GraphRAGService`** (`services/services.py`): estrazione di entità e relazioni dai chunk, import in Neo4j
- **`ClusterService`** (`services/clustering.py`): clustering Louvain/Leiden per la detection delle community
- **`GraphExpander`** (`utils/graph_expansion.py`): espansione adattiva del grafo a partire da nodi seed
- **`TileReranker`** (`tilellm/tools/reranker.py`): cross-encoder per il reranking dei risultati
- **`GraphRepository`** (`repository/repository.py`): astrazione CRUD per Neo4j

## 3. Creazione del Knowledge Graph (`/api/kg/create`)

**Endpoint:** `POST /api/kg/create`  
**Scopo:** importare i chunk da un namespace del vector store, estrarre entità e relazioni, salvarle in Neo4j, generare report delle community e indicizzarli per la ricerca semantica.

### 3.1 Flusso dettagliato

```
1. Recupero chunk dal vector store
   ↓
2. Estrazione GraphRAG (entità + relazioni) per ogni chunk
   ↓
3. Import in Neo4j (nodi + archi)
   ↓
4. Clustering gerarchico (Leiden, 3 livelli)
   ↓
5. Generazione report di community (LLM)
   ↓
6. Arricchimento con domande sintetiche (LLM)
   ↓
7. Indexing dei report nel vector store (embedding)
   ↓
8. Esportazione Parquet + upload MinIO (opzionale)
```

#### **1. Recupero dei chunk** (`community_graph_service.py:144`)
Il servizio interroga il vector store (configurato via `engine`) nel `namespace` specificato, con un limite opzionale di chunk (`limit`).

#### **2. Estrazione GraphRAG** (`services/services.py`)
Per ogni chunk, un LLM viene invocato con un prompt specializzato per identificare:
- **Entità** (es. `PERSON`, `ORGANIZATION`, `LOCATION`) con proprietà (nome, descrizione, tipo)
- **Relazioni** (es. `WORKED_FOR`, `LOCATED_IN`, `PART_OF`) tra le entità identificate

Le entità sono deduplicate: se un'entità con lo stesso nome e tipo esiste già, viene riutilizzata.

#### **3. Import in Neo4j** (`repository/repository.py`)
- Ogni entità diventa un **nodo** con label corrispondente al tipo e proprietà `name`, `description`, `namespace`, `index_name`, `source_id` (riferimento al chunk)
- Ogni relazione diventa un **arco** tra i nodi, con tipo e proprietà (es. `weight`, `context`)

#### **4. Clustering gerarchico** (`services/clustering.py`)
Il `ClusterService` applica l'algoritmo **Leiden** con tre livelli di risoluzione:
- **Livello 0** (risoluzione 1.2): community grandi, panoramica generale
- **Livello 1** (risoluzione 0.8): community medie
- **Livello 2** (risoluzione 0.5): community piccole, altamente specifiche

Ogni community è identificata da un `community_id` univoco.

#### **5. Generazione report di community** (`services/clustering.py`)
Per ogni community, un LLM genera un report strutturato con:
- **Titolo**: sintesi del topic della community
- **Sommario**: panoramica dei contenuti
- **Rating** (0-5): valutazione della coesione e rilevanza
- **Full report**: descrizione dettagliata con elenco delle entità principali

#### **6. Arricchimento con domande sintetiche** (`utils/synthetic_qa.py`)
Ogni report è arricchito con **3 domande sintetiche** generate dallo stesso LLM, che migliorano la retrievability semantica (es. "Cosa fa l'organizzazione X?").

#### **7. Indexing dei report nel vector store** (`community_graph_service.py:313`)
I report (contenuto completo + domande sintetiche) sono convertiti in `Document` LangChain e inseriti in un namespace dedicato (`{namespace}-reports`) dello stesso vector store, utilizzando gli embedding forniti (`llm_embeddings`). Questo abilita la successiva ricerca semantica sui report.

#### **8. Esportazione Parquet + MinIO** (`community_graph_service.py:330`)
I report, le entità e le relazioni sono scritti in file Parquet e, se configurato, caricati su MinIO per analisi batch o riutilizzo offline.

### 3.2 Ruolo degli LLM
- **Estrazione entità/relazioni**: prompt di GraphRAG per identificare entità e collegamenti
- **Generazione report**: sintesi della community in linguaggio naturale
- **Domande sintetiche**: creazione di query esemplificative per migliorare la ricerca

### 3.3 Ruolo degli embedding
- I chunk originali sono già embedded nel vector store
- I report vengono a loro volta embedded (con lo stesso modello) per abilitare la ricerca semantica

**File di riferimento:**
- `tilellm/modules/knowledge_graph/controllers.py:213` – endpoint `/create`
- `tilellm/modules/knowledge_graph/logic.py:216` – logica `create_graph`
- `tilellm/modules/knowledge_graph/services/community_graph_service.py:144` – `create_community_graph`
- `tilellm/modules/knowledge_graph/services/clustering.py` – clustering e generazione report
- `tilellm/modules/knowledge_graph/utils/synthetic_qa.py` – domande sintetiche

## 4. Interrogazione: Global Search (`/api/kg/qa`)

**Endpoint:** `POST /api/kg/qa`  
**Scopo:** rispondere a una domanda cercando **solo** nei report delle community, senza accedere ai chunk originali. Ideale per domande ad alto livello che richiedono una visione d'insieme.

### 4.1 Flusso dettagliato

```
1. Ricerca semantica ibrida sui report
   ↓
2. RRF (Reciprocal Rank Fusion) [opzionale]
   ↓
3. Reranking con cross‑encoder
   ↓
4. Map‑Reduce con LLM:
   - MAP: analisi singoli report
   - REDUCE: sintesi finale
   ↓
5. Aggiornamento chat history
```

#### **1. Ricerca semantica ibrida** (`community_graph_service.py:864`)
- La domanda è convertita in **embedding denso** tramite `llm_embeddings`
- Se configurato, viene generato anche un **vettore sparse** (encoder SPLADE) per la ricerca keyword
- Viene eseguita una ricerca ibrida (densa + sparse) nel namespace `{namespace}-reports` del vector store

#### **2. RRF (Reciprocal Rank Fusion)** (`utils/rrf.py`)
I risultati delle due ricerche (densa e sparse) sono fusi tramite **Reciprocal Rank Fusion**, che bilancia recall e precisione producendo una ranked list robusta.

#### **3. Reranking con cross‑encoder** (`community_graph_service.py:884`)
I documenti risultanti sono **rerankati** da un cross‑encoder (`TileReranker`) che valuta direttamente la pertinenza query‑documento, selezionando i top‑k più rilevanti.

#### **4. Map‑Reduce con LLM** (`community_graph_service.py:950`)
- **MAP**: ogni report selezionato viene analizzato da un LLM con un prompt che chiede di estrarre le informazioni rilevanti per la domanda (nella lingua originale dell'utente). I report irrilevanti restituiscono `NOT_RELEVANT`.
- **REDUCE**: le risposte parziali sono combinate da un'altra invocazione LLM che sintetizza una risposta finale, tenendo conto della history della conversazione.

#### **5. Aggiornamento chat history** (`community_graph_service.py:1032`)
La domanda e la risposta sono aggiunte al dizionario `chat_history_dict` per mantenere il contesto dialogico.

### 4.2 Ruolo degli LLM
- **Fase MAP**: estrazione mirata dal singolo report
- **Fase REDUCE**: sintesi coesa e multilingue

### 4.3 Ruolo degli embedding e del reranking
- Embedding densi e sparse per la retrieval ibrida
- Cross‑encoder per il reranking, che migliora drasticamente la precisione

**File di riferimento:**
- `tilellm/modules/knowledge_graph/controllers.py:327` – endpoint `/qa`
- `tilellm/modules/knowledge_graph/logic.py:358` – logica `query_graph`
- `tilellm/modules/knowledge_graph/services/community_graph_service.py:864` – `query_with_global_search`
- `tilellm/modules/knowledge_graph/utils/rrf.py` – implementazione RRF

## 5. Interrogazione: Integrated Hybrid Search (`/api/kg/hybrid`)

**Endpoint:** `POST /api/kg/hybrid`  
**Scopo:** eseguire una pipeline completa che combina **global search** (sui report), **local search** (sui chunk originali) e **graph expansion** (espansione adattiva dal knowledge graph), seguita da reranking e sintesi LLM. È il "cuore" del sistema GraphRAG.

### 5.1 Flusso dettagliato

```
1. Analisi della query (tipo detection)
   ↓
2. Parallel retrieval:
   - Task A: Global search (report)
   - Task B: Local search (chunk)
   ↓
3. Graph expansion (adaptive multi‑hop)
   ↓
4. Reranking finale (cross‑encoder)
   ↓
5. Sintesi LLM con contesto unificato
   ↓
6. Aggiornamento chat history
```

#### **1. Analisi della query** (`services/services.py`)
Il `GraphRAGService` classifica la domanda in uno di tre tipi:
- **Technical**: richiede precisione, poca espansione (es. "Qual è la formula di...")
- **Exploratory**: richiede esplorazione, media espansione (es. "Parlami di...")
- **Relational**: richiede profondità delle relazioni, massima espansione (es. "Quali sono le connessioni tra X e Y?")

I pesi tra ricerca vettoriale, keyword e grafo sono regolati di conseguenza.

#### **2. Parallel retrieval** (`community_graph_service.py:1286`)
Vengono lanciate **in parallelo** due ricerche:

- **Task A: Global search**  
  Identico a `/api/kg/qa`, ma limitato ai top‑5 report.

- **Task B: Local search**  
  Ricerca ibrida (densa + keyword) sul namespace originale dei chunk, con alpha determinato dai pesi. I risultati sono fusi con RRF.

#### **3. Graph expansion (adaptive multi‑hop)** (`utils/graph_expansion.py:39`)
Dai chunk locali più rilevanti si estraggono gli ID dei **nodi seed** collegati. Il servizio `GraphExpander` esegue un'espansione **adattiva**:

- **Numero di hop dinamico**:  
  `technical=1`, `exploratory=2`, `relational=3` (configurabile in `HOP_CONFIG`)

- **Early stopping**:  
  Se un hop produce pochi nuovi nodi (< `MIN_NEW_NODES_PER_HOP`) o si raggiunge il limite assoluto di nodi (`MAX_NODES_ABSOLUTE`), l'espansione si ferma.

- **Cypher query ottimizzata**:  
  Recupera nodi e relazioni filtrando per `namespace`/`index_name` e ordinando per peso delle relazioni.

L'espansione trasforma i frammenti isolati in una **narrazione coerente**, catturando le connessioni tra entità.

#### **4. Reranking finale** (`community_graph_service.py:1480`)
Tutti i contesti (report globali, chunk locali, entità del grafo) sono uniti in una lista di `Document` LangChain e **rerankati** da un cross‑encoder, selezionando i 20 più pertinenti.

#### **5. Sintesi LLM** (`community_graph_service.py:1496`)
Il contesto finale viene passato a un LLM insieme alla history della conversazione, con l'istruzione di produrre una risposta nella lingua dell'utente, integrando le diverse fonti (report, documenti, grafo).

#### **6. Aggiornamento chat history** (`community_graph_service.py:1535`)
La domanda e la risposta sono aggiunte alla history per mantenere il contesto.

### 5.2 Ruolo degli LLM
- **Classificazione del tipo di query**
- **Sintesi finale del contesto** (integrazione delle tre fonti)
- (Opzionale) generazione di domande sintetiche durante la creazione dei report

### 5.3 Ruolo degli embedding
- **Embedding densi** per la ricerca vettoriale
- **Embedding sparse** (SPLADE) per la ricerca keyword
- **Embedding dei report** per la global search

### 5.4 Ruolo del reranking
- **Cross‑encoder** (es. `TileReranker`) migliora l'ordinamento dei risultati prima della sintesi LLM.

### 5.5 Ruolo del knowledge graph (Neo4j)
- Memorizza entità e relazioni estratte
- Fornisce il meccanismo di **adaptive graph expansion** per recuperare connessioni rilevanti a partire dai seed

**File di riferimento:**
- `tilellm/modules/knowledge_graph/controllers.py:344` – endpoint `/hybrid`
- `tilellm/modules/knowledge_graph/logic.py:407` – logica `context_fusion_graph_search`
- `tilellm/modules/knowledge_graph/services/community_graph_service.py:1206` – `context_fusion_search`
- `tilellm/modules/knowledge_graph/utils/graph_expansion.py:39` – `expand_from_seeds` (adaptive expansion)

## 6. Componenti chiave

### 6.1 Adaptive Graph Expansion (`utils/graph_expansion.py`)
**Definizione:** espansione multi‑hop che adatta il numero di hop in base al tipo di query.

**Configurazione hop:**
```python
HOP_CONFIG = {
    'technical': 1,      # Query precise → 1 hop (contesto focalizzato)
    'exploratory': 2,    # Query esplorative → 2 hop (contesto ampio)
    'relational': 3      # Query relazionali → 3 hop (profondità connessioni)
}
```

**Early stopping:**
- Se un hop produce meno di `MIN_NEW_NODES_PER_HOP` (default: 3) nuovi nodi, l'espansione si ferma
- Limite assoluto: `MAX_NODES_ABSOLUTE` (default: 200) nodi totali

**Cypher query ottimizzata:** filtra per namespace, index_name e ordina per peso delle relazioni.

### 6.2 Reciprocal Rank Fusion (RRF) (`utils/rrf.py`)
**Scopo:** fondere le ranked list di ricerca densa e sparse in una lista unica robusta.

**Formula:** `score = 1 / (k + rank)` per ciascun risultato, somma dei punteggi tra le liste.

**Vantaggi:** bilancia recall (sparse) e precisione (dense), migliorando la qualità complessiva della retrieval.

### 6.3 Synthetic QA Enrichment (`utils/synthetic_qa.py`)
**Scopo:** arricchire i report delle community con domande sintetiche generate da LLM.

**Flusso:**
1. Per ogni report, l'LLM genera 3 domande esemplificative
2. Le domande sono incluse nel metadata del report
3. Durante l'indexing, il contenuto del report + domande viene embedded

**Effetto:** migliora la retrievability semantica, soprattutto per query formulate in modo diverso dal contenuto del report.

### 6.4 Query Type Detection (`utils/query_analysis.py`)
**Scopo:** classificare la query per regolare pesi di ricerca e strategia di espansione.

**Metodi:**
- **LLM‑based:** prompt all'LLM per classificazione
- **Heuristic fallback:** analisi lessicale (parole chiave tecniche, interrogativi, ecc.)

**Pesi regolati:**
- **Technical:** più peso a ricerca vettoriale, meno a grafo
- **Exploratory:** bilanciato tra vettoriale, keyword e grafo
- **Relational:** più peso al grafo, espansione profonda

## 7. Diagrammi di flusso

### 7.1 Creazione Knowledge Graph
```
[Vector Store]
    ↓ (recupera chunk)
[GraphRAG Service]
    ↓ (estrai entità/relazioni)
[Neo4j Import]
    ↓ (salva nodi+archi)
[Cluster Service]
    ↓ (Leiden clustering)
[LLM Report Generation]
    ↓ (genera report)
[Synthetic QA]
    ↓ (aggiungi domande)
[Vector Store Indexing]
    ↓ (embedding report)
[Parquet Export]
    ↓ (upload MinIO)
```

### 7.2 Integrated Hybrid Search
```
[User Question]
    ↓
[Query Type Detection] → (technical/exploratory/relational)
    ↓
[Parallel Retrieval]
├─ Global Search (report) → [RRF] → [Reranking]
└─ Local Search (chunk) → [RRF] → [Reranking]
    ↓
[Graph Expansion] → (adaptive multi‑hop)
    ↓
[Context Fusion] → (unisci report+chunk+graph)
    ↓
[Cross‑Encoder Reranking] → (top‑20 pertinenti)
    ↓
[LLM Synthesis] → (risposta finale)
    ↓
[Update Chat History]
```

## 8. Esempi di utilizzo

### 8.1 Creazione grafo
```bash
curl -X POST http://localhost:8000/api/kg/create \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "bancaitalia",
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "***",
      "index_name": "tilellm"
    },
    "limit": 1000,
    "overwrite": true,
    "llm_key": "openai",
    "model": "gpt-4"
  }'
```

### 8.2 Hybrid Search
```bash
curl -X POST http://localhost:8000/api/kg/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quali sono le relazioni tra Mario Draghi e la BCE?",
    "namespace": "bancaitalia",
    "engine": {
      "name": "pinecone",
      "type": "serverless",
      "apikey": "***",
      "index_name": "tilellm"
    },
    "search_type": "hybrid",
    "use_reranking": true,
    "llm_key": "openai",
    "model": "gpt-4"
  }'
```

## 9. Riferimenti ai file sorgente

| Componente | File | Righe rilevanti |
|------------|------|-----------------|
| **Endpoint `/create`** | `controllers.py` | 213-224 |
| **Logica `create_graph`** | `logic.py` | 216-254 |
| **`create_community_graph`** | `community_graph_service.py` | 144-245 |
| **Clustering & report** | `clustering.py` | varie |
| **Synthetic QA** | `synthetic_qa.py` | tutte |
| **Endpoint `/qa`** | `controllers.py` | 327-341 |
| **`query_with_global_search`** | `community_graph_service.py` | 864-1048 |
| **RRF** | `rrf.py` | tutte |
| **Endpoint `/hybrid`** | `controllers.py` | 344-369 |
| **`context_fusion_search`** | `community_graph_service.py` | 1206-1561 |
| **`GraphExpander`** | `graph_expansion.py` | 39-208 |
| **Query analysis** | `query_analysis.py` | tutte |

## 10. Conclusioni

Il modulo **Knowledge Graph** di Tiledesk LLM realizza un'architettura GraphRAG completa e modulare che:
1. **Costruisce automaticamente** un knowledge graph da documenti esistenti nel vector store
2. **Genera report di community** gerarchici arricchiti con domande sintetiche
3. **Offre tre modalità di interrogazione** progressive (global, hybrid, full expansion)
4. **Adatta dinamicamente** la strategia di retrieval in base al tipo di query
5. **Integra multiple tecniche** di IR: embedding densi/sparsi, RRF, cross‑encoder reranking, adaptive graph expansion

Il sistema è progettato per essere estensibile con nuovi algoritmi di clustering, strategie di expansion e modelli di reranking, mantenendo l'integrazione con l'ecosistema Tiledesk LLM (Pinecone, Qdrant, OpenAI, Azure, ecc.).

---

*Documento aggiornato al: 02 Gennaio 2026*  
*Modulo: `tilellm/modules/knowledge_graph`*