# Roadmap — Tiledesk LLM

> Aggiornamento: 2026-03-16

## Stato Attuale — Cosa è già fatto

| Area | Componente | Note |
|------|-----------|------|
| **Ingestion** | `add_item` (Pinecone serverless/pod, Qdrant, Milvus) | url, pdf, docx, txt, md, xlsx, xls, csv |
| **Ingestion** | `add_item_hybrid` (Pinecone serverless, Qdrant, Milvus) | dense + sparse vectors |
| **Ingestion** | `pdf_ocr` pipeline (Docling + MinIO) | testo, tabelle, immagini con OCR |
| **Loader** | `StructuredDocxLoader` | heading_path, tabelle markdown, URL download |
| **Loader** | `ExcelLoader` / `CSVLoader` | sheet-by-sheet, col_names metadata, chunking verticale |
| **Loader** | Trafilatura per web scraping | fast-path su url, fallback a Unstructured |
| **Retrieval** | Hybrid search (dense + sparse) | Pinecone serverless, Qdrant, Milvus |
| **Qualità** | Situated context (Contextual Retrieval) | `add_item` + `add_item_hybrid` + pdf_ocr ✅ |
| **Metadata** | `CommonChunkMetadata` schema condiviso | tutti i backend, heading_path, ref_tables/images |
| **RAPTOR** | Modulo completo `tilellm/modules/raptor/` | build, retrieve, collapsed_tree, tree_traversal |
| **Compliance** | `tilellm/modules/compliance_checker/` | check, RTM CSV, NL endpoint `/api/compliance/ask` |
| **Agenti** | Workflow LangGraph `/api/v2/qa` | guardia → intent_router → compliance / rag_core |
| **Queue** | TaskIQ + Redis Stream | pdf_ocr asincrono, worker separato |
| **Graph** | FalkorDB knowledge graph | community detection, Louvain/Leiden, QA su grafo |
 
 ---
 
 ## Priorità Alta — Prossime attività
 
 ### 1. Agentic Retrieval (Fase 3)
 
Il retrieval attuale è "flat" (un vettore → top-k). La fase 3 introduce tool atomici e un ReAct loop.

- [ ] Definire `PDFRetrievalTools` con tool atomici:
  - `search_text(query, namespace, doc_id, top_k)`
  - `search_tables(query, namespace, doc_id, top_k)`
  - `search_images(query, namespace, doc_id, top_k)`
  - `get_table_from_minio(table_id, fmt="markdown")` — fetch parquet/md da MinIO
  - `get_adjacent_chunks(chunk_id, window=2)` — chunk precedente/successivo
  - `get_parent_chunk(chunk_id)` — fetch parent (per parent-child, futuro)
- [ ] ReAct loop con LangGraph (max 6-8 step: reason → act → observe → generate)
- [ ] Nuovo endpoint `POST /api/qa/agentic` che usa il retrieval agent
- [ ] Integrazione opzionale dei `raptor_results` nel context del ReAct loop

**Effort stimato:** medio | **Impatto:** alto (risponde meglio a query cross-modal)

---

### 2. Nuovi formati — completare le lacune

#### PPTX
- [ ] `type=pptx` in `ItemSingle` e in tutti i repository
- [ ] `PPTXLoader` in `structured_loaders.py` (python-pptx)
  - slide per slide → testo + note
  - immagini embedded → MinIO + vision caption (opzionale)
  - metadata: `slide_number`, `title`, `layout`

#### Immagini embedded in DOCX
- [ ] `StructuredDocxLoader`: estrai immagini embedded → upload MinIO
- [ ] Vision LLM caption per ogni immagine (come pdf_ocr)
- [ ] `ref_images` nel metadata dei paragrafi adiacenti

#### Sitemap / Crawling batch
- [ ] Nuovo endpoint `POST /api/ingest/sitemap`
  - Riceve URL sitemap → parse → lista URL
  - Dispatch via TaskIQ (un task per URL)
  - Rate limiting configurabile (`crawl_delay_ms`)
- [ ] Supporto `robots.txt` (rispetto del crawl delay)

**Effort stimato:** basso-medio | **Impatto:** medio

---

### 3. Unificazione endpoint ingestion (Fase 4 parziale)

Oggi ci sono due entry point separati: `/api/scrape/single` e `/api/pdf/scrape`. Questo causa duplicazione logica e configurazione inconsistente.

- [ ] `POST /api/ingest` — endpoint unificato con routing automatico per tipo
  - Riceve `ItemSingle`-like con `type` → instrada alla pipeline corretta
  - `pdf` con `use_ocr=true` → `pdf_ocr` pipeline
  - `pdf` senza OCR → `add_item` standard
  - `docx`, `xlsx`, `csv` → loaders strutturati
  - `url` → Trafilatura + fallback
- [ ] Deprecare gradualmente i vecchi endpoint (mantenere backward compat con redirect)

**Effort stimato:** medio | **Impatto:** medio (semplifica l'integrazione per i client)

---

## Priorità Media

### 4. Document Classifier (Fase 4)

Prima dell'ingestion, classificare automaticamente il documento per scegliere la pipeline ottimale.

- [ ] `DocumentClassifier` LangGraph node con output strutturato (`DocumentProfile`)
  - `doc_type`: financial_report | technical_manual | academic_paper | web_article | legal_contract | unknown
  - `modality`: text_only | multimodal | layout_heavy
  - `estimated_complexity`: low | medium | high
  - `recommended_chunk_size`, `use_situated_context`, `use_raptor`, `use_colpali`
- [ ] Integrazione nel workflow di ingestion: classifier → pipeline selection → ingestion
- [ ] Attivazione automatica di RAPTOR per documenti classificati come `high complexity + long`

**Effort stimato:** medio | **Impatto:** alto (pipeline adattiva senza configurazione manuale)

---

### 5. Self-RAG / CRAG (Fase 5)

Aggiungere un loop di auto-correzione al retrieval: se la risposta non è ben fondata, rilancia.

- [ ] **Relevance evaluator**: LLM-as-judge leggero sul retrieval (i chunk recuperati sono rilevanti?)
- [ ] **Answer grounding evaluator**: la risposta generata è supportata dai chunk? (già parzialmente in `hallucination_node`)
- [ ] **Query rewriting**: su fallimento retrieval → riscrive la query e rilancia (max 2 retry)
- [ ] **CRAG**: se nessun chunk è rilevante, fallback a ricerca web (Trafilatura su query Google/Bing)

**Effort stimato:** medio | **Impatto:** alto (riduce hallucination)

---

### 6. Parent-Child Retrieval

- [ ] Indicizzazione a doppio livello: chunk piccoli (50-150 token) per retrieval preciso + chunk grandi (400-800 token) come contesto per il LLM
- [ ] `parent_chunk_id` nel metadata dei chunk piccoli
- [ ] Al retrieval: trovato il piccolo per similarità → fetch del parent da Redis o vector store
- [ ] Valutare compatibilità con `LangChain Multi-Vector Retriever` e i backend supportati (Pinecone/Qdrant/Milvus)

**Effort stimato:** medio | **Impatto:** medio-alto

---

### 7. Tabelle HTML da web scraping

Le tabelle nelle pagine web sono attualmente ignorate (estratto solo testo).

- [ ] Post-processing HTML: rilevare `<table>` → pandas DataFrame → markdown
- [ ] Trattamento identico a tabelle PDF: descrizione LLM + upload MinIO + metadata `col_names`
- [ ] Aggiungere al pipeline Trafilatura (come fallback se la tabella non è in output)

**Effort stimato:** basso | **Impatto:** medio

---

## Priorità Bassa / Futuro

### 8. ColPali — PDF visually-rich (Fase 6)

- [ ] Valutare hosting locale ColPali (~7B params) vs. API esterna
- [ ] Se fattibile: path alternativo per PDF con layout visivamente complessi (grafici, schemi, form scansionati)
- [ ] Attivazione da `document_classifier` quando `modality = layout_heavy`

**Blocco:** decisione infrastrutturale sul hosting del modello.

---

### 9. Audio / Video

- [ ] `type=audio` / `type=video` → trascrizione con Whisper (locale o API)
- [ ] Output trascrizione → pipeline testo standard (chunking, situated context, ecc.)
- [ ] Metadata: `timestamp_start`, `timestamp_end` per ogni chunk → link temporale al media

---

### 10. MCP Integration (Model Context Protocol)

- [ ] Tools MCP per accesso a sistemi aziendali: database, ERP, CRM, API REST
- [ ] Usabili dal ReAct loop (Fase 3) come tool aggiuntivi
- [ ] Esempi: `query_database(sql)`, `call_api(endpoint, params)`, `get_document_from_cms(id)`

---

### 11. Miglioramenti infrastrutturali

| Item | Dettaglio |
|------|-----------|
| **Embedding dimension auto-detection** | Oggi hardcoded in alcuni punti (es. Pinecone pod). Rendere dinamico. |
| **Validation agent** (Fase 4) | Dopo ingestion: verifica coverage (tutti i chunk indicizzati?), auto-retry su fallimento parziale |
| **Auto-escalation scraping** | `scrape_type 0 → 3 → 5` automatico su fallimento (JS detection, stealth mode) |
| **Benchmark retrieval** | Suite di test end-to-end su dataset noti (DocVQA, ViDoRe) per misurare impatto di ogni feature |
| **Monitoring / observability** | Dashboard LLM costs, embedding costs, retrieval latency per namespace |
| **Tenant namespace design** | Chiarire se `doc_id` è globalmente unico o serve `{tenant_id}/{doc_id}` |

---

## Domande Aperte Bloccanti
1. **ColPali hosting**: `~7B params` — hosting locale fattibile o API esterna? Determina se Fase 6 è concreta.
2. **Unificazione namespace**: `{client_namespace}_{doc_id}` o namespace flat? Impatta tutto il retrieval multi-doc.
3. **Situated context cost**: il costo LLM per chunk è accettabile in produzione? O serve una versione leggera (`heading_path + primi 150 char`)?
4. **Parent-child e Multi-Vector Retriever**: il LangChain `MultiVectorRetriever` usa un docstore Redis — è compatibile con tutti e tre i vector store backend?

---
## Dipendenze tra Aree

```
Formato PPTX
    │
    ▼
add_item / add_item_hybrid  ←── Document Classifier (attiva pipeline giusta)
    │                                   │
    ▼                                   ▼
situated_context               RAPTOR (già ✅)
    │
    ▼
Vector Store (Pinecone / Qdrant / Milvus)
    │
    ▼
Agentic Retrieval (ReAct loop + PDFRetrievalTools)
    │
    ▼
Self-RAG / CRAG (loop di auto-correzione)
    │
    ▼
/api/v2/qa (LangGraph workflow — già ✅)
```

---

## Riepilogo priorità


| # | Area | Effort | Impatto | Blocchi |
|---|------|--------|---------|---------|
| 1 | Agentic Retrieval (ReAct + tools) | M | Alto | — |
| 2 | PPTX loader | B | Medio | — |
| 3 | Sitemap / crawling batch | B-M | Medio | — |
| 4 | Unificazione `/api/ingest` | M | Medio | — |
| 5 | Document Classifier | M | Alto | — |
| 6 | Self-RAG / CRAG | M | Alto | — |
| 7 | Parent-child retrieval | M | M-Alto | Valutare LangChain MultiVectorRetriever |
| 8 | Tabelle HTML da web | B | Medio | — |
| 9 | Immagini embedded DOCX | B-M | Medio | — |
| 10 | ColPali | Alto | Alto | Decisione hosting |
| 11 | Audio/Video (Whisper) | M | Medio | — |
| 12 | MCP integration | M-Alto | Medio | — |

> **Legenda effort:** B = bassa (1-3 gg), M = media (1-2 sett), Alto = > 2 sett

