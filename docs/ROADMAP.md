# Roadmap — Tiledesk LLM

> Update: 2026-03-16

## Current State — What's Already Done

| Area | Component | Notes |
|------|-----------|-------|
| **Ingestion** | `add_item` (Pinecone serverless/pod, Qdrant, Milvus) | url, pdf, docx, txt, md, xlsx, xls, csv |
| **Ingestion** | `add_item_hybrid` (Pinecone serverless, Qdrant, Milvus) | dense + sparse vectors |
| **Ingestion** | `pdf_ocr` pipeline (Docling + MinIO) | text, tables, images with OCR |
| **Loader** | `StructuredDocxLoader` | heading_path, markdown tables, download URL |
| **Loader** | `ExcelLoader` / `CSVLoader` | sheet-by-sheet, col_names metadata, vertical chunking |
| **Loader** | Trafilatura for web scraping | fast-path on url, fallback to Unstructured |
| **Retrieval** | Hybrid search (dense + sparse) | Pinecone serverless, Qdrant, Milvus |
| **Quality** | Situated context (Contextual Retrieval) | `add_item` + `add_item_hybrid` + pdf_ocr ✅ |
| **Metadata** | `CommonChunkMetadata` shared schema | all backends, heading_path, ref_tables/images |
| **RAPTOR** | Complete module `tilellm/modules/raptor/` | build, retrieve, collapsed_tree, tree_traversal |
| **Compliance** | `tilellm/modules/compliance_checker/` | check, RTM CSV, NL endpoint `/api/compliance/ask` |
| **Agents** | LangGraph Workflow `/api/v2/qa` | guard → intent_router → compliance / rag_core |
| **Queue** | TaskIQ + Redis Stream | async pdf_ocr, separate worker |
| **Graph** | FalkorDB knowledge graph | community detection, Louvain/Leiden, QA on graph |

---

## High Priority — Next Activities

### 1. Agentic Retrieval (Phase 3)

Current retrieval is "flat" (one vector → top-k). Phase 3 introduces atomic tools and a ReAct loop.

- [ ] Define `PDFRetrievalTools` with atomic tools:
  - `search_text(query, namespace, doc_id, top_k)`
  - `search_tables(query, namespace, doc_id, top_k)`
  - `search_images(query, namespace, doc_id, top_k)`
  - `get_table_from_minio(table_id, fmt="markdown")` — fetch parquet/md from MinIO
  - `get_adjacent_chunks(chunk_id, window=2)` — previous/next chunks
  - `get_parent_chunk(chunk_id)` — fetch parent (for parent-child, future)
- [ ] ReAct loop with LangGraph (max 6-8 steps: reason → act → observe → generate)
- [ ] New endpoint `POST /api/qa/agentic` using the retrieval agent
- [ ] Optional integration of `raptor_results` in the ReAct loop context

**Estimated effort:** medium | **Impact:** high (better answers to cross-modal queries)

---

### 2. New Formats — Complete the Gaps

#### PPTX
- [ ] `type=pptx` in `ItemSingle` and all repositories
- [ ] `PPTXLoader` in `structured_loaders.py` (python-pptx)
  - slide by slide → text + notes
  - embedded images → MinIO + vision caption (optional)
  - metadata: `slide_number`, `title`, `layout`

#### Embedded Images in DOCX
- [ ] `StructuredDocxLoader`: extract embedded images → upload to MinIO
- [ ] Vision LLM caption for each image (like pdf_ocr)
- [ ] `ref_images` in metadata of adjacent paragraphs

#### Sitemap / Batch Crawling
- [ ] New endpoint `POST /api/ingest/sitemap`
  - Receives sitemap URL → parse → list of URLs
  - Dispatch via TaskIQ (one task per URL)
  - Configurable rate limiting (`crawl_delay_ms`)
- [ ] `robots.txt` support (respect crawl delay)

**Estimated effort:** low-medium | **Impact:** medium

---

### 3. Unify Ingestion Endpoints (Phase 4 partial)

Today there are two separate entry points: `/api/scrape/single` and `/api/pdf/scrape`. This causes logic duplication and inconsistent configuration.

- [ ] `POST /api/ingest` — unified endpoint with automatic routing by type
  - Receives `ItemSingle`-like with `type` → routes to correct pipeline
  - `pdf` with `use_ocr=true` → `pdf_ocr` pipeline
  - `pdf` without OCR → standard `add_item`
  - `docx`, `xlsx`, `csv` → structured loaders
  - `url` → Trafilatura + fallback
- [ ] Gradually deprecate old endpoints (keep backward compat with redirects)

**Estimated effort:** medium | **Impact:** medium (simplifies client integration)

---

## Medium Priority

### 4. Document Classifier (Phase 4)

Before ingestion, automatically classify the document to choose the optimal pipeline.

- [ ] `DocumentClassifier` LangGraph node with structured output (`DocumentProfile`)
  - `doc_type`: financial_report | technical_manual | academic_paper | web_article | legal_contract | unknown
  - `modality`: text_only | multimodal | layout_heavy
  - `estimated_complexity`: low | medium | high
  - `recommended_chunk_size`, `use_situated_context`, `use_raptor`, `use_colpali`
- [ ] Integration in ingestion workflow: classifier → pipeline selection → ingestion
- [ ] Automatic activation of RAPTOR for documents classified as `high complexity + long`

**Estimated effort:** medium | **Impact:** high (adaptive pipeline without manual configuration)

---

### 5. Self-RAG / CRAG (Phase 5)

Add a self-correction loop to retrieval: if the answer is not well-grounded, retry.

- [ ] **Relevance evaluator**: LLM-as-judge on retrieval (are retrieved chunks relevant?)
- [ ] **Answer grounding evaluator**: is the generated answer supported by chunks? (already partially in `hallucination_node`)
- [ ] **Query rewriting**: on retrieval failure → rewrite query and retry (max 2 retries)
- [ ] **CRAG**: if no chunk is relevant, fallback to web search (Trafilatura on Google/Bing query)

**Estimated effort:** medium | **Impact:** high (reduces hallucination)

---

### 6. Parent-Child Retrieval

- [ ] Dual-level indexing: small chunks (50-150 tokens) for precise retrieval + large chunks (400-800 tokens) as context for LLM
- [ ] `parent_chunk_id` in metadata of small chunks
- [ ] At retrieval: found small by similarity → fetch parent from Redis or vector store
- [ ] Evaluate compatibility with `LangChain Multi-Vector Retriever` and supported backends (Pinecone/Qdrant/Milvus)

**Estimated effort:** medium | **Impact:** medium-high

---

### 7. HTML Tables from Web Scraping

Tables in web pages are currently ignored (only text extracted).

- [ ] HTML post-processing: detect `<table>` → pandas DataFrame → markdown
- [ ] Same treatment as PDF tables: LLM description + MinIO upload + `col_names` metadata
- [ ] Add to Trafilatura pipeline (as fallback if table not in output)

**Estimated effort:** low | **Impact:** medium

---

## Low Priority / Future

### 8. ColPali — Visually-rich PDFs (Phase 6)

- [ ] Evaluate local ColPali hosting (~7B params) vs. external API
- [ ] If feasible: alternative path for PDFs with visually complex layouts (charts, diagrams, scanned forms)
- [ ] Activation from `document_classifier` when `modality = layout_heavy`

**Blocker:** infrastructural decision on model hosting.

---

### 9. Audio / Video

- [ ] `type=audio` / `type=video` → transcription with Whisper (local or API)
- [ ] Transcription output → standard text pipeline (chunking, situated context, etc.)
- [ ] Metadata: `timestamp_start`, `timestamp_end` for each chunk → temporal link to media

---

### 10. MCP Integration (Model Context Protocol)

- [ ] MCP tools for accessing enterprise systems: databases, ERP, CRM, REST APIs
- [ ] Usable from ReAct loop (Phase 3) as additional tools
- [ ] Examples: `query_database(sql)`, `call_api(endpoint, params)`, `get_document_from_cms(id)`

---

### 11. Infrastructural Improvements

| Item | Detail |
|------|--------|
| **Embedding dimension auto-detection** | Today hardcoded in some places (e.g. Pinecone pod). Make dynamic. |
| **Validation agent** (Phase 4) | After ingestion: verify coverage (all chunks indexed?), auto-retry on partial failure |
| **Auto-escalation scraping** | `scrape_type 0 → 3 → 5` automatic on failure (JS detection, stealth mode) |
| **Retrieval benchmark** | End-to-end test suite on known datasets (DocVQA, ViDoRe) to measure impact of each feature |
| **Monitoring / observability** | Dashboard LLM costs, embedding costs, retrieval latency per namespace |
| **Tenant namespace design** | Clarify if `doc_id` is globally unique or needs `{tenant_id}/{doc_id}` |

---

## Open Blocking Questions

1. **ColPali hosting**: `~7B params` — local hosting feasible or external API? Determines if Phase 6 is concrete.
2. **Namespace unification**: `{client_namespace}_{doc_id}` or flat namespace? Impacts all multi-doc retrieval.
3. **Situated context cost**: is LLM cost per chunk acceptable in production? Or is a lightweight version needed (`heading_path + first 150 chars`)?
4. **Parent-child and Multi-Vector Retriever**: LangChain's `MultiVectorRetriever` uses a Redis docstore — is it compatible with all three vector store backends?

---

## Dependencies Between Areas

```
PPTX Format
    │
    ▼
add_item / add_item_hybrid  ←── Document Classifier (activates right pipeline)
    │                                   │
    ▼                                   ▼
situated_context               RAPTOR (already ✅)
    │
    ▼
Vector Store (Pinecone / Qdrant / Milvus)
    │
    ▼
Agentic Retrieval (ReAct loop + PDFRetrievalTools)
    │
    ▼
Self-RAG / CRAG (self-correction loop)
    │
    ▼
/api/v2/qa (LangGraph workflow — already ✅)
```

---

## Priority Summary

| # | Area | Effort | Impact | Blockers |
|---|------|--------|--------|----------|
| 1 | Agentic Retrieval (ReAct + tools) | M | High | — |
| 2 | PPTX loader | L | Medium | — |
| 3 | Sitemap / batch crawling | L-M | Medium | — |
| 4 | Unify `/api/ingest` | M | Medium | — |
| 5 | Document Classifier | M | High | — |
| 6 | Self-RAG / CRAG | M | High | — |
| 7 | Parent-child retrieval | M | M-High | Evaluate LangChain MultiVectorRetriever |
| 8 | HTML tables from web | L | Medium | — |
| 9 | Embedded images DOCX | L-M | Medium | — |
| 10 | ColPali | High | High | Hosting decision |
| 11 | Audio/Video (Whisper) | M | Medium | — |
| 12 | MCP integration | M-High | Medium | — |

> **Effort legend:** L = low (1-3 days), M = medium (1-2 weeks), High = > 2 weeks
