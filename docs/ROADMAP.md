# Roadmap — Tiledesk LLM

> Update: 2026-03-28

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
| **RAPTOR** | Complete module `tilellm/modules/raptor/` ✅ v2 | build, retrieve, collapsed_tree, tree_traversal + TaskIQ async + 9 critical bug fixes |
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

### 2. HyDE (Hypothetical Document Embeddings) ✅ IMPLEMENTED

Improve retrieval performance for zero-shot tasks by generating a hypothetical document based on the query.

- [x] Implement HyDE generator node in LangGraph (`tilellm/agents/nodes.py::hyde_node`)
- [x] Integrate HyDE with `POST /api/v2/qa` workflow (inserted between intent_router and rag_core)
- [x] Add configuration options: `use_hyde: bool`, `retrieval_query: Optional[str]` in `QuestionAnswer` model
- [x] Updated `controller_utils.py::contextualize_query` to use `retrieval_query or question`

**Impact:** Better retrieval for complex queries through synthetic document embeddings

---

### 3. New Formats — Complete the Gaps

#### PPTX
- [ ] `type=pptx` in `ItemSingle` and all repositories
- [ ] `PPTXLoader` in `structured_loaders.py` (python-pptx)
  - slide by slide → text + notes
  - embedded images → MinIO + vision caption (optional)
  - metadata: `slide_number`, `title`, `layout`

#### Embedded Images in DOCX ✅ IMPLEMENTATO
- [x] `StructuredDocxLoader`: extract embedded images (`w:drawing` + legacy `w:pict`) → upload to MinIO (`{doc_id}/docx_images/{image_id}.png`)
- [x] Vision LLM caption for each image via `generate_image_caption` (same flow as pdf_ocr)
- [x] `ref_images` in metadata of adjacent paragraphs (±1 para_index proximity)
- [x] New pipeline `process_docx_with_images` in `tilellm/modules/ingestion/docx_processor.py` (DI-decorated, skip_delete pattern, situated context support)
- [x] Routing: `POST /api/ingestion` with `type=docx + use_ocr=True` → docx image pipeline

#### Direct Text Auto-Detection ✅ IMPLEMENTED
- [x] New module `tilellm/modules/ingestion/text_processor.py` with auto-detection logic
- [x] **Markdown detection** (starts with `#`, heading patterns) → `MarkdownChunker` with heading hierarchy awareness
- [x] **Tabular detection** (pipe `|` patterns) → table extraction, headers+rows parsing, preserve formatting
- [x] **Plain text fallback** → `RecursiveCharacterTextSplitter` standard chunking
- [x] Table extraction: `_parse_pipe_table()` returns structured `{headers, rows, alignment, row_count}`
- [x] Integrated in all repositories: Pinecone, Qdrant, Milvus (both `add_item` and `add_item_hybrid`)
- [x] Graceful fallback to plain text if format detection/processing fails

#### Sitemap / Batch Crawling
- [ ] New endpoint `POST /api/ingest/sitemap`
  - Receives sitemap URL → parse → list of URLs
  - Dispatch via TaskIQ (one task per URL)
  - Configurable rate limiting (`crawl_delay_ms`)
- [ ] `robots.txt` support (respect crawl delay)

**Estimated effort:** low-medium | **Impact:** medium

---

### 3. Unify Ingestion Endpoints (Phase 4 partial) — IN PROGRESS

Today there are two separate entry points: `/api/scrape/single` and `/api/pdf/scrape`. This causes logic duplication and inconsistent configuration.

- [x] **New Router:** `POST /api/ingestion` — unified entry point that routes requests to the appropriate pipeline based on document type and configuration (pdf+OCR → Docling, hybrid → add_item_hybrid, default → add_item).
- [ ] Gradually deprecate old endpoints (keep backward compat with redirects)
- [ ] Implement support for all document types through the unified router (PPTX, sitemap, etc.)

---

## Medium Priority

### 5. Document Classifier (Phase 4)

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

### 7. HTML Tables from Web Scraping ✅ IMPLEMENTED

Tables in web pages are now extracted as separate `Document` objects with structured metadata.

- [x] `_extract_html_tables(html, url)` in `tilellm/tools/document_tools.py`: BeautifulSoup finds all `<table>` elements → `pd.read_html()` → DataFrame → markdown (`df.to_markdown()`)
- [x] `col_names` metadata (comma-separated headers), `element_type="table"`, `table_index`
- [x] Integrated in all scraping paths: Trafilatura (scrape_type 0/1), Playwright+BS4 (scrape_type 2), Playwright+stealth (scrape_type 5), fallback selectors
- [ ] LLM semantic description for tables (future: same as PDF tables via `TableSemanticLinker`)
- [ ] MinIO upload of table content as parquet/markdown (future: same as PDF tables)

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

## OCR Engine Analysis

### Engines currently in use

| Engine | Where | Notes |
|--------|-------|-------|
| **RapidOCR** | Docling default (`do_ocr=True` in `_init_docling`) | Fast, CPU-friendly, 80+ languages. Log silenced: `logging.getLogger("RapidOCR").setLevel(logging.ERROR)`. |
| **PyPDFLoader** | `add_item` pipeline (standard, no `use_ocr`) | Text-only extraction, no layout / table / image support. Emits a warning recommending Docling for complex PDFs. |

### Other OCR engines available via Docling (not yet exposed)

| Engine | Import | Strengths | How to enable |
|--------|--------|-----------|---------------|
| **EasyOCR** | `EasyOcrOptions` | GPU-accelerated, good for handwriting and complex layouts | Pass `ocr_options=EasyOcrOptions()` to `PdfPipelineOptions` |
| **Tesseract** | `TesseractOcrOptions` | Battle-tested, 100+ languages, configurable PSM | Pass `ocr_options=TesseractOcrOptions(lang="ita+eng")` |
| **Tesseract CLI** | `TesseractCliOcrOptions` | Same as above, shell-based | Alternative Tesseract backend |

**To expose engine selection**: add `ocr_engine: Literal["rapidocr", "easyocr", "tesseract"] = "rapidocr"` to `PDFScrapingRequest` and pass the right `OcrOptions` instance when building `PdfPipelineOptions` in `_init_docling` (currently hardcoded at `__init__` time — should move to per-request).

### Advanced methods implemented in pdf_ocr pipeline (extract_md_simple=False)

| Method | Status | Detail |
|--------|--------|--------|
| Docling full extraction (text+tables+images+formulas) | ✅ | `_process_pdf_docling` in `docling_processor.py` |
| MinIO upload for images (PNG) and tables (Parquet + MD) | ✅ | `_process_single_image`, `_process_tables_batch` |
| Vision LLM caption for images | ✅ | `generate_image_caption` via `@inject_llm_chat_async` |
| LLM semantic description for tables | ✅ | `TableSemanticLinker.link_table_to_context` |
| Synthetic Q&A for tables | ✅ | Generated in `TableSemanticLinker` |
| Cross-modal refs (ref_tables, ref_images, surrounding_text) | ✅ | `_compute_cross_modal_refs` — bbox Euclidean proximity, top-3 per page |
| Contextual Retrieval (situated context) | ✅ | `enrich_chunks_with_situated_context` applied to text chunks |
| `CommonChunkMetadata` for all chunk types | ✅ | Consistent metadata across text / tables / images |
| skip_delete (prevent namespace collision) | ✅ | `_first_index_done` flag in `process_pdf_document_with_embeddings` |
| DuckDB metadata index for tables | ✅ | `document_tables_metadata` table in `docling_processor.py` |
| ContextAwareChunker (hierarchy-aware) | ✅ | Used when `extract_structure=True` |
| LangGraph Markdown extraction agent | ✅ | `extract_md_simple=True` path |

### Known issues in pdf_ocr pipeline (extract_md_simple=False)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | `_get_surrounding_text` stub | `docling_processor.py:634` returns `"Element on page N"` as initial caption context | LOW — `_compute_cross_modal_refs` overrides `surrounding_text` in metadata later; caption quality slightly reduced |
| 2 | OCR always on | `_init_docling` sets `do_ocr=True` unconditionally at init time, ignoring request's `use_ocr` | LOW — `use_ocr` is only used for routing (pdf → Docling), not Docling config; overhead for text-only PDFs |
| 3 | `_init_neo4j` always attempts connection | `docling_processor.__init__` tries Neo4j even when KG is disabled | LOW — fails gracefully (`self.graph_repository = None`), but logs errors on every request if Neo4j not deployed |
| 4 | `process_from_minio` hardcodes `.pdf` suffix | `docling_processor.py:181` `suffix=".pdf"` | LOW — only matters if MinIO-based ingestion is used for non-PDF files |

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
| 8 | ~~HTML tables from web~~ ✅ | — | — | Done |
| 9 | ~~Embedded images DOCX~~ ✅ | — | — | Done |
| 10 | ColPali | High | High | Hosting decision |
| 11 | Audio/Video (Whisper) | M | Medium | — |
| 12 | MCP integration | M-High | Medium | — |

> **Effort legend:** L = low (1-3 days), M = medium (1-2 weeks), High = > 2 weeks
