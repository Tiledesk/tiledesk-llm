# **TILEDESK LLM - Changelog**

### **Authors**:
    * Gianluca Lorenzo
    * Andrea Sponziello
### **Copyright**: Tiledesk SRL


---
## [2026-04-11]
### 0.10.1-rc1 (local inference performance — Fase 1)
- Added: `tilellm/tools/_gpu_concurrency.py` — shared primitives for local GPU inference. Exposes `GLOBAL_GPU_LOCK` (single process-wide `threading.Lock` that serializes forward pass across **all** local models — reranker + sparse encoders — to prevent concurrent-allocation OOMs on the same CUDA device), `is_cuda_oom(exc)`, `cuda_empty_cache_safe()`, and `build_token_budget_batches()` (TEI-inspired greedy packing: sort by length + pack while `batch_size * max_len <= budget`).
- Changed: `tilellm/tools/reranker.py` — `TileReranker._gpu_lock` now aliases `GLOBAL_GPU_LOCK` (was a local `Lock()`). `build_token_budget_batches` moved to `_gpu_concurrency.py` and re-exported from `reranker.py` for backward compatibility with `tests/unit/tools/test_local_reranker_batching.py`. No behavioral changes to the OOM-recovery path — existing tests unchanged.
- Added: `TiledeskSpladeEncoder` (local path) — **fp16 on CUDA at load** (`self.splade.model.half()`, ~2× speedup, halved VRAM), **warmup dummy predict** on load (moves cuDNN autotune off the hot path), **token-budget batching** with `_run_sparse_batched_with_oom_recovery`, **`GLOBAL_GPU_LOCK`** serializing every forward pass (including `encode_queries`), **OOM recovery** with halving budget and max 3 retries. New ctor params: `token_budget_per_batch=2048`, `max_items_per_batch=16`, `max_oom_retries=3`. Removed `torch.cuda.empty_cache()` from the happy path (was called after every batch — 100–200 ms per call of wasted sync).
- Added: `TiledeskBGEM3` (local path) — **warmup** on load (fp16 already handled internally by FlagEmbedding via `use_fp16=True`), **token-budget batching** + **`GLOBAL_GPU_LOCK`** + **OOM recovery** (same pattern as Splade). New ctor params: `token_budget_per_batch=2048`, `max_items_per_batch=8` (lower default — BGE-M3 is heavier), `max_oom_retries=3`. Removed `torch.cuda.empty_cache()` from the happy path.
- Fixed: `TiledeskSparseEncoders._get_cached_encoder` — LRU eviction now properly releases VRAM (`del oldest_encoder` + `cuda_empty_cache_safe()`). Previously `OrderedDict.pop` only dropped the Python reference while PyTorch retained the CUDA allocation until the next GC cycle, causing gradual VRAM leaks across model swaps.
- Fixed: `TiledeskSparseEncoders.clear_cache` — now clears inside `_cache_lock` (was racing with concurrent `_get_cached_encoder` calls) and uses `cuda_empty_cache_safe()` (previously crashed if `torch` import had failed because the `and` chain referenced `torch.cuda` before the None check).
- Refactored: `_chunk_text` now takes `max_words` parameter (Splade uses 300 words / ~390 tokens given 512 ctx; default kept at 300 for other callers). Extracted `_estimate_text_tokens` helper (~1.3 tokens/word heuristic, same as reranker).
- Expected impact: local reranker latency ~3–4 s → ~1–1.5 s (fp16 speedup + no `empty_cache` per batch); elimination of OOM failures under concurrent upserts (single lock serializes all forward passes); stable p99 latency under concurrency. `autocast`/`bfloat16` considered but not adopted — for our models (bge-reranker-v2-m3, Splade, BGE-M3) pure `.half()` is strictly better (same speedup, half the memory, no runtime casting). Documented as fallback if numerical issues arise.
- Note: Remote paths unaffected — `TEIReranker`, `PineconeReranker`, `TEISparseEncoder` do not touch local GPU and inherit no changes.
- Added: Semantic cache for recurring queries (`tilellm/shared/cache/semantic_cache.py`). Two-level architecture: L1 exact match (SHA-256 hash, 24h TTL) and L2 semantic match (cosine similarity on query embeddings computed in Python with numpy, 6h TTL). No additional infrastructure required — uses the existing Redis instance already in use by TaskIQ.
- Added: `use_cache: bool = False` field to `QuestionAnswer` (`tilellm/models/llm.py`). Cache is opt-in per request; disabled by default.
- Added: `cache_hit: Optional[bool]` and `cache_similarity: Optional[float]` fields to `GraphState` (`tilellm/models/graph_state.py`).
- Added: `cache_lookup_node` and `cache_store_node` LangGraph nodes in `tilellm/agents/nodes.py`. `cache_lookup` is placed after `intent_router` (QA path only); `cache_store` is placed after `validatore` on successful response. On cache hit, `rag_core`/`raptor`/LLM are fully bypassed.
- Added: Cache integration in `POST /api/qa` (`tilellm/__main__.py`) — same L1+L2 lookup/store logic for the non-agentic endpoint.
- Added: Cache invalidation hooks in `add_item` and `add_item_hybrid` (`tilellm/controller/controller.py`) — strategy B (full namespace invalidation) on any document update.
- Added: Cache invalidation in `DELETE /api/namespace/{namespace}` (`tilellm/__main__.py`).
- Added: `GET /api/cache/stats` and `DELETE /api/cache/namespace/{namespace}` admin endpoints.
- Added: Unit tests in `tests/unit/shared/test_semantic_cache.py` covering exact/semantic hit/miss, case-insensitive normalization, namespace invalidation isolation, stats, and `use_cache=False` bypass (uses `fakeredis`).
- Docs: `docs/CACHE_IMPLEMENTATION_PLAN.md` updated to reflect implementation status.
- Docs: `API_DOCUMENTATION.md` — added Semantic Cache APIs section; updated `QuestionAnswer` fields.
- Config: Threshold and TTL configurable via env vars: `CACHE_SIMILARITY_THRESHOLD` (default 0.90), `CACHE_TTL_EXACT` (default 86400), `CACHE_TTL_SEMANTIC` (default 21600).

---
## [2026-04-07] (3)
### 0.10.0-rc8 (evaluation)
- Added: `evaluation/` standalone benchmark module (Opzione B — separato da `tests/`). Install with `pip install tilellm[evaluation]`.
- Added: `evaluation/config.py` — `EvalConfig` dataclass with all pipeline flags + `matrix_configs()` helper for the 8-run benchmark matrix.
- Added: `evaluation/datasets/` — loaders for HotpotQA (`hotpotqa_loader.py`), QASPER (`qasper_loader.py`), MS MARCO (`msmarco_loader.py`) via HuggingFace `datasets`. Shared `EvalSample` + `Passage` models in `base.py`.
- Added: `evaluation/pipeline/ingest.py` — async ingest of passages via `POST /api/ingestion` with configurable concurrency (default 10).
- Added: `evaluation/pipeline/query.py` — async querying via `POST /api/v2/qa` with tqdm progress bar, fills `sample.answer` + `sample.contexts`.
- Added: `evaluation/pipeline/cleanup.py` — `DELETE /api/namespace/{ns}/{token}` after each run.
- Added: `evaluation/evaluate.py` — RAGAS evaluation (5 metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness) + `EvalReport` with RAG Score (harmonic mean) + JSON persistence.
- Added: `evaluation/compare.py` — loads multiple `EvalReport` JSON files and renders a markdown comparison table sorted by RAG Score.
- Added: `evaluation/run_benchmark.py` — CLI entrypoint: `python -m evaluation.run_benchmark [--matrix] [--dataset] [--hybrid] [--situated] [--hyde] [--raptor] ...`
- Added: `ragas>=0.2.0`, `datasets>=2.0.0`, `tqdm>=4.0.0` as optional deps under `[evaluation]` extra in `pyproject.toml`.
- Docs: `docs/RAGAS_EVALUATION_PLAN.md` — all phases marked as implemented.

---
## [2026-04-07] (2)
### 0.10.0-rc8 (patch)
- Fixed: `create_embedding_instance` returns a tuple `(model, dimension)` — unpacked correctly in `cache_lookup_node`, `cache_store_node`, and `/api/qa` cache blocks (`embedding_model, _ = await create_embedding_instance(...)`).
- Added: L4 embedding cache (`tilellm/shared/cache/embedding_cache.py`) — `CachedEmbeddings` wraps any LangChain `Embeddings` object, caches `aembed_query` results in Redis with key `sha256(text)`, TTL configurable via `CACHE_TTL_EMBEDDING` (default 3600s). Integrated transparently in `create_embedding_instance`.
- Added: pdf_ocr cache invalidation hook in `tilellm/modules/task_executor/tasks.py` — `SemanticCache.invalidate_namespace` called after `process_pdf_document_with_embeddings` completes successfully (before webhook notification).
- Added: Prometheus metrics module `tilellm/shared/cache/metrics.py` with counters/histograms: `tiledesk_cache_requests_total`, `tiledesk_cache_lookup_duration_seconds`, `tiledesk_cache_store_duration_seconds`, `tiledesk_cache_invalidations_total`, `tiledesk_embedding_cache_requests_total`. Graceful no-op fallback if `prometheus-client` not installed.
- Added: `GET /metrics` endpoint — standard Prometheus scrape endpoint (excluded from OpenAPI schema).
- Added: `prometheus-client ^0.21.0` to `pyproject.toml` dependencies.
- Docs: `docs/CACHE_IMPLEMENTATION_PLAN.md` — all 7 implementation phases now marked complete.

---
## [2026-04-02]
### 0.10.0-rc7
- Fixed: Hybrid upsert failing for text content — `MetadataItem.model_dump()` included `namespace: None` in vector metadata causing Pinecone `400 Bad Request`. Applied `exclude_none=True`.

---
## [2026-03-26]
### 0.10.0-rc6
- Added: `_extract_html_tables(html, source_url)` in `tilellm/tools/document_tools.py` — extracts all `<table>` elements from raw HTML using BeautifulSoup, converts each to a pandas DataFrame via `pd.read_html()`, renders as markdown (`df.to_markdown()` with `df.to_string()` fallback), and returns one `Document` per table with metadata: `element_type="table"`, `col_names` (comma-separated headers), `table_index`, `type="url"`. Tables with no data rows and tables where pandas fails to parse are silently skipped.
- Added: HTML table extraction integrated into all scraping paths: Trafilatura (`scrape_type=0/1`, appended to text doc), Playwright+BS4 (`scrape_page`, `scrape_type=2`), Playwright+stealth (`scrape_page_complex`, `scrape_type=5`), and fallback selectors (`scrape_page_fallback_selectors`). In each path the raw HTML is available before the BS4/Trafilatura text-only transform, so table structure is preserved.
- Docs: `docs/ROADMAP.md` — marked "HTML Tables from Web Scraping" as implemented; listed future TODOs (LLM semantic description, MinIO parquet upload — same as PDF tables).

---
## [2026-03-26]
### 0.10.0-rc5
- Added: Embedded image extraction for DOCX documents. `StructuredDocxLoader` now exposes `load_with_images()` returning `(documents, image_records)` and `_extract_images(doc)` which scans all paragraphs for modern `w:drawing` (via `a:blip r:embed`) and legacy `w:pict` (via `v:imagedata r:id`) image relationships. Each image is identified by `docx_img_{para_index}_{md5[:8]}`.
- Added: `tilellm/modules/ingestion/docx_processor.py` — async pipeline `process_docx_with_images()` decorated with `@inject_llm_chat_async` + `@inject_repo_async`. Uploads images to MinIO (`{doc_id}/docx_images/`), generates vision LLM captions via `generate_image_caption` (same function used by pdf_ocr), populates `ref_images` in metadata of paragraphs within ±1 para_index, applies situated context if enabled, then indexes image captions + text/tables to vector store with the skip_delete pattern.
- Added: Routing `type=docx + use_ocr=True` in `POST /api/ingestion` → `process_docx_with_images`. `_build_pdf_request` updated to also map `llm_provider → llm` and `llm_model → model` (required by `@inject_llm_chat_async`).
- Refactored: `StructuredDocxLoader` — extracted `_open_doc()`, `_extract_documents(doc)` private helpers; `load()` backward-compatible; paragraphs now carry `_para_index` internal metadata key for cross-referencing with images.
- Docs: `docs/ROADMAP.md` — marked Embedded Images in DOCX as done; added OCR Engine Analysis section documenting RapidOCR (current default), EasyOCR, Tesseract options via Docling, and a review of all advanced methods implemented in the pdf_ocr pipeline (`extract_md_simple=False`) with known issues table.
- Docs: `docs/UNIFIED_INGESTION_PLAN.md` — section 4.4 DOCX fully updated with new pipeline diagram, metadata schema, and ref_images spec.

---
## [2026-03-26]
### 0.10.0-rc4
- Added: `POST /api/ingestion` unified ingestion endpoint (`tilellm/modules/ingestion/controllers.py`) — single entry point that routes to the correct pipeline based on document type and configuration: `pdf + use_ocr=True` → Docling OCR pipeline, `hybrid=True` → `add_item_hybrid`, default → `add_item`. Old endpoints `/api/scrape/single` and `/api/pdf/scrape` remain active for backward compatibility.
- Fixed: `unified_ingestion` controller no longer reads the raw HTTP request body (`await request.json()`); instead builds `PDFScrapingRequest` from `item.model_dump()` via the new `_build_pdf_request()` helper, avoiding body-consumption issues and fragile field mapping.
- Docs: Updated `docs/UNIFIED_INGESTION_PLAN.md` — section 1.1 marked as implemented, section 1.2 (HyDE) expanded with concrete TODOs and LangGraph integration notes.
- Docs: Removed duplicate HyDE entry from `docs/ROADMAP.md` (was listed twice as sections 2 and 4).

---
## [2026-03-24]
### 0.10.0-rc3
- Fixed: Situated context for text content.


---
## [2026-03-24]
### 0.10.0-rc2
- Fixed: Chat history management for gpt-5.x.

---
## [2026-03-14]
### 0.10.0-rc1
- Added: `CommonChunkMetadata` Pydantic schema (`tilellm/models/chunk_metadata.py`) as the canonical metadata contract shared across all ingestion pipelines and vector stores, with safe defaults and `extra="allow"` for forward compatibility.
- Added: `tilellm/shared/situated_context.py` implementing Anthropic's Contextual Retrieval technique: `enrich_chunks_with_situated_context()` prepends an LLM-generated situating sentence to each chunk before embedding, with concurrency control via semaphore and graceful fallback on LLM error.
- Added: `use_situated_context` flag (default `False`) to `ItemSingle` and `PDFScrapingRequest`; also `llm_provider` and `llm_model` fields to `ItemSingle` to configure the LLM used for enrichment without requiring the full QA model stack.
- Added: `tilellm/shared/markdown_utils.py` exposing `MarkdownChunker` as a shared utility importable outside the `pdf_ocr` module.
- Added: `heading_path` metadata field populated by `load_document()` for Markdown documents using the element hierarchy from `UnstructuredMarkdownLoader` (e.g. `"Introduction > Background > Methods"`).
- Added: `_compute_cross_modal_refs()` in `pdf_ocr/logic.py` computing real bbox-proximity-based `surrounding_text` for tables and images, and `ref_tables`/`ref_images` cross-references for text chunks, replacing the previous `"Element on page N"` stub.
- Added: Tables saved as `.md` (pipe-delimited Markdown) on MinIO in addition to Parquet (`{doc_id}/tables/{table_id}.md`); `md_path` propagated to vector store metadata.
- Fixed: Namespace collision in `pdf_ocr` pipeline where sequential `aadd_documents` calls (tables → images → text) each deleted the previous call's vectors. Resolved via `skip_delete` kwarg on `aadd_documents` (Pinecone, Milvus, Qdrant) tracked by `_first_index_done` in the orchestrator.
- Fixed: Pinecone `aadd_documents` now deletes by `metadata_id` filter instead of `delete_all=True` when `metadata_id` is provided, preventing accidental namespace wipe of other documents.
- Fixed: `id` and `metadata_id` standard fields added to all metadata dicts in `pdf_ocr` `_index_*` functions, ensuring Tiledesk filter compatibility.
- Added: `trafilatura>=2.0` as dependency; `get_content_by_url()` now tries Trafilatura first (fast path, strips JS/ads/nav) for `scrape_type=0` and `scrape_type=1`, with automatic fallback to `UnstructuredURLLoader` on empty content or failure.
- Added: `_extract_file_name()` helper in `document_tools.py` extracting clean filenames from HTTP URLs, presigned MinIO/S3 URLs, and local paths; `load_document()` now always sets `file_name` and normalises `page` (PyPDFLoader 0-indexed → 1-indexed; DOCX/TXT/MD default to 1).
- Added: `file_name` and `page` safety net in `chunk_documents` for Pinecone Serverless, Qdrant, and Milvus repositories.
- Added: Warning log when `PyPDFLoader` is used on a PDF, recommending the `/api/pdf/scrape` endpoint with Docling for complex documents.
- Updated: `pdf_ocr` `_index_text_chunks`, `_index_tables_to_vector_store`, and `_index_images_to_vector_store` use `CommonChunkMetadata` for metadata construction.

---
## [2026-03-13]
### 0.9.0
- Fixed: Resolved `BlockingIOError` in containerized environments by implementing a custom `TruncatingFormatter` for stdout.
- Added: Enhanced logging observability with a dual-handler system: truncated console output and full-length rotating file logs.
- Added: Dynamic logging configuration via environment variables (`LOG_FILE_PATH`, `LOG_LEVEL_STDOUT`, `LOG_LEVEL_FILE`).
- Updated: Standardized Gunicorn and Uvicorn log formats to ensure consistency across the application stack.
- Fixed: Refactored `handle_exception` utility to use standard logging with `exc_info=True` instead of direct `traceback.print_exc()`, ensuring full error persistence in rotating logs and preventing stdout buffer overflows.

---
## [2026-03-02]
### 0.9.0-rc4
- Fixed: tags management in `add_item` function, when `type="text"`

---
## [2026-02-27]
### 0.9.0-rc3
- Fixed: MinIO connection error on startup

---
## [2026-02-25]
### 0.9.0-rc2
- Fixed: ModuleNotFoundError in the multimodal_search module
- Updated: API documentation and README.md with a new section on conversation history management 

---
## [2026-02-25]
### 0.9.0-rc1
- Added: GraphRAG support integrated with FalkorDB for enhanced information retrieval.
- Added: Taskiq integration configured with Redis to handle heavy asynchronous tasks.
- Added: Experimental pdf_ocr module for extracting Markdown from PDF files (unstable).
- Modified: Offloaded heavy processing workloads to dedicated workers via Taskiq/Redis.

---
## [2026-01-27]
### 0.8.2-rc1
- Added: tags field to the `ItemSingle` and `QuestionAnswer` model in order to allow filter documents by tags during querying.

---
## [2026-01-14]
### 0.8.1-rc1
- Added: Pinecone reranker profile for reranking.

---
## [2026-01-13]
### 0.8.0-rc3
- Fixed: Correctly display errors in `/api/ask`.

---
## [2026-01-03]
### 0.8.0-rc2
- Added: Docker management in the `docker` folder.
- Added: Knowledge Graph creation from vector store chunks via `/api/kg/create`.
- Added: Hierarchical clustering using Leiden algorithm.
- Added: Global search capability on community reports.

---
## [2025-12-23]
### 0.7.3-rc2
- Updated: tiledesk-train-jobworker to version 0.0.41
---

## [2025-12-23]
### 0.7.3-rc1
- Fixed: Correct error messages and set response status to 400 on error.
- Modified: MCP servers can now filter tools to be used via `enabled_tools` parameter.
- Added: New endpoint `/api/listcompleteitems/namespace/{namespace}/all` for retrieving all items with full text content.
- Added: Knowledge Graph managed with Neo4j.
  - Graph creation starting from chunks in the repository.
  - Advanced query using 3. Balancing Matrix.
---


## [2025-12-19]
### 0.7.2-rc2
- Fixed: Restored missing Groq provider in the providers list.
---


## [2025-12-17]
### 0.7.2-rc1
- Added: implemented structured output response for /api/ask endpoint
- Added: structured output for /api/ask.
- Improved: interfaces for /api/scrape/single and /api/qa.
- Added: custom_headers and api_key for /api/ask to support vLLM.
---


## [2025-12-13]
### 0.7.1-rc2
- Fixed: Session leak issues in web scraping functions by adding proper try-finally blocks for browser cleanup.
- Fixed: Improved CAPTCHA error handling to propagate clear error messages to users instead of returning empty results.
- Fixed: Added proper resource cleanup for UnstructuredURLLoader sessions.
- Added: Documentation for reasoning models management via `/api/thinking` endpoint, including support for GPT-5, Claude 4/4.5, Gemini 2.5/3.0, and DeepSeek reasoner models.
---


## [2025-12-09]
### 0.7.1-rc1
- Added: New parameters for the /api/ask endpoint:
  - structured_output: Boolean flag indicating if the response should be a structured JSON output.
  - output_schema: The JSON schema the LLM must conform to in its response.
---


## [2025-12-07]
### 0.7.0-rc1
- Upgrade: Updated LangChain dependency to version 1.1.0.
- Changed: Reworked history management to support direct injection or context summarization.
- Added: New parameters for the /api/ask endpoint:
  - contextualize_prompt: Toggles injecting history as text vs. structured messages.
  - max_history_messages: Sets the maximum number of turns to retain.
  - summarize_old_history: Enables summarization of old history before discarding.
- Added: New parameter for the /api/qa endpoint:
  - contextualize_prompt: Toggles contextualize_q_system_prompt usage.
---


## [2025-12-01]
### 0.6.2-rc3
- Fixed: Injected chat history directly into the prompt. 
- Added: contextualize_prompt field (default=False) to enable/disable chat history system prompt contextualization
---


## [2025-11-28]
### 0.6.2-rc2
- Added: doc_batch_size to hybrid scraper to manage embedding generation for large documents
---

## [2025-11-27]
### 0.6.2-rc1
- Fixed: Issue resolved in the cache key management for vector stores.
- 
---

## [2025-11-11]
### 0.6.1
---

## [2025-10-28]
### 0.6.1-rc2
- Added: tools_log field to /api/ask response JSON.
---

## [2025-10-27]
### 0.6.1-rc1
- Added: Internal tools for pdf->images file conversion.
- Added: /api/convert API for pdf->images file conversion.
---

## [2025-10-25]
### 0.6.0-rc1
- Added: MCP support with multimodal LLM.
- Added: Internal tools for pdf->text and xlsx->csv file conversion.
- Added: Parameter tools:[] to /api/ask to support internal tools.
- Added: /api/tools API for internal tool discovery.
- Minor Fixed: General improvements and fixes.

---


## [2025-10-08]
### 0.5.3-rc1
- Updated: Conversion functionality.
- Minor Fixed: **Library dependencies** resolved.

---

## [2025-10-08]
### 0.5.2-rc3
- Minor Fixed: OpenAI cache key logic.

---

## [2025-10-07]
### 0.5.2-rc2
- Fixed: Cache key issue.

---

## [2025-09-19]
### 0.5.1-rc1
- Added: New endpoint `/api/convert`.
- Added: **Base64 file upload support** to `/api/ask`.

---

## [2025-09-19]
### 0.5.1-rc1
- Fixed: **Scraping logic** to wait for `networkidle`.

---

## [2025-09-09]
### 0.5.0-rc3
- Fixed: Qdrant integration when cache is enabled.

---

## [2025-09-09]
### 0.5.0-rc2
- Fixed: Filter is now correctly disabled if `_similarity_threshold_` is `1.0`.

---

## [2025-09-09]
### 0.5.0-rc1
- Added: **Global cache** implemented.
- Improved: Performance optimization for the `/api/qa` endpoint.

---

## [2025-08-29]
### 0.4.12-rc3
- Updated: Playwright to version `1.55.0`.

---

## [2025-08-06]
### 0.4.11-rc2
- Fixed: Asynchronous re-ranking function.

---

## [2025-08-01]
### 0.4.11-rc1
- Updated: Cache now managed with **TTL (Time-To-Live)** and `max_size`.

---

## [2025-07-30]
### 0.4.10-rc2
- Added: `browser_headers` parameter for the scraping process.

---

## [2025-07-21]
### 0.4.10-rc1
- Added: Cache support for the re-ranking feature.

---

## [2025-07-14]
### 0.4.9-rc1
- Added: **Re-ranking** functionality introduced.

---

## [2025-07-21]
### 0.4.8-rc1
- Added: **Mistral AI support**.

---

## [2025-07-16]
### 0.4.8-rc1
- Added: Simple **MCP (Multi-Container Project)** support.

---

## [2025-07-04]
### 0.4.7
- (No specific changes listed for this version)

---

## [2025-07-04]
### 0.4.7-rc1
- Added: **Hybrid batch size** parameter.

---

## [2025-06-20]
### 0.4.6-rc1
- Added: **Timed cache** implemented for embedding, vector store, and chat.

---

## [2025-06-20]
### 0.4.5-rc2
- Added: Timed cache implemented for embedding, vector store, and chat. (Duplicated entry, kept for consistency).

---

## [2025-05-16]
### 0.4.5-rc2
- Updated: `/api/qa` now includes the `chunk_only` parameter.
- Added: Local **hybrid search** enabled on Qdrant.

---

## [2025-05-16]
### 0.4.5-rc1
- Fixed: Hybrid indexing logic.

---

## [2025-05-16]
### 0.4.3-rc2
- Upgraded: **Torch library** dependency.

---

## [2025-05-16]
### 0.4.3-rc1
- Added: Chunks parameters added to `/api/qa`.
- Added: Support for **Qdrant vector store**.

---

## [2025-05-16]
### 0.4.2-rc4
- Fixed: Issue with `/api/qa` endpoint.

---

## [2025-05-16]
### 0.4.2-rc3
- Fixed: Asynchronous connection.

---

## [2025-05-16]
### 0.4.2-rc2
- Fixed: Scraping error handling.

---

## [2025-05-07]
### 0.4.2-rc1
- Upgraded: Python version to **3.12**.
- Upgraded: LangChain version to **0.3.25**.
- Added: **Stream support** for `/api/ask`, `/api/thinking`, and `/api/qa`.

---

## [2025-03-08]
### 0.4.1-rc2
- Added: Stream support for `/api/thinking`.

---

## [2025-02-16]
### 0.4.1-rc1
- Fixed: Parsing of Claude-3.7 response when thinking is enabled.

---

## [2025-02-16]
### 0.4.0-rc3
- Added: `/api/thinking` endpoint for **O1 and Claude-3.7**.

---

## [2025-02-16]
### 0.4.0-rc2
- Minor Fixed.

---

## [2025-02-16]
### 0.4.0-rc1
- Added: General **stream support**.

---

## [2024-10-10]
### 0.3.2-rc2
- Fixed: `/api/id/{id}/namespace/{namespace}/{token}` endpoint.
- Added: **Sentence embedding with `bge-m3`**.
- Added: Hybrid search with `bg3-m3`.
- Modified: Deleted the environment variable for vector store configuration.

---

## [2024-09-23]
### 0.3.0
- Added: **Hybrid search** capability.
- Added: Indexing based on **Spade**.
- Minor Fixed.

---

## [2024-09-17]
### 0.2.20
- Upgraded: Worker to version `0.0.27`.

---

## [2024-09-14]
### 0.2.19
- Upgraded: Worker to version `0.0.25`.

---

## [2024-09-14]
### 0.2.18
- Upgraded: Worker component.
- Modified: Default value for `scrape_type` is now `4`.

---

## [2024-09-05]
### 0.2.17
- Fixed: NLTK download issue in the Dockerfile.

---

## [2024-09-04]
### 0.2.16
- Fixed: `max_tokens` set to `1024` if `citations=True`.

---

## [2024-09-04]
### 0.2.15
- Fixed: Citations appearing without quotes.

---

## [2024-09-04]
### 0.2.14
- Modified: Citations now appear **without quotes**.

---

## [2024-09-04]
### 0.2.13
- Modified: Source logic on QA.

---

## [2024-08-31]
### 0.2.12
- Added: **Citations** feature.

---

## [2024-07-31]
### 0.2.11
- Fixed: Logging issue.

---

## [2024-07-31]
### 0.2.10
- Fixed: Write log functionality.
- Updated: Library versions.

---

## [2024-07-29]
### 0.2.9
- Added: `n_messages` parameter to `/api/ask` to set the maximum number of messages to include.

---

## [2024-07-27]
### 0.2.8
- Added: **History** support on `/api/ask`.

---

## [2024-07-26]
### 0.2.7
- Added: `scrape_type=3|4` options.
- Added: `similarity_threshold` parameter to `/api/qa`.

---

## [2024-07-09]
### 0.2.6
- Added: `DELETE /api/chunk/<chunk_id>/namespace/<namespace>` endpoint.
- Added: `search_type` parameter with options `similarity` or `mmr`.

---

## [2024-07-01]
### 0.2.5
- Fixed: User-agent setting for scraping.

---

## [2024-07-01]
### 0.2.4
- Fixed: `scrape_type=0` functionality.
- Added: **`/api/ask`** endpoint for LLM querying.

---

## [2024-06-21]
### 0.2.3
- Fixed: Deletion of chunks from a namespace by metadata ID.
- Added: `/api/desc/namespace/{ns}` endpoint for namespace description.

---

## [2024-06-15]
### 0.2.2
- Fixed: Indexing of `.txt` documents.

---

## [2024-06-15]
### 0.2.1
- Updated: LangChain to version `0.1.16`.
- Modified: Prompt for Q&A functionality.

---

## [2024-06-08]
### 0.2.0
- Refactored: Repository structure to manage both **Pod and Serverless** environments.

---

## [2024-06-07]
### 0.1.21
- Added: Support for **PDF, DOCX, and TXT** file formats.

---

## [2024-06-06]
### 0.1.20
- Added: `log_conf.json` file.

---

## [2024-06-06]
### 0.1.19
- Minor Fixed: Returns `400` status code if the URL is incorrect.

---

## [2025-05-20]
### 0.1.18
- Added: `scrape_type = 0|1` options.
- Added: **`trainer_worker`** as a Node.js application.

---

## [2025-05-20]
### 0.1.17
- Added: `PIENCONE_TYPE = "serverless|pod"` environment variable.

---

## [2025-05-18]
### 0.1.16
- Added: `/api/scrape/single` endpoint without Redis queue.
- Added: `/api/scrape/enqueue` to enqueue items into the Redis queue.

---

## [2025-05-14]
### 0.1.15
- Minor Fixed: Dockerfile adjustments.

---

## [2025-05-07]
### 0.1.14
- Added: Parameter to `entrypoint.sh`.

---

## [2025-05-06]
### 0.1.13
- Fixed: Deletion of IDs from namespace (max `top_k` limit is now `10k`).

---

## [2025-05-03]
### 0.1.12
- Fixed: Sends status `200` to webhook and ID for status `400`.
- Added: **DELETE of namespace** via POST method.
- Fixed: `/api/scrape/status` now checks status in both Redis and Pinecone.

---

## [2025-05-02]
### 0.1.11
- Fixed: `log_conf.json` configuration.

---

## [2025-05-02]
### 0.1.10
- Fixed: Metadata fields cannot be `None`.
- Added: `TILELLM_ROLE=qa|train` to manage QA and training roles.

---

## [2025-05-01]
### 0.1.9
- Modified: `log_conf.json` level changed to **INFO**.

---

## [2024-04-30]
### 0.1.8
- Added: `log_conf.json` to the Dockerfile.

---

## [2024-04-24]
### 0.1.7
- Fixed: Logging functionality.

---

## [2024-04-22]
### 0.1.6
- Fixed: JSON response for `/api/delete/id`.

---

## [2024-04-20]
### 0.1.5
- Added: `GET /api/listitems/namespace/{namespace}` to list items by namespace.
- Added: `POST /api/list/namespace` to list all namespaces.
- Fixed: Several asynchronous functions.

---

## [2024-04-19]
### 0.1.4
- Updated: Webhook now supports status codes `300` and `400`.

---

### 0.1.3
- Added: Delete chunk by ID/namespace using a POST method.
- Added: **Expiration time** to the Redis cache.
