# **TILEDESK LLM - Changelog**

### **Authors**:
    * Gianluca Lorenzo
    * Andrea Sponziello
### **Copyright**: Tiledesk SRL


---
## [2026-05-18]
### 0.10.1-rc24 (feat: /api/v2/qa simplified agentic endpoint with optional guard/grader; fix: rag_node StreamingResponse in graph; fix: lgraph date filter findall tuple)

**`tilellm/modules/api_v2/controllers.py`** — rinominato l'endpoint agentico completo da `/qa` a `/query`; aggiunto `/qa` semplificato che usa `QASimpleRequest` e il nuovo `simple_app`. Helper `_build_response()` condiviso tra i due endpoint.

**`tilellm/modules/api_v2/models.py`** — `QASimpleRequest` (già presente): estende `QuestionAnswer` con `use_guard: bool = True` e `use_hallucination_grader: bool = True`.

**`tilellm/agents/workflow.py`** — aggiunto `simple_app`: workflow LangGraph ridotto per `/api/v2/qa`. Pipeline: `_start → [guardia] → intent_router → compliance_node | rag_core → [validatore] → fail_safe_node`. Routing dinamico tramite `_simple_guard_router` (salta `guardia` se `use_guard=False`) e `_simple_after_rag_router` (salta `validatore` se `use_hallucination_grader=False`). Il workflow completo `app` (con HyDE, cache, RAPTOR) rimane invariato per `/api/v2/query`.

**`tilellm/agents/nodes.py`** — `rag_node`: forza `stream=False` prima di chiamare `ask_with_memory`/`ask_hybrid_with_memory` quando `question_answer.stream=True`; il grafo richiede un `RetrievalResult` ispezionabile, non una `StreamingResponse`. `fail_safe_node`: gestione esplicita di `RetrievalResult` (`.model_dump()`), `JSONResponse` (`.body` bytes) e fallback a `{}` per qualsiasi altro tipo, evitando `AttributeError` su oggetti `StreamingResponse`.

**`tilellm/modules/lgraph/logic.py`** — `_filter_chunks_by_date`: corretto `date_re.findall(chunk.text)` → `date_re.finditer(chunk.text)` con `.group(0)`. `findall` con regex a gruppi catturanti restituisce tuple invece di stringhe, causando `TypeError: expected string or bytes-like object, got 'tuple'` in `_parse_it_date`.

---
## [2026-05-15]
### 0.10.1-rc23 (perf: /api/qa parallel init + retrieval_query propagation; fix: Docling GPU meta-tensor fallback + async linker)

**`tilellm/controller/controller.py`** — `ask_hybrid_with_memory`: `initialize_embeddings_and_index` and `create_contextualize_query` now run in parallel via `asyncio.gather`, saving one sequential LLM round-trip per request. Contextualized query is propagated as `retrieval_query` **before** `fetch_question_vectors` so the dense/sparse embedding is computed on the reformulated text (improves recall in chat-history sessions). When the query changes, `precomputed_query_embedding` is invalidated to prevent a stale semantic-cache hit. Removed stale Italian inline comments.

**`tilellm/controller/controller_utils.py`** — `fetch_question_vectors` and `fetch_question_vectors_nopar`: use `retrieval_query or question` as the text for both dense and sparse encoding.

**`tilellm/modules/pdf_ocr/services/docling_processor.py`** — `_init_docling`: added CPU fallback via `AcceleratorOptions(device="cpu")` when a `RuntimeError: meta tensor` is raised during GPU init. Prevents hard crashes on pods where the GPU memory state is stale after a prior OOM.

**`tilellm/modules/pdf_ocr/services/markdown_extractor.py`** — same CPU fallback pattern in `_init_docling`.

**`tilellm/modules/pdf_ocr/logic.py`** — `_generate_image_semantic_links` and `_generate_table_descriptions`: pass `graph_repository=None` to `ImageSemanticLinker`/`TableSemanticLinker` (both are sync-only; passing an `AsyncFalkorGraphRepository` caused `RuntimeWarning: coroutine 'update_node' was never awaited`).

---
## [2026-05-11]
### 0.10.1-rc19 (fix: TaskIQ double-processing of same document after worker crash)

Fixed a bug where a PDF document could be processed twice after a worker crash.

**Root cause**: when a worker crashes (e.g. OOM from concurrent Docling runs) between `"Task finished"` log and the Redis `XACK`, the stream message stays in PEL. On restart, `_startup_reclaim` re-queued it as a fresh `attempt 1/3` even if the task had already completed.

**`tilellm/modules/task_executor/broker.py`** — `_startup_reclaim`: before re-dispatching a PEL entry, checks if a result key already exists in Redis for that `task_id`. If found, only `XACK` is sent (no `XADD`). Logs `"già completato, solo ACK"` and tracks `total_skipped` in the startup summary.

**`tilellm/modules/task_executor/tasks.py`** — moved `"Task finished"` log outside the webhook block so it always appears; added comment documenting that `XACK` happens after this point.

**Recommendation**: set `PDF_MAX_CONCURRENT=1` on GPU pods to prevent the concurrent Docling OOM that triggers this condition.

---
## [2026-05-04]
### 0.10.1-rc17 (perf: RAM reduction for /api/pdf/scrape ingestion)

Reduces RAM consumption during PDF ingestion by eliminating per-task model reload, releasing PIL image data early, and forcing Python GC at task boundaries.

#### Fix A — `ProductionDocumentProcessor` singleton

- `tilellm/modules/pdf_ocr/services/docling_processor.py`: added module-level `_processor_instance` singleton and `get_or_create_processor()` async factory.
- Previously, every `/api/pdf/scrape` task instantiated a new `ProductionDocumentProcessor`, which calls `DocumentConverter.__init__()` and loads all Docling ONNX models (layout EfficientDet, TableFormer, RapidOCR) — 2-4 GB RAM per instantiation. Under concurrent load (multiple tasks dispatched by TaskIQ) this caused model copies to accumulate until GC ran.
- `get_or_create_processor()` uses double-checked locking (`asyncio.Lock`) to initialize once on first call and reuse across all subsequent tasks. Models stay resident; each task pays only the conversion cost.
- Removed `finally: await processor.close()` from `process_pdf_document_with_embeddings` in `logic.py` — the singleton must not have its Redis client closed between tasks.
- Removed `duckdb_conn.register(safe_table_name, df)` from `_process_single_table`: the registration kept Python DataFrame references alive inside the DuckDB connection for the process lifetime. The registered view was never queried in the current codebase; removed it and the `table_data['duckdb_table']` field.

#### Fix B — Release PIL image data after captioning

- `tilellm/modules/pdf_ocr/logic.py` (`_generate_image_captions`): sets `image_data['image_data'] = None` after converting the PIL Image to bytes and generating the caption. A 300 DPI A4 page as RGB numpy array ≈ 26 MB; a 50-page scanned PDF produces ≈ 1.3 GB of image data held unnecessarily for the rest of the task.
- Additional bulk cleanup in `process_pdf_document_with_embeddings`: after the image captioning block (regardless of `include_images` flag) iterates `result['images']` and sets `image_data = None` for any remaining PIL objects, covering the case where images were extracted by Docling but captioning was skipped.

#### Fix C — `gc.collect()` in `GpuCleanupMiddleware`

- `tilellm/modules/task_executor/broker.py`: added `gc.collect()` call immediately after `cuda_empty_cache_safe()` in `GpuCleanupMiddleware.post_execute`.
- Forces Python's cyclic garbage collector to run at task boundaries, releasing PIL objects, DataFrames, and LangChain document lists that may have cyclic references preventing prompt reference-count cleanup.

#### Files changed

`tilellm/modules/pdf_ocr/services/docling_processor.py`,
`tilellm/modules/pdf_ocr/logic.py`,
`tilellm/modules/task_executor/broker.py`

---
## [2026-05-04]
### 0.10.1-rc16 (perf: GPU load reduction for TaskIQ ingestion workers)

Reduces performance degradation on GPU pods running `/api/pdf/scrape` with a local SPLADE sparse encoder under sustained ingestion load.

#### `GpuCleanupMiddleware` — VRAM cache flush between tasks

- New `TaskiqMiddleware` in `tilellm/modules/task_executor/broker.py` that calls `torch.cuda.empty_cache()` after every task with `task_type` in `{"pdf_ocr", "scraping", "raptor_build"}`.
- PyTorch's caching allocator retains freed blocks for reuse across tasks; under sustained load with variable-length chunk inputs these accumulate and reduce the pool available for new allocations, eventually triggering OOM recovery (budget halving → slower batches → queue buildup). Flushing the cache at task boundaries — not per batch — resets the allocator to a clean state for the next task.
- Logs a `[GPU] cache flushed after <task_type>: freed X MB | VRAM Y → Z MB / <total> MB` line at `INFO` level after every flush. `freed > 0` indicates fragmented blocks were recovered; `freed ≈ 0` means the previous task left no cached blocks (normal for light tasks).
- `torch.cuda.empty_cache()` is a no-op when CUDA is not available — safe on CPU-only workers.
- `PDF_MAX_CONCURRENT` (env var, default `2`) limits how many PDF tasks enter the Docling pipeline simultaneously per worker process. Set to `1` on single-GPU pods to prevent two Docling instances from competing for VRAM. Note: TaskIQ dispatches all received tasks as async coroutines immediately (the "Task started" log fires before the semaphore), so seeing N task starts in the log does not mean N Doclings are running — only `PDF_MAX_CONCURRENT` tasks actually enter processing at a time.

#### Dedicated single-thread GPU executor — replaces `GLOBAL_GPU_LOCK`

The previous design used a process-wide `threading.Lock` (`GLOBAL_GPU_LOCK`) acquired per batch inside `asyncio.to_thread` workers. Under concurrent ingestion this caused thread pool growth (blocked threads accumulate in the default `ThreadPoolExecutor`) and lock contention latency.

The new design routes all local GPU inference through a single-thread `ThreadPoolExecutor`:

- **`tilellm/tools/_gpu_concurrency.py`**: new `_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1)` and `async def run_on_gpu(fn, *args, **kwargs)` coroutine. `GLOBAL_GPU_LOCK` retained as a no-op for backward compatibility.
- **`tilellm/tools/sparse_encoders.py`**: removed `GLOBAL_GPU_LOCK` from `_run_sparse_batched_with_oom_recovery`, `TiledeskSpladeEncoder.encode_queries`, `TiledeskBGEM3.encode_queries`, and `TiledeskBGEM3.encode_documents` (legacy path). Added `aencode_documents` and `aencode_queries` async methods to `TiledeskSparseEncoders`: local GPU encoders (SPLADE, BGE-M3) use `run_on_gpu`; TEI encoders use `asyncio.to_thread` (remote HTTP, must not block the GPU thread).
- **`tilellm/tools/reranker.py`**: removed `_gpu_lock = GLOBAL_GPU_LOCK` class attribute and `with self._gpu_lock:` from `_predict_local_with_oom_recovery`. `TileReranker.arerank_documents` now uses `run_on_gpu`. `PineconeReranker.arerank_documents` unchanged (remote API call).
- **All call sites** updated to `await encoder.aencode_documents(...)` / `await encoder.aencode_queries(...)`, eliminating the remaining sync calls that were blocking the event loop: `controller_utils.py` (`fetch_question_vectors`, `fetch_question_vectors_nopar`), `qdrant_repository_local.py` (×3), `pinecone_repository_serverless.py` (×3), `pinecone_repository_base.py` (×1), `milvus_repository.py` (×2).

#### Files changed

`tilellm/tools/_gpu_concurrency.py`,
`tilellm/tools/sparse_encoders.py`,
`tilellm/tools/reranker.py`,
`tilellm/controller/controller_utils.py`,
`tilellm/store/qdrant/qdrant_repository_local.py`,
`tilellm/store/pinecone/pinecone_repository_serverless.py`,
`tilellm/store/pinecone/pinecone_repository_base.py`,
`tilellm/store/milvus/milvus_repository.py`,
`tilellm/modules/task_executor/broker.py`

---
## [2026-05-03]
### 0.10.1-rc15 (feat: Temporal Digest — agentic query, hybrid search, reranking, history, multi-backend metadata filter)

Complete overhaul of the `temporal_digest` module with production-ready retrieval quality features and a new agentic query endpoint.

#### New endpoint: `POST /api/digest/qa` (agentic query)

- LLM extracts `date_from`, `date_to`, and `query_mode` from free-form natural language questions and conversation history (e.g. "cosa hanno fatto la settimana scorsa?" → `date_from: 2026-04-27`, `date_to: 2026-05-02`, `query_mode: temporal`).
- Relative date expressions resolved against server date or optional `today` field.
- Returns `extracted_date_from`, `extracted_date_to`, `extracted_query_mode`, `agent_reasoning` alongside the answer.
- 2 LLM calls per request: parameter extraction + answer synthesis.

#### Hybrid search + reranking in digest query

- `DigestQueryRequest` and `DigestAgentRequest` now support `search_type: "hybrid"`, `sparse_encoder` (`string` or `TEIConfig`), `reranking` (`bool`, `TEIConfig`, `PineconeRerankerConfig`), `reranker_model`, `reranking_multiplier`.
- When reranking is enabled, the semantic path fetches `top_k × reranking_multiplier` candidates and reranks via `TileReranker.arerank_documents()` before passing to the LLM.
- Temporal path also reranks digest candidates when `reranking` is set.

#### Conversation history

- `DigestQueryRequest` and `DigestAgentRequest` accept `chat_history_dict` (same format as `/api/qa`) and `max_history_messages`.
- History is prepended to the LLM prompt before the evidence block in both `_query_temporal` and `_query_semantic`.

#### `additional_metadata` in `POST /api/pdf/scrape`

- New `additional_metadata: Dict[str, Any]` field in `PDFScrapingRequest`.
- Arbitrary key-value pairs merged into every chunk's payload at ingestion time.
- `date` value in `DD/MM/YYYY` auto-converted to ISO `YYYY-MM-DD`.
- Applied at all 4 indexing paths: text chunks, table chunks, image captions, markdown pipeline.

#### `_metadata_filter` applied to all vector store backends

- `get_chunks_from_repo` now reads `getattr(question_answer, '_metadata_filter', None)` and applies it as a filter on all backends:
  - **Qdrant**: `FieldCondition` + `DatetimeRange` for ISO date strings (`_is_iso_date` guard).
  - **Pinecone serverless/pod**: MongoDB-style filter dict, merged with tags filter via `$and`.
  - **Milvus**: `build_filter` extended with `metadata_filter` parameter; `convert_to_expression` handles `{field: {$op: value}}` → Milvus string expressions.
- Fixes false-positive `already_existed: true` on digest generation (digests were not being found due to ignored type/date filters).

#### Query router observability

- `classify_query_debug(question)` returns `(mode, matched_pattern)`.
- `DigestService.query()` logs routing decisions: `[query_router] auto → 'temporal' | pattern=... | question=...`.

#### Bug fixes

- **`TiledeskSparseEncoders(None)` crash** in `aadd_documents`: `sparse_encoder` initialization and all sparse encoding steps are now conditional on `sparse_encoder is not None`.
- **Qdrant `DatetimeRange`**: `models.Range` only accepts floats; ISO date strings now routed to `models.DatetimeRange`.
- **`RetrievalChunksResult` missing `namespace`**: fallback objects in `_fetch_source_chunks` and `_fetch_digests` now include `namespace=qa.namespace`.
- **Duplicate digest vectors**: removed `skip_delete=True` from `_index_digest` — `aadd_documents` now deletes by `metadata_id` before inserting.
- **Evidence block date headers**: `_build_evidence_block` uses `digest_date_from`/`digest_date_to` as header for digest chunks (was showing empty `file_name`/`source`), giving the LLM explicit date context per digest.
- **Double embedding (Qdrant non-hybrid)**: `langchain_qdrant._validate_collection_for_dense()` was calling `embed_documents(["dummy_text"])` on every search. Non-hybrid path now bypasses LangChain `asearch` and calls `index.query_points()` directly.

#### Files changed

`tilellm/modules/temporal_digest/models/schemas.py`,
`tilellm/modules/temporal_digest/services/digest_service.py`,
`tilellm/modules/temporal_digest/services/query_router.py`,
`tilellm/modules/temporal_digest/logic.py`,
`tilellm/modules/temporal_digest/controllers.py`,
`tilellm/modules/pdf_ocr/models/pdf_scraping.py`,
`tilellm/modules/pdf_ocr/logic.py`,
`tilellm/store/qdrant/qdrant_repository_local.py`,
`tilellm/store/pinecone/pinecone_repository_serverless.py`,
`tilellm/store/pinecone/pinecone_repository_pod.py`,
`tilellm/store/milvus/milvus_repository.py`

---
## [2026-04-27]
### 0.10.1-rc14 (feat: RAPTOR performance optimizations)

Performance improvements across the entire RAPTOR hierarchical summarization pipeline, addressing the main throughput bottlenecks.

#### `tilellm/modules/raptor/services/raptor_service.py`
- **Parallelized LLM summary generation.** The per-cluster summary loop now uses `asyncio.gather()`, running all LLM calls for a level concurrently instead of sequentially. This reduces wall-clock time roughly proportional to the number of clusters per level.
- **Batched vector store writes.** Leaf-node indexing and summary-node indexing both previously issued one `aadd_documents([doc])` call per node. Each call now collects all documents first and issues a single `aadd_documents(all_docs)` call per phase, eliminating round-trip overhead.
- **UMAP+GMM offloaded to thread pool.** `RAPTORClustering.perform_clustering` is now wrapped in `asyncio.to_thread()`, preventing the CPU-bound UMAP dimensionality reduction and GMM fitting from blocking the event loop.

#### `tilellm/modules/raptor/utils/clustering.py`
- **Eliminated O(n²) float comparison in `perform_clustering`.** The previous implementation used `np.allclose` in a nested loop (`find_matching_indices`) to map local-cluster embeddings back to their original indices. The rewritten code tracks array indices directly through UMAP slicing operations, reducing complexity from O(n²·d) to O(n).
- **BIC early stopping in `get_optimal_clusters`.** GMM fitting previously tried all values from 1 to `max_clusters` (default: 50) unconditionally. The function now stops after 3 consecutive iterations without improvement, reducing the average number of fits from 50 to a fraction of that on typical RAPTOR inputs.

#### `tilellm/modules/raptor/repository/raptor_repository.py`
- **Added `batch_index_nodes`.** New method that builds all `Document` objects from a node list and delegates to a single `aadd_documents` call, used by `RaptorService` instead of looping `index_summary_embedding`.
- **Pipeline Redis deletes in `delete_tree`.** Node key deletions now execute in a single pipelined round-trip instead of one `await redis.delete()` per node.
- **MGET in `list_trees` and `get_tree_by_doc_id`.** Both methods previously fetched tree metadata with N individual `GET` calls after `scan_iter`. They now issue a single `MGET` for all scanned keys.

#### `tilellm/modules/raptor/controllers.py`
- **Parallelized `/summarize` endpoint.** The per-group LLM summarization loop in `_summarize_logic` now uses `asyncio.gather()`.

---
## [2026-04-26]
### 0.10.1-rc13 (feat: FalkorDB graph robustness, clustering fixes, entity deduplication, graph optimization)

A comprehensive set of fixes and improvements for the FalkorDB Knowledge Graph pipeline, addressing data loss during concurrent execution, incorrect clustering due to FalkorDB's internal result cap, LLM context-window saturation, and entity duplication across extraction runs.

#### Graph creation robustness

- **Fixed: node deletion during concurrent runs.**
  Two TaskIQ workers picking up the same graph-creation task would delete each other's work via the `delete_nodes_by_metadata` call at the start of a new run. Three complementary fixes:
  1. **Redis distributed lock** on `create_community_graph` prevents a second worker from entering the creation flow while one is already running. Lock key: `graph_lock:{graph_name}`, TTL controlled by `GRAPH_LOCK_TTL_SECONDS` (default: 86400 s / 24 h).
  2. **`retry_on_error=False`** on `task_falkor_graph_create`: TaskIQ's retry-on-error mechanism re-submitted the task with `overwrite=True`, which caused the wipe+restart cycle on transient failures. Auto-retry is now disabled; failures must be re-submitted explicitly.
  3. **Fixed `LIMIT 10000` bug** in `delete_nodes_by_metadata` (`async_falkor_repository.py` and `falkor_repository.py`): the deletion query ran only once, leaving nodes beyond the first 10 000 untouched. It now loops in batches until FalkorDB reports 0 deleted rows.

- **Added: MinIO checkpoint / resume for graph extraction.**
  If the TaskIQ worker is killed mid-extraction, the next run with `overwrite=False` resumes from the last completed window instead of restarting from scratch.
  - `MinIOStorageService.save_checkpoint(graph_name, window_idx, entity_node_map_delta, chunk_ids)` — saves a per-window Parquet file to `checkpoints/{graph_name}/window_{n:06d}.parquet`.
  - `MinIOStorageService.load_checkpoints(graph_name)` — aggregates all checkpoint files and returns `(entity_node_map, processed_chunk_ids, last_window_idx)`.
  - `MinIOStorageService.delete_checkpoints(graph_name)` — cleans up checkpoint files after a successful run.
  - `import_from_vector_store`: on `overwrite=False`, loads checkpoints to restore `entity_node_map` and skip already-processed chunks; saves a checkpoint after each window; deletes all checkpoints on completion.

#### Clustering robustness

- **Fixed: Redis lock on community report generation.**
  `generate_hierarchical_reports` is now protected by the same Redis lock (`graph_lock:{graph_name}`) used for graph creation, preventing concurrent clustering workers from triggering redundant cleanup and report deletion.

- **Fixed: `retry_on_error=False`** on all three FalkorDB clustering tasks: `task_falkor_louvain_cluster`, `task_falkor_leiden_cluster`, `task_falkor_hierarchical_cluster`. Prevents destructive re-runs on transient failures.

- **Fixed: FalkorDB `Query timed out` during clustering on large graphs.**
  The single `OPTIONAL MATCH` query for fetching the full graph caused a timeout on graphs with 30 000+ nodes (connection-level default: 1 s).
  - `_execute_query` now accepts an optional `timeout` parameter (ms), passed directly to `graph.query(..., timeout=timeout)`.
  - `get_all_nodes_and_relationships` rewritten as two separate queries (nodes then relationships, excluding `CommunityReport` nodes) with `query_timeout=300000` (5 minutes).

- **Fixed: FalkorDB `resultset_size=10000` cap truncating clustering input.**
  FalkorDB silently caps query results at 10 000 rows. On a 33 000-node graph both queries returned exactly 10 000 rows, causing Leiden to cluster a partial graph and produce 2 600+ spurious communities.
  Both queries in `get_all_nodes_and_relationships` now paginate with `SKIP $skip LIMIT 5000`, looping until the last page is shorter than the page size.

- **Added: configurable Leiden resolution and minimum community size.**
  `GraphClusterRequest` now exposes:
  - `resolutions: Optional[List[float]]` — per-level resolution `[L0, L1, L2]`. Defaults to `[0.05, 0.15, 0.35]` (lowered from the previous hardcoded `[1.2, 0.8, 0.5]`), producing fewer and larger communities on sparse graphs.
  - `min_community_size: int` (default: 8) — communities below this threshold are discarded without generating a report.

- **Fixed: LLM context-window saturation during community report generation.**
  `_generate_community_report_igraph` now caps prompt content via `max_prompt_chars` (default: 18 000 chars ≈ 4 500 tokens). Entity descriptions are truncated to 200 characters each; entities get 2/3 of the budget and relationships 1/3; excess items are replaced by a `[... N more not shown]` marker.
  `GraphClusterRequest` exposes `max_community_prompt_chars: int` (default: 18 000) so callers can tune it per-request for smaller-context models.

#### Entity deduplication (Level 1 + Level 2)

Root cause of graph sparsity: each extraction run created new nodes for every entity, even if an identical or near-identical node already existed in the graph, inflating node counts, breaking connectivity, and causing Leiden to detect thousands of isolated communities.

- **Level 1 — Name normalization.**
  Added `AsyncFalkorGraphRepository._normalize_name(name)` → `name.strip().lower()`. All `entity_node_map` keys are now normalized, so "ASL Bari", "asl bari", and " ASL Bari " resolve to the same node. Applied in `batch_create_nodes` (UNWIND path and individual-create fallback) and `batch_create_relationships` (source/target lookup).

- **Level 2 — Pre-load existing nodes before extraction.**
  Added `AsyncFalkorGraphRepository.load_entity_name_map(namespace, graph_name)` — paginates the graph in batches of 5 000, returning `normalized_name → node_id` for all existing non-`CommunityReport` nodes.
  In `import_from_vector_store`, when `overwrite=False`, the entity map is pre-populated from the graph before the window loop. Checkpoint data (if present) is merged on top as the more recent source of truth.

- **Deduplication during window loop.**
  `batch_create_nodes` now accepts `existing_entity_node_map`. Entities whose normalized name already appears in it are skipped (no `CREATE` issued) and their existing `node_id` is returned directly. Callers pass the current `entity_node_map` on every window, so entities seen in window 1 are not re-created in window 400.

#### Graph optimization and reimport

- **Added: `POST /api/kg-falkor/optimize` — embedding-based entity deduplication.**
  New `GraphOptimizer` service (`services/graph_optimizer.py`) runs the full pipeline as an async TaskIQ task:
  1. Export the full graph from FalkorDB → Parquet bytes.
  2. Batch-embed all entity `name: description` strings via the configured LLM embedder (optimised for TEI/vLLM).
  3. Find near-duplicate pairs with **SimSIMD** `cdist` (cosine, chunked at 2 000 rows to cap peak memory) + Union-Find transitive grouping.
  4. Build the merge plan with **DuckDB** (used as in-memory SQL engine via pandas bridge): canonical nodes get unioned `source_ids` (vector store references preserved), relationships are redirected and deduplicated.
  5. Save the optimised snapshot to MinIO (`graph_snapshots/{graph_name}/{timestamp}/`).
  6. Wipe FalkorDB and reimport from the optimised snapshot. Community reports are restored from MinIO as-is — no re-clustering required.
  - `dry_run=true` returns the merge plan stats without touching FalkorDB.
  - `similarity_threshold` (default 0.92) and `embedding_batch_size` (default 256) are API parameters.

- **Added: `POST /api/kg-falkor/reimport` — restore graph from MinIO snapshot.**
  Wipes FalkorDB and reimports nodes + relationships from any saved snapshot (latest by default, or a specific `snapshot_timestamp`). Community reports optionally restored from MinIO. Useful for rollback after a failed optimization, version management, or disaster recovery.

- **Added: Community reports persisted to MinIO Parquet per Leiden level.**
  After each Leiden level in `_generate_hierarchical_reports_locked`, community reports are saved to `community_reports/{graph_name}/level_{n:02d}.parquet` via `MinIOStorageService.save_community_reports`. This makes reports available for the reimport pipeline without re-clustering.

- **Added: MinIO graph snapshot API** (`save_graph_snapshot`, `load_graph_snapshot`, `list_graph_snapshots`) in `MinIOStorageService`.
  Path: `graph_snapshots/{graph_name}/{timestamp}/nodes.parquet` and `.../relationships.parquet`.

- **New schemas**: `GraphOptimizeRequest`, `GraphOptimizeResponse`, `GraphReimportRequest`, `GraphReimportResponse`.
- **New TaskIQ tasks**: `task_falkor_optimize_graph`, `task_falkor_reimport_graph` (both `retry_on_error=False`).

---
## [2026-04-25]
### 0.10.1-rc12
- Added: **Situated Context Profiles** support. Prompts for Contextual Retrieval can now be stored as YAML files in `tilellm/shared/profiles/situated_context/`.
- Added: New `determina.yaml` profile optimized for administrative acts, helping Knowledge Graph extraction by anchoring facts to dates and document IDs.
- Improved: Global support for `profile` and `custom_prompt` in `SituatedContextConfig` across all ingestion pipelines (PDF OCR, DOCX) and repositories (Qdrant, Pinecone, Milvus).
- Fixed: Double description noise in PDF OCR. The system now intelligently skips redundant situated context LLM calls for tables and images that already have semantic descriptions.
- Fixed: `TypeError` in `top_p_range` validator in `tilellm/models/llm.py` when `top_p` is None.

## [2026-04-24]
### 0.10.1-rc11
- Fixed: Semantic cache validation error in `POST /api/qa` where `RetrievalResult` failed due to missing required `namespace` field.
- Improved: Robust result extraction for caching. The system now correctly handles `JSONResponse` objects from the controller, preventing empty or failed results from being stored in Redis.
- Added: New fields `cache_level` and `cache_similarity` to `RetrievalResult` model.
- Improved: Cache metadata mapping in both `/api/qa` and `/api/v2/qa` (LangGraph). Hits now return explicit cache level ('exact' or 'semantic') and similarity score in the JSON response.
- Fixed: LangGraph `cache_lookup_node` and `cache_store_node` now correctly inject `namespace` and handle serialization using `jsonable_encoder`.

---
### 0.10.1-rc10
- Added: `table_chunker.py`



---
## [2026-04-23]
### 0.10.1-rc5 (feat: HTML page title as file_name + source_file_name in citations)

Two complementary improvements for UX-friendly source attribution on HTML content:

#### HTML page title as chunk `file_name`
For web pages (all `type=url` scrape paths), `metadata.file_name` is now set to the text of the HTML `<title>` tag instead of the URL basename. A URL like `https://example.com/products/123` would previously produce `"123"` as the label; it now produces `"Product Alpha – Acme Store"`. The title is extracted from the raw HTML, so no extra network request is needed.

- Added: `_extract_html_title(html: str) -> str` helper in `tilellm/tools/document_tools.py` — parses `<title>` via BeautifulSoup (already available), returns empty string on failure.
- Changed: `_handle_trafilatura_scrape()` — sets `file_name` on all returned docs (text + table) from the `<title>` of the downloaded HTML. Covers scrape_type 0 and 1.
- Changed: `handle_chromium_loader()` — builds a `{url → title}` map from `raw_htmls` before transformation; applies `file_name` to all resulting docs by matching `source`. Covers scrape_type 3 and 4.
- Changed: `scrape_page()` — sets `file_name` on text and table docs from `<title>` of Playwright-fetched HTML. Covers scrape_type 2.
- Changed: `scrape_page_complex()` — same pattern. Covers scrape_type 5.
- Changed: `scrape_page_fallback_selectors()` — same pattern. Covers the robust fallback path.
- Unchanged: `handle_unstructured_loader()` (rare fallback) — `file_name` still falls back to URL basename via repository safety net. Acceptable; UnstructuredURLLoader does not expose raw HTML.

#### `source_file_name` in `Citation`
Added `source_file_name` to the `Citation` object returned when `citations=true`. The field contains the human-readable file or page name stored in `metadata.file_name` at ingestion time (e.g. `"price-list.pdf"`, `"Home – Acme Corp"`). Useful for building labelled links in UX (`[source_file_name](source_name)`) instead of exposing raw URLs. The field is `null` when metadata is absent. Fully backward-compatible — existing consumers that ignore unknown fields are unaffected.

- Changed: `tilellm/models/schemas/retrieval_schemas.py` — added `source_file_name: Optional[str] = None` to `Citation`.
- Changed: `tilellm/controller/controller_utils.py` — `extract_ids_sources()` now builds a `{source_url → file_name}` map from retrieved document metadata; `format_result()` populates `Citation.source_file_name` via this map (post-LLM enrichment, no prompt change required).

---
## [2026-04-22]
### 0.10.1-rc4 (fix: situated context regression on per-row table chunks)

Fixed a retrieval regression introduced when `situated_context` was enabled on table data: per-row chunks all received the same generic table-level description, causing near-identical embeddings and making row-level retrieval unreliable.

- Added `_SITUATED_CONTEXT_TABLE_ROW_PROMPT` — a dedicated prompt for `element_type=table_rows` chunks that generates a specific natural-language sentence for each row referencing its actual cell values (e.g. *"Product Alpha (SKU: A001) costs €9.99."*). Previously, all rows shared one table-level description → embeddings collapsed → queries like *"price of product X"* returned wrong rows.
- Fixed `doc_context` selection for table chunks: `enrich_chunks_with_situated_context` now uses `metadata["source"]` (the source URL) instead of the first 150 characters of pipe-delimited markdown content, which was useless as document context.
- Added 12 unit tests in `tests/unit/shared/test_situated_context_table.py` covering prompt routing by `element_type`, source URL as doc_context, uniqueness of per-row responses, and integration with `enrich_chunks_with_situated_context`.

---
### 0.10.1-rc3 (table-aware web ingestion)

Optimization of the ingestion pipeline for web pages containing HTML tables (product catalogs, price lists, data tables). Tables are now preserved as structured chunks instead of being fragmented by the generic text splitter.

#### New files
- Added: `tilellm/modules/ingestion/table_chunker.py` — centralised table splitter with three strategies:
  - `atomic` — one chunk per table, always.
  - `per_row` — one chunk per data row, header repeated in each chunk. Best for product-catalog queries ("give me product X").
  - `adaptive` (default) — atomic if total table size ≤ `max_table_chars` (default 8 000 chars ≈ 2 000 tokens), split by row groups otherwise. Balances cost and retrieval quality.
  - Each chunk carries enriched metadata: `element_type` (`"table"` or `"table_rows"`), `chunk_type="table"`, `col_names`, `table_index`, `row_range` (string `"start-end"`, Pinecone-compatible), `total_rows`, `row_index`.
  - Fallback: if markdown parsing fails, returns the document intact (graceful degradation).

#### Modified files
- Changed: `tilellm/models/llm.py` — new `TableOptions` Pydantic model (`enable`, `strategy`, `max_table_chars`, `header_repeat`). Added `table_options: Optional[TableOptions] = None` to `ItemSingle`. Default `TableOptions()` applies adaptively with zero config; `{"enable": false}` restores previous behaviour.
- Changed: `tilellm/models/__init__.py` — exports `TableOptions` and `SituatedContextConfig`.
- Changed: `tilellm/tools/document_tools.py`:
  - `_extract_html_tables` — fallback when `tabulate` is not installed now produces proper pipe-delimited markdown natively (previously fell back to `df.to_string()` which has no `|` delimiters and broke downstream parsing).
  - `handle_chromium_loader` — captures raw HTML before the transformer runs; calls `_extract_html_tables(raw_html, url)` and appends table Documents to the result. Covers **scrape_type 3** (Html2TextTransformer) and **scrape_type 4** (BeautifulSoupTransformer), which previously discarded table structure entirely. scrape_type 0/1/2/5 already extracted tables; behaviour unchanged.
- Changed: `tilellm/store/pinecone/pinecone_repository_serverless.py` — `chunk_documents`: if `element_type == "table"` and `table_options.enable=True`, routes to `split_table_document` instead of `chunk_data_extended`. Non-table documents follow the existing splitter path unchanged.
- Changed: `tilellm/store/qdrant/qdrant_repository_local.py` — same bypass as above.
- Changed: `tilellm/store/milvus/milvus_repository.py` — same bypass as above.
- Changed: `tilellm/shared/situated_context.py`:
  - Added `_SITUATED_CONTEXT_TABLE_PROMPT` — table-aware prompt that injects `col_names` and instructs the LLM to describe the table's topic and data type (e.g. *"Product pricing table with columns SKU, Name, Price; this row describes product X."*).
  - `_generate_situated_context` now accepts optional `chunk_metadata: dict` and selects the table prompt when `element_type` is `"table"` or `"table_rows"`.
  - `enrich_chunks_with_situated_context` passes each doc's metadata to `_generate_situated_context` — fully backward-compatible (existing callers without table metadata use the original generic prompt).


With `strategy="per_row"`, each product row becomes an independent chunk with its header repeated — optimal for queries like *"what is the price of product X?"*. With `strategy="adaptive"` (default), small tables remain as a single chunk and large ones are split by row groups.

---
## [2026-04-12]
### 0.10.1-rc2 (unified ingestion — auto type detection)
- Added: `tilellm/models/document_type.py` — `DocumentType(str, Enum)` with 11 canonical values: `auto`, `url`, `pdf`, `docx`, `text`, `txt`, `md`, `xlsx`, `xls`, `csv`, `regex_custom`. Extends `str` so all existing string comparisons in repositories are backward-compatible without modification.
- Added: `tilellm/modules/ingestion/type_detector.py` — deterministic, zero-cost (no network, no LLM) auto-detection with 6-level priority chain: (1) magic bytes from base64 `file_content` — PDF (`%PDF`), OLE2/XLS (`\xD0\xCF\x11\xE0`), ZIP-ambiguous DOCX/XLSX falls through; (2) extension from `file_name`; (3) extension from `source` URL path (query-string/fragment stripped); (4) URL heuristic — `http(s)` source with no known file extension → `url`; (5) direct text content (`content` field, no source) → `text`; (6) no signal → `None`. Public API: `detect_document_type(source, content, file_name, file_content)` and `resolve_item_type(current_type, ...)`.
- Changed: `ItemSingle.type` — from `str | None` to `Optional[DocumentType]` with full description of accepted values. Pydantic validates and rejects unknown values; `"auto"` and `None` both trigger auto-detection. Backward-compatible — callers that already pass `"pdf"`, `"docx"` etc. are unaffected.
- Added: `DocumentType.TEXT = "text"` — new explicit type for direct text content passed in the `content` field. Routed to the `process_auto_detected_text` path in all repositories (markdown, tabular, or plain sub-format auto-detected at chunk time). Previously only worked implicitly via `type=None`; now first-class.
- Changed: `tilellm/modules/ingestion/controllers.py` — `unified_ingestion` calls `_resolve_type(item)` at entry, applies `model_copy(update={"type": resolved})` to propagate the resolved type, logs auto-detection events. Routing conditions updated to `DocumentType` enum comparisons. Fixed latent bug: `type=None + source="file.pdf"` previously fell into the direct-text `else` branch (attempting `process_auto_detected_text(content=None)`); now correctly resolved to `pdf` before routing.
- Added: `tests/unit/modules/ingestion/test_type_detector.py` — 51 tests across `TestMagicFromBase64`, `TestExtFromPath`, `TestDetectDocumentType`, `TestResolveItemType`. All pure-function, no network, no GPU.

---
## [2026-04-02]
### 0.10.0
- Feature: Unified ingestion for all document types (PDF, DOCX, HTML with scraping) through the new endpoint `POST /api/ingestion`. Automatic routing to the appropriate pipeline (OCR, hybrid, default) based on the provided options.
- Feature: Automatic extraction of all HTML tables from web pages in every scraping path (Trafilatura, Playwright+BS4, Playwright+stealth, fallback selector). For each table, a specific document is generated with metadata (columns, index, type), converted to markdown, with structure preserved.
- Feature: Extraction of embedded images for DOCX documents — the loader now finds both modern (`w:drawing`) and legacy (`w:pict`) images, stores them locally, and generates automatic captions with LLM vision. Captions are associated with/around the relevant text paragraphs and referenced in the `"ref_images"` metadata.
- Improvement: Introduced the `CommonChunkMetadata` Pydantic schema as the single metadata contract between pipeline and vectorstore, with safe defaults and forward compatibility.
- Feature: Optional generation of “situated context” via LLM for each ingested chunk, following Anthropic's Contextual Retrieval technique. Allows configuration of LLM provider and model directly from the API call.
- Fix: Resolved bug in hybrid Pinecone upsert caused by unwanted inclusion of `namespace: None` in metadata (which resulted in error 400).
- Fix: Improved chat history management for gpt-5.x and situational context for text content.
- Docs: Updated roadmap and unified ingestion documentation detailing implementation status, pipelines, technical details of image and table extraction, OCR engines, and notes on future developments.

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
