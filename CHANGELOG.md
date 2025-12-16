# **TILEDESK LLM - Changelog**

### **Authors**:
    * Gianluca Lorenzo
    * Andrea Sponziello
### **Copyright**: Tiledesk SRL

---

## [2025-12-13]
### 0.70.1-rc2
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