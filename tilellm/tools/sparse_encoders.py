from threading import Lock
from typing import Dict, Optional, Union, List, TYPE_CHECKING
import logging
from collections import OrderedDict

if TYPE_CHECKING:
    from tilellm.models.llm import TEIConfig

try:
    import torch
except ImportError:
    torch = None

try:
    from pinecone_text.sparse import SpladeEncoder
except ImportError:
    SpladeEncoder = None

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

# Shared GPU concurrency primitives (same lock used by reranker → one lock
# serializes forward pass across ALL local GPU models in the process).
from tilellm.tools._gpu_concurrency import (
    GLOBAL_GPU_LOCK,
    is_cuda_oom,
    cuda_empty_cache_safe,
    build_token_budget_batches,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _estimate_text_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word. Same heuristic as reranker."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def _chunk_text(text: str, max_words: int = 300) -> List[str]:
    """
    Split text into chunks of at most `max_words` words.

    Args:
        text: Input text.
        max_words: Words per chunk. Use 300 for Splade (~390 tokens < 512),
                   much higher (1500+) for BGE-M3 which supports 8192 tokens.
    """
    if not text:
        return [""]

    words = text.split()
    if len(words) <= max_words:
        return [text]

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def _merge_sparse_vectors(vectors: List[Dict[str, List]]) -> Dict[str, List]:
    if not vectors:
        return {"indices": [], "values": []}

    merged = {}
    for vec in vectors:
        if not vec or 'indices' not in vec or 'values' not in vec:
            continue
        for idx, val in zip(vec['indices'], vec['values']):
            # Max pooling
            if idx in merged:
                if val > merged[idx]:
                    merged[idx] = val
            else:
                merged[idx] = val

    sorted_indices = sorted(merged.keys())
    return {
        "indices": sorted_indices,
        "values": [merged[idx] for idx in sorted_indices]
    }


def _run_sparse_batched_with_oom_recovery(
    contents: List[str],
    text_lengths: List[int],
    encode_batch_fn,
    *,
    token_budget: int,
    max_items_per_batch: int,
    max_oom_retries: int = 3,
    log: logging.Logger = logger,
) -> List[Dict[str, List]]:
    """
    Generic token-budget batched sparse encoder with CUDA OOM recovery.

    Mirrors TileReranker._predict_local_with_oom_recovery but generalized
    for single-text items (sparse encoders take texts, not pairs).

    Args:
        contents: Texts to encode, in original order.
        text_lengths: Token-length estimate for each text (same order).
        encode_batch_fn: Callable that takes a list of texts and returns a
                         list of sparse vectors (same length). Called INSIDE
                         the GLOBAL_GPU_LOCK by this helper.
        token_budget: Initial token budget per batch.
        max_items_per_batch: Hard cap on items per batch.
        max_oom_retries: Number of OOM recovery retries (halving budget each).
        log: Logger for warnings / errors.

    Returns:
        List of sparse vectors in the original input order.

    Raises:
        torch.cuda.OutOfMemoryError if OOM persists after max_oom_retries.
    """
    n = len(contents)
    if n == 0:
        return []

    results: List[Optional[Dict[str, List]]] = [None] * n
    processed: set = set()

    current_budget = token_budget
    oom_retries = 0

    while len(processed) < n:
        # Indices not yet processed
        remaining_idx = [i for i in range(n) if i not in processed]
        remaining_lengths = [text_lengths[i] for i in remaining_idx]

        # Build batches in the "remaining" space, then translate to original indices
        batches_local = build_token_budget_batches(
            remaining_lengths,
            token_budget=current_budget,
            max_pairs_per_batch=max_items_per_batch,
        )
        batches = [[remaining_idx[j] for j in b] for b in batches_local]

        log.debug(
            f"Sparse encoder batching: {len(remaining_idx)} texts → {len(batches)} batches "
            f"(budget={current_budget}, max_items={max_items_per_batch})"
        )

        try:
            for batch in batches:
                batch_texts = [contents[i] for i in batch]
                # GLOBAL_GPU_LOCK serializes forward pass across all local models.
                # Acquired per-batch (not per-call) so other threads don't starve.
                with GLOBAL_GPU_LOCK:
                    batch_results = encode_batch_fn(batch_texts)

                if len(batch_results) != len(batch):
                    raise RuntimeError(
                        f"encode_batch_fn returned {len(batch_results)} vectors "
                        f"for {len(batch)} inputs"
                    )

                for j, orig_idx in enumerate(batch):
                    results[orig_idx] = batch_results[j]
                    processed.add(orig_idx)

            # All batches completed successfully
            break

        except Exception as e:
            if not is_cuda_oom(e):
                raise

            oom_retries += 1
            if oom_retries > max_oom_retries:
                log.error(
                    f"CUDA OOM: max retries ({max_oom_retries}) exceeded, "
                    f"budget={current_budget}, processed {len(processed)}/{n}"
                )
                raise

            new_budget = max(256, current_budget // 2)
            log.warning(
                f"CUDA OOM (retry {oom_retries}/{max_oom_retries}): "
                f"halving token budget {current_budget} → {new_budget}"
            )
            cuda_empty_cache_safe()
            current_budget = new_budget
            # while loop will rebuild batches only for remaining texts

    assert all(r is not None for r in results), \
        f"Missing sparse vectors: {sum(1 for r in results if r is None)} of {n}"
    return results  # type: ignore[return-value]


# =============================================================================
# Splade encoder (local, GPU)
# =============================================================================

class TiledeskSpladeEncoder:
    """
    Local Splade sparse encoder backed by pinecone_text.SpladeEncoder.

    Optimizations (vs baseline):
      - fp16 on CUDA at load time (~2x speedup on Ampere+ GPUs)
      - warmup on load (triggers cuDNN autotune off the hot path)
      - GLOBAL_GPU_LOCK serializes forward pass across threads
      - Token-budget batching replaces fixed-size batching (conservative memory)
      - OOM recovery with halving budget
      - No torch.cuda.empty_cache() in happy path
    """

    # Tuning knobs — class-level so they apply to cached instances consistently
    _use_fp16: bool = True
    _warmup_on_load: bool = True

    def __init__(
        self,
        token_budget_per_batch: int = 2048,
        max_items_per_batch: int = 16,
        max_oom_retries: int = 3,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
        self.token_budget_per_batch = token_budget_per_batch
        self.max_items_per_batch = max_items_per_batch
        self.max_oom_retries = max_oom_retries

        self.logger.info(f"Init of SpladeEncoder on device: {self.device}")
        self.splade = SpladeEncoder(device=self.device)

        # fp16 conversion on CUDA: ~2x speedup, halved memory
        if self._use_fp16 and self.device == 'cuda' and torch is not None:
            try:
                # SpladeEncoder wraps a HuggingFace MLM model at .model
                if hasattr(self.splade, "model") and self.splade.model is not None:
                    self.splade.model = self.splade.model.half()
                    self.logger.info("SpladeEncoder converted to fp16")
            except Exception as fp16_err:
                self.logger.warning(
                    f"fp16 conversion failed for SpladeEncoder, using fp32: {fp16_err}"
                )

        # Warmup: one dummy encode_documents call to trigger cuDNN autotune.
        # Moves the ~200-500 ms autotune cost off the first real query.
        if self._warmup_on_load and self.device == 'cuda':
            try:
                _ = self.splade.encode_documents(["warmup text"])
                self.logger.info("SpladeEncoder warmup completed")
            except Exception as warmup_err:
                self.logger.warning(f"Warmup encode failed for SpladeEncoder: {warmup_err}")

        self.logger.info("SpladeEncoder loaded")

    def encode_documents_with_batch(
        self,
        contents: List[str],
        batch_size: Optional[int] = 10,
    ) -> List[Dict[str, List]]:
        """
        Encode texts with token-budget batching + OOM recovery.

        Legacy `batch_size` is treated as a cap on items per batch (the
        token budget provides the memory-aware limit). Pass `batch_size=None`
        or 0 to encode everything in a single call (no batching wrapper).
        """
        if not contents:
            return []

        # Legacy path: no batching
        if not batch_size:
            self.logger.info(
                f"Encoding {len(contents)} documents with SpladeEncoder (single call)"
            )
            return self.splade.encode_documents(contents)

        # Use batch_size as cap; token_budget provides the memory-aware bound
        max_items = min(batch_size, self.max_items_per_batch)

        text_lengths = [_estimate_text_tokens(t) for t in contents]

        def encode_batch(batch_texts: List[str]) -> List[Dict[str, List]]:
            return self.splade.encode_documents(batch_texts)

        self.logger.debug(
            f"Encoding {len(contents)} documents with SpladeEncoder "
            f"(token_budget={self.token_budget_per_batch}, max_items={max_items})"
        )
        return _run_sparse_batched_with_oom_recovery(
            contents=contents,
            text_lengths=text_lengths,
            encode_batch_fn=encode_batch,
            token_budget=self.token_budget_per_batch,
            max_items_per_batch=max_items,
            max_oom_retries=self.max_oom_retries,
            log=self.logger,
        )

    def encode_documents(
        self,
        contents: List[str],
        batch_size: Optional[int] = 10,
    ) -> List[Dict[str, List]]:
        if not contents:
            return []

        all_chunks = []
        doc_map = []  # (start_index, length)

        # Splade max_seq ~256/512 → 300 words per chunk (~390 tokens)
        for text in contents:
            chunks = _chunk_text(text, max_words=300)
            doc_map.append((len(all_chunks), len(chunks)))
            all_chunks.extend(chunks)

        chunk_results = self.encode_documents_with_batch(all_chunks, batch_size=batch_size)

        results = []
        for start, length in doc_map:
            doc_vectors = chunk_results[start: start + length]
            results.append(_merge_sparse_vectors(doc_vectors))

        return results

    def encode_queries(self, query: str) -> Dict[str, List]:
        self.logger.debug(f"Encoding query: '{query}'")
        with GLOBAL_GPU_LOCK:
            return self.splade.encode_queries(query)


# =============================================================================
# BGE-M3 encoder (local, GPU, dense + sparse + colbert)
# =============================================================================

class TiledeskBGEM3:
    """
    Local BGE-M3 encoder backed by FlagEmbedding.BGEM3FlagModel.

    BGE-M3 ships its own fp16 support (use_fp16=True), so we don't apply
    .half() manually — the library handles mixed precision internally
    (using autocast for numerically sensitive ops).

    Optimizations (vs baseline):
      - warmup on load
      - GLOBAL_GPU_LOCK serializes forward pass across threads
      - Token-budget batching + OOM recovery
      - No torch.cuda.empty_cache() in happy path
      - Chunking uses 1500 words (~2000 tokens) given 8192-token context
    """

    _warmup_on_load: bool = True

    def __init__(
        self,
        token_budget_per_batch: int = 2048,
        max_items_per_batch: int = 8,
        max_oom_retries: int = 3,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
        self.use_fp16_bool = True if self.device == 'cuda' else False
        self.token_budget_per_batch = token_budget_per_batch
        self.max_items_per_batch = max_items_per_batch
        self.max_oom_retries = max_oom_retries

        self.logger.info(
            f"Init of BGEM3FlagModel ('BAAI/bge-m3') on device: {self.device}, "
            f"use_fp16: {self.use_fp16_bool}"
        )
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=self.use_fp16_bool)

        # Warmup: one dummy encode to trigger cuDNN autotune off the hot path.
        if self._warmup_on_load and self.device == 'cuda':
            try:
                _ = self.model.encode(
                    ["warmup text"],
                    return_dense=True, return_sparse=True, return_colbert_vecs=False,
                )
                self.logger.info("BGEM3FlagModel warmup completed")
            except Exception as warmup_err:
                self.logger.warning(f"Warmup encode failed for BGEM3: {warmup_err}")

        self.logger.info("BGEM3FlagModel loaded.")

    def encode_documents(
        self,
        contents: List[str],
        batch_size: Optional[int] = 10,
    ) -> List[Dict[str, List]]:
        if not contents:
            return []

        # Legacy path: single call, no batching wrapper
        if not batch_size:
            self.logger.debug(f"Encoding {len(contents)} documents with BGEM3FlagModel")
            with GLOBAL_GPU_LOCK:
                output = self.model.encode(
                    contents,
                    return_dense=True, return_sparse=True, return_colbert_vecs=False,
                )
            return self._convert_sparse_vectors(output['lexical_weights'])

        max_items = min(batch_size, self.max_items_per_batch)
        text_lengths = [_estimate_text_tokens(t) for t in contents]

        def encode_batch(batch_texts: List[str]) -> List[Dict[str, List]]:
            output = self.model.encode(
                batch_texts,
                return_dense=True, return_sparse=True, return_colbert_vecs=False,
            )
            return self._convert_sparse_vectors(output['lexical_weights'])

        self.logger.debug(
            f"Encoding {len(contents)} documents with BGEM3 "
            f"(token_budget={self.token_budget_per_batch}, max_items={max_items})"
        )
        return _run_sparse_batched_with_oom_recovery(
            contents=contents,
            text_lengths=text_lengths,
            encode_batch_fn=encode_batch,
            token_budget=self.token_budget_per_batch,
            max_items_per_batch=max_items,
            max_oom_retries=self.max_oom_retries,
            log=self.logger,
        )

    def _convert_sparse_vectors(self, lexical_weights: List[Dict]) -> List[Dict[str, List]]:
        return [{
            'indices': [int(k) for k in doc_dict.keys()],
            'values': [float(doc_dict[k]) for k in doc_dict.keys()]
        } for doc_dict in lexical_weights]

    def encode_queries(self, query: str) -> Dict[str, List]:
        self.logger.debug(f"Encoding query: '{query}'")
        with GLOBAL_GPU_LOCK:
            output = self.model.encode(
                [query],
                return_dense=False, return_sparse=True, return_colbert_vecs=False,
            )
        return self._convert_sparse_vectors(output['lexical_weights'])[0]


# =============================================================================
# TEI sparse encoder (remote, no GPU concurrency concerns)
# =============================================================================

class TEISparseEncoder:
    def __init__(self, config: "TEIConfig"):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.url = config.url.rstrip("/") if config.url else ""
        self.headers = config.custom_headers if config.custom_headers else {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self.logger.info(f"Init of TEISparseEncoder with url: {self.url}")

    def _call_tei(self, texts: List[str]) -> List[Dict[str, List]]:
        import httpx
        try:
            payload = {
                "inputs": texts,
            }
            if self.config.name:
                 payload["model"] = self.config.name

            response = httpx.post(f"{self.url}/embed_sparse", json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            results = []
            for doc_vector in data:
                indices = []
                values = []
                for item in doc_vector:
                    indices.append(item["index"])
                    values.append(item["value"])
                results.append({"indices": indices, "values": values})
            return results

        except Exception as e:
            self.logger.error(f"Error calling TEI: {e}")
            raise e

    def encode_documents_with_batch(self, contents: List[str], batch_size: Optional[int] = 8) -> List[Dict[str, List]]:
        import time
        import httpx

        if not batch_size:
            batch_size = 8

        results = []
        i = 0
        current_batch_size = batch_size
        while i < len(contents):
            batch = contents[i:i + current_batch_size]
            self.logger.info(f"Processing sparse encoding batch {i//current_batch_size + 1}/{(len(contents)-1)//current_batch_size + 1} with {len(batch)} documents (batch size: {current_batch_size})")
            batch_retry_count = 0
            max_retries = 3
            batch_success = False

            while batch_retry_count < max_retries and not batch_success:
                try:
                    batch_results = self._call_tei(batch)
                    results.extend(batch_results)
                    batch_success = True
                    i += current_batch_size

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 413 and current_batch_size > 1:
                        # Payload too large, reduce batch size and retry
                        new_batch_size = max(1, current_batch_size // 2)
                        self.logger.warning(f"Payload too large (413) for batch size {current_batch_size}. Reducing to {new_batch_size}")
                        current_batch_size = new_batch_size
                        batch = contents[i:i + current_batch_size]
                        batch_retry_count += 1
                        time.sleep(0.5 * batch_retry_count)  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"HTTP error calling TEI sparse encoder: {e}")
                        raise e
                except Exception as e:
                    self.logger.error(f"Error calling TEI sparse encoder: {e}")
                    raise e

            if not batch_success:
                raise RuntimeError(f"Failed to process batch after {max_retries} retries")

        return results

    def encode_documents(self, contents: List[str], batch_size: Optional[int] = 8) -> List[Dict[str, List]]:
        if not contents:
            return []

        all_chunks = []
        doc_map = []

        for text in contents:
            # We don't use transformers here to keep it optional.
            # _chunk_text will fallback to tiktoken or word split.
            chunks = _chunk_text(text)
            doc_map.append((len(all_chunks), len(chunks)))
            all_chunks.extend(chunks)

        chunk_results = self.encode_documents_with_batch(all_chunks, batch_size=batch_size)

        results = []
        for start, length in doc_map:
            doc_vectors = chunk_results[start : start + length]
            results.append(_merge_sparse_vectors(doc_vectors))

        return results

    def encode_queries(self, query: str) -> Dict[str, List]:
        results = self._call_tei([query])
        return results[0]


# =============================================================================
# LRU encoder cache (process-wide singleton)
# =============================================================================

class TiledeskSparseEncoders:
    """
    LRU cache for sparse encoders (max 2 models).

    Eviction properly releases VRAM by deleting the old encoder and
    calling cuda.empty_cache(), otherwise PyTorch retains allocations
    until the next GC cycle.
    """

    _encoder_cache: OrderedDict = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)
    _cache_lock = Lock()

    def __init__(self, model_name: Union[str, "TEIConfig"]):
        if hasattr(model_name, "provider") and model_name.provider == "tei":
             self.model_name = "tei_" + model_name.url
             self.config = model_name
        else:
            self.model_name = model_name.lower()
            self.config = None

        self.encoder = self._get_cached_encoder(self.model_name, self.config)
        self.device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'

    @classmethod
    def _get_cached_encoder(
        cls,
        model_name: str,
        config: Optional["TEIConfig"] = None,
    ) -> Union[TiledeskSpladeEncoder, TiledeskBGEM3, TEISparseEncoder]:
        with cls._cache_lock:
            # Cache hit
            if model_name in cls._encoder_cache:
                cls._logger.info(f"Reusing cached instance of: {model_name}")
                # Move to end (MRU)
                encoder = cls._encoder_cache.pop(model_name)
                cls._encoder_cache[model_name] = encoder
                return encoder

            # Cache miss — load new encoder
            if config and hasattr(config, "provider") and config.provider == "tei":
                cls._logger.info("Creating new TEISparseEncoder instance")
                encoder = TEISparseEncoder(config)
            elif model_name == "splade":
                if SpladeEncoder is None:
                    raise ImportError("Pinecone SpladeEncoder is not available. Install 'ml' extras.")
                cls._logger.info("Creating new SpladeEncoder instance")
                encoder = TiledeskSpladeEncoder()
            elif model_name == "bge-m3":
                if BGEM3FlagModel is None:
                    raise ImportError("BGEM3FlagModel is not available. Install 'ml' extras.")
                cls._logger.info("Creating new BGEM3 instance")
                encoder = TiledeskBGEM3()
            else:
                raise ValueError(f"Unsupported model: {model_name}. Use 'splade', 'bge-m3' or TEIConfig.")

            # LRU eviction — properly release VRAM on evicted model
            if len(cls._encoder_cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._encoder_cache))
                oldest_encoder = cls._encoder_cache.pop(oldest_key)
                cls._logger.info(f"Removing oldest encoder from cache: {oldest_key}")
                del oldest_encoder
                cuda_empty_cache_safe()

            cls._encoder_cache[model_name] = encoder
        return encoder

    def encode_documents(self, contents: List[str], batch_size: Optional[int] = 10) -> List[Dict[str, List]]:
        if not self.encoder:
            raise ValueError("Encoder not initialized")

        if batch_size and batch_size <= 0:
            raise ValueError("Batch size must be positive integer")

        return self.encoder.encode_documents(contents, batch_size)

    def encode_queries(self, query: str) -> Dict[str, List]:
        if not self.encoder:
            raise ValueError("Encoder not initialized")
        return self.encoder.encode_queries(query)

    @classmethod
    def clear_cache(cls):
        """Clear encoder cache and release GPU memory (thread-safe)."""
        cls._logger.warning("Clearing encoder cache and GPU memory")
        with cls._cache_lock:
            cls._encoder_cache.clear()
        cuda_empty_cache_safe()
