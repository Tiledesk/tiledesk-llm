"""
Shared GPU concurrency & batching primitives for local inference modules.

Used by:
  - tilellm/tools/reranker.py (CrossEncoder local path)
  - tilellm/tools/sparse_encoders.py (Splade / BGE-M3 local paths)

Design principles:
  - GLOBAL_GPU_LOCK: one process-wide Lock that serializes forward pass on GPU
    across ALL local models (reranker + sparse encoders). Without it, multiple
    threads (FastAPI async workers via asyncio.to_thread, or TaskIQ workers)
    can collide on the same CUDA device, causing OOM from concurrent
    allocations on the same model. Acquired PER-BATCH (not per-call) to keep
    latency fair under contention.
  - build_token_budget_batches: conservative TEI-inspired packing. Single
    source of truth; re-exported from reranker.py for backward compatibility.
  - is_cuda_oom / cuda_empty_cache_safe: tiny helpers that tolerate
    `torch is None` (ml extras not installed).

This module MUST NOT import heavy ML libraries (sentence-transformers,
FlagEmbedding, ...). It only touches torch, and only lazily.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, List

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Dedicated single-thread GPU executor
# ---------------------------------------------------------------------------

_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")
"""
Single-thread executor for all local GPU inference (sparse encoders + reranker).

Guarantees that only ONE thread ever executes a GPU forward pass at any time,
without requiring an explicit lock. Callers queue naturally via the executor's
internal work queue — no thread pool growth, no lock contention, FIFO ordering.

Replace asyncio.to_thread / run_in_executor(None, ...) with run_on_gpu() for
any operation that runs on the GPU (SPLADE, BGE-M3, CrossEncoder).
Do NOT use for remote I/O (TEI HTTP calls, Pinecone reranker API) — those
should stay in the default thread pool so they don't block the GPU thread.
"""


async def run_on_gpu(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Run a synchronous GPU function in the dedicated single-thread GPU executor.

    Ensures only one forward pass runs at a time without explicit locking.
    Supports both positional and keyword arguments.

    Usage:
        result = await run_on_gpu(encoder.encode_documents, contents, batch_size)
        score  = await run_on_gpu(model.predict, pairs, convert_to_numpy=True)
    """
    loop = asyncio.get_running_loop()
    call = partial(fn, *args, **kwargs) if kwargs else partial(fn, *args)
    return await loop.run_in_executor(_GPU_EXECUTOR, call)


# ---------------------------------------------------------------------------
# Process-wide GPU lock (kept for backward compatibility, no longer used internally)
# ---------------------------------------------------------------------------

GLOBAL_GPU_LOCK = Lock()
"""
Deprecated: use run_on_gpu() instead.

Kept to avoid ImportError in code that still references it.
"""


# ---------------------------------------------------------------------------
# CUDA helpers (torch-optional)
# ---------------------------------------------------------------------------

def is_cuda_oom(exc: BaseException) -> bool:
    """
    True iff the exception is a torch.cuda.OutOfMemoryError.

    Tolerates torch being unavailable (ml extras not installed) — in that
    case it always returns False, because no CUDA OOM can have occurred.
    """
    return (
        torch is not None
        and hasattr(torch, "cuda")
        and hasattr(torch.cuda, "OutOfMemoryError")
        and isinstance(exc, torch.cuda.OutOfMemoryError)
    )


def cuda_empty_cache_safe() -> None:
    """
    Call torch.cuda.empty_cache() if CUDA is available, no-op otherwise.

    This is expensive (sync + allocator flush, 100-200 ms) — ONLY call on
    OOM recovery and model eviction, NEVER in the happy path.
    """
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Token-budget batching (TEI-inspired)
# ---------------------------------------------------------------------------

def build_token_budget_batches(
    pair_lengths: List[int],
    token_budget: int = 2048,
    max_pairs_per_batch: int = 16,
) -> List[List[int]]:
    """
    Conservative token-budget packing in stile TEI.

    Strategy:
      1. Sort items by length ascending — groups short items together and
         minimizes padding waste within each batch.
      2. Greedy packing: add to the current batch while
         ``(len(batch) + 1) * new_max_len <= token_budget`` AND
         ``len(batch) + 1 <= max_pairs_per_batch``. The cost of a forward
         is ``batch_size * max_seq_len``, so we use that product as a
         conservative estimate.
      3. Items whose own length exceeds the budget are admitted as
         singleton batches — the tokenizer will apply its own max-length
         truncation, we just make sure they don't get dropped.

    Args:
        pair_lengths: Per-item token-length estimates (original order).
                      Name says "pair" for backward compatibility with the
                      original reranker usage; semantically it's just
                      "length estimates for N items".
        token_budget: Max ``batch_size * max_seq_len`` per batch.
        max_pairs_per_batch: Hard cap on items per batch.

    Returns:
        List of batches, each a list of ORIGINAL indices (0-based).
        Invariants: every index appears exactly once, no batch is empty.

    This is a pure function — no GPU, no torch, no IO. Testable in
    isolation (see tests/unit/tools/test_local_reranker_batching.py).
    """
    if not pair_lengths:
        return []

    # Sort by length ASC, keep mapping back to original index
    sorted_indices = sorted(range(len(pair_lengths)), key=lambda i: pair_lengths[i])

    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_max_len: int = 0

    for idx in sorted_indices:
        length = pair_lengths[idx]
        new_max_len = max(current_max_len, length)
        projected_cost = (len(current_batch) + 1) * new_max_len

        would_exceed_budget = projected_cost > token_budget
        would_exceed_items = len(current_batch) >= max_pairs_per_batch

        if current_batch and (would_exceed_budget or would_exceed_items):
            # Close current batch
            batches.append(current_batch)
            current_batch = []
            current_max_len = 0

        current_batch.append(idx)
        current_max_len = max(current_max_len, length)

    if current_batch:
        batches.append(current_batch)

    return batches
