"""
Unit tests for local CrossEncoder batching helpers in tilellm/tools/reranker.py.

Nessuna dipendenza da torch, sentence-transformers o GPU reale.
Le funzioni testate sono pure (solo Python/stdlib) — importabili senza
installare il gruppo opzionale 'ml'.

Copertura:
  - estimate_pair_tokens
  - build_token_budget_batches
  - _predict_local_with_oom_recovery (via mock CrossEncoder)
"""
import threading
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from tilellm.tools.reranker import (
    estimate_tokens,
    estimate_pair_tokens,
    build_token_budget_batches,
)


# ============================================================================
# estimate_pair_tokens
# ============================================================================

class TestEstimatePairTokens:
    def test_basic_sum(self):
        """estimate_pair_tokens == estimate_tokens(q) + estimate_tokens(t) + overhead"""
        q = "test query about something interesting"
        t = "this is a document that talks about interesting topics"
        expected = estimate_tokens(q) + estimate_tokens(t) + 8
        assert estimate_pair_tokens(q, t) == expected

    def test_custom_overhead(self):
        q = "hello"
        t = "world"
        overhead = 16
        expected = estimate_tokens(q) + estimate_tokens(t) + overhead
        assert estimate_pair_tokens(q, t, overhead=overhead) == expected

    def test_empty_text(self):
        result = estimate_pair_tokens("query", "")
        assert result == estimate_tokens("query") + 0 + 8

    def test_empty_both(self):
        result = estimate_pair_tokens("", "")
        assert result == 8  # solo overhead


# ============================================================================
# build_token_budget_batches
# ============================================================================

class TestBuildTokenBudgetBatches:

    def test_empty_input(self):
        assert build_token_budget_batches([]) == []

    def test_all_indices_covered(self):
        """Ogni indice deve apparire esattamente una volta."""
        lengths = [100, 200, 50, 300, 150, 80, 250, 120, 90, 170]
        batches = build_token_budget_batches(lengths, token_budget=2048, max_pairs_per_batch=16)
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(len(lengths)))

    def test_no_empty_batch(self):
        lengths = [50] * 10
        batches = build_token_budget_batches(lengths, token_budget=2048, max_pairs_per_batch=4)
        for batch in batches:
            assert len(batch) > 0

    def test_respects_max_pairs(self):
        """Nessun batch deve superare max_pairs_per_batch."""
        lengths = [50] * 20
        max_pairs = 4
        batches = build_token_budget_batches(lengths, token_budget=100_000, max_pairs_per_batch=max_pairs)
        for batch in batches:
            assert len(batch) <= max_pairs

    def test_respects_token_budget(self):
        """batch_size * max_len_in_batch non deve superare il budget (nella maggior parte dei casi)."""
        lengths = [500] * 10  # ogni pair costa ~500 token
        # budget = 1000 → max 2 pair per batch (2 * 500 = 1000)
        batches = build_token_budget_batches(lengths, token_budget=1000, max_pairs_per_batch=100)
        for batch in batches:
            max_len = max(lengths[i] for i in batch)
            cost = len(batch) * max_len
            # Cost può essere ≤ budget, o > budget solo se il batch è singleton
            # (pair che supera il budget da sola → ammessa in batch singleton)
            assert cost <= 1000 or len(batch) == 1

    def test_oversized_single_pair_gets_own_batch(self):
        """Una pair che supera il budget da sola deve finire in un batch singleton."""
        lengths = [10, 10, 5000, 10]  # indice 2 ha lunghezza enorme
        batches = build_token_budget_batches(lengths, token_budget=200, max_pairs_per_batch=16)
        # Indice 2 deve essere da solo nel suo batch
        for batch in batches:
            if 2 in batch:
                assert len(batch) == 1, f"Oversized pair deve essere singleton, trovato batch: {batch}"

    def test_sort_by_length_groups_similar_lengths(self):
        """Le pair corte devono stare insieme e le lunghe insieme."""
        # 5 pair corte (50 token) + 5 pair lunghe (400 token)
        lengths = [400, 50, 400, 50, 400, 50, 400, 50, 400, 50]
        # budget = 500 → le pair corte (50) ci stanno molte insieme; le lunghe (400) poche
        batches = build_token_budget_batches(lengths, token_budget=500, max_pairs_per_batch=100)

        # Recupera le lunghezze originali per ogni batch
        short_idxs = {i for i, l in enumerate(lengths) if l == 50}
        long_idxs  = {i for i, l in enumerate(lengths) if l == 400}

        for batch in batches:
            batch_set = set(batch)
            # Un batch non deve mescolare short e long SE possono stare separati
            has_short = bool(batch_set & short_idxs)
            has_long  = bool(batch_set & long_idxs)
            # Dopo il sort-by-length, short e long non dovrebbero mescolarsi
            assert not (has_short and has_long), \
                f"Batch mescola pair corte e lunghe: {batch}"

    def test_single_pair(self):
        batches = build_token_budget_batches([100], token_budget=2048, max_pairs_per_batch=16)
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_exact_budget_boundary(self):
        """4 pair da 100 con budget=400 e max_pairs=4 → 1 solo batch."""
        lengths = [100, 100, 100, 100]
        batches = build_token_budget_batches(lengths, token_budget=400, max_pairs_per_batch=4)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_budget_forces_split(self):
        """4 pair da 100 con budget=200 → 2 batch (2+2)."""
        lengths = [100, 100, 100, 100]
        batches = build_token_budget_batches(lengths, token_budget=200, max_pairs_per_batch=100)
        # budget 200 / 100 token per pair = 2 pair max per batch
        for batch in batches:
            assert len(batch) <= 2


# ============================================================================
# TileReranker._predict_local_with_oom_recovery  (via mock CrossEncoder)
# ============================================================================

def _make_reranker_with_mock_ce(token_budget=2048, max_pairs=16, max_oom_retries=3):
    """
    Costruisce un TileReranker con un CrossEncoder mockato (nessun torch, nessun modello reale).
    """
    from tilellm.tools.reranker import TileReranker

    # Pulisce cache e bypassa il caricamento del modello reale
    TileReranker.clear_cache()

    mock_ce = MagicMock()
    # predict restituisce un array numpy di score (1.0 * indice nel batch, come placeholder)
    mock_ce.predict = MagicMock(
        side_effect=lambda pairs, **kwargs: np.array([float(i) for i in range(len(pairs))])
    )

    with patch("tilellm.tools.reranker.CrossEncoder", return_value=mock_ce):
        with patch.object(TileReranker, "_use_fp16", False):
            with patch.object(TileReranker, "_warmup_on_load", False):
                reranker = TileReranker(
                    "mock-model",
                    token_budget_per_batch=token_budget,
                    max_pairs_per_batch=max_pairs,
                    max_oom_retries=max_oom_retries,
                )
    # Inietta direttamente il mock nel modello cachato
    reranker.model = mock_ce
    return reranker, mock_ce


class TestPredictLocalWithOomRecovery:

    def test_all_pairs_scored(self):
        """Tutti gli indici devono avere uno score nel risultato."""
        reranker, mock_ce = _make_reranker_with_mock_ce()
        pairs = [("query", f"doc {i}") for i in range(20)]
        pair_lengths = [estimate_pair_tokens("query", f"doc {i}") for i in range(20)]

        scores = reranker._predict_local_with_oom_recovery(
            pairs=pairs, pair_lengths=pair_lengths,
            token_budget=2048, max_pairs=16
        )
        assert len(scores) == 20
        assert all(isinstance(s, float) for s in scores)

    def test_scores_in_original_order(self):
        """
        I punteggi devono essere nell'ordine originale dei pair, non nell'ordine
        interno dei batch (che è ordinato per lunghezza).
        """
        from tilellm.tools.reranker import TileReranker

        TileReranker.clear_cache()

        # Mock che restituisce lo score come lunghezza del testo del secondo elemento
        call_counter = {"n": 0}

        def predict_side_effect(pairs, **kwargs):
            # Ritorna l'indice nella sequenza globale, per tracciabilità
            scores = np.array([float(len(p[1])) for p in pairs])
            return scores

        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(side_effect=predict_side_effect)

        with patch("tilellm.tools.reranker.CrossEncoder", return_value=mock_ce):
            with patch.object(TileReranker, "_use_fp16", False):
                with patch.object(TileReranker, "_warmup_on_load", False):
                    reranker = TileReranker("mock-model", token_budget_per_batch=2048, max_pairs_per_batch=16)
        reranker.model = mock_ce

        texts = ["short", "a very long text here with many words", "mid length text", "tiny"]
        pairs = [("q", t) for t in texts]
        pair_lengths = [estimate_pair_tokens("q", t) for _, t in pairs]

        scores = reranker._predict_local_with_oom_recovery(
            pairs=pairs, pair_lengths=pair_lengths,
            token_budget=4096, max_pairs=16
        )

        # Lo score di ogni pair deve corrispondere alla lunghezza del testo originale
        expected = [float(len(t)) for t in texts]
        assert scores == expected, f"Expected {expected}, got {scores}"

    def test_oom_recovery_halves_budget(self):
        """
        Su OOM, il metodo deve dimezzare il budget e riprovare con i pair rimasti.
        Usa torch.cuda.OutOfMemoryError se disponibile, altrimenti RuntimeError come fallback.
        """
        import tilellm.tools.reranker as rm
        from tilellm.tools.reranker import TileReranker

        TileReranker.clear_cache()

        # Determina la classe OOM da usare
        try:
            import torch
            OomClass = torch.cuda.OutOfMemoryError
        except (ImportError, AttributeError):
            OomClass = RuntimeError

        call_count = {"n": 0}

        def predict_with_oom(pairs, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OomClass("Simulated OOM")
            return np.array([1.0] * len(pairs))

        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(side_effect=predict_with_oom)

        with patch("tilellm.tools.reranker.CrossEncoder", return_value=mock_ce):
            with patch.object(TileReranker, "_use_fp16", False):
                with patch.object(TileReranker, "_warmup_on_load", False):
                    reranker = TileReranker(
                        "mock-model",
                        token_budget_per_batch=2048,
                        max_pairs_per_batch=16,
                        max_oom_retries=3,
                    )
        reranker.model = mock_ce

        # Costruisce un mock_torch che espone OutOfMemoryError = OomClass e
        # cuda.empty_cache() come no-op. In questo modo il codice di recovery
        # può fare isinstance(e, torch.cuda.OutOfMemoryError) e torch.cuda.empty_cache().
        mock_torch = MagicMock()
        mock_torch.cuda.OutOfMemoryError = OomClass
        mock_torch.cuda.empty_cache = MagicMock()
        # _device viene letto come class attr; forziamo 'cuda' per coprire il branch
        original_device = TileReranker._device

        pairs = [("query", f"doc {i}" * 5) for i in range(4)]
        pair_lengths = [estimate_pair_tokens("query", f"doc {i}" * 5) for i in range(4)]

        with patch.object(TileReranker, "_device", "cuda"):
            with patch.object(rm, "torch", mock_torch):
                scores = reranker._predict_local_with_oom_recovery(
                    pairs=pairs, pair_lengths=pair_lengths,
                    token_budget=2048, max_pairs=16
                )

        assert len(scores) == 4
        assert call_count["n"] >= 2, "Deve avere chiamato predict almeno 2 volte (OOM + retry)"
        mock_torch.cuda.empty_cache.assert_called()  # empty_cache invocato nel recovery

    def test_gpu_lock_serializes_concurrent_calls(self):
        """
        Con _gpu_lock, N thread concorrenti devono serializzare i forward pass.
        Verifica che non ci siano esecuzioni sovrapposte.
        """
        from tilellm.tools.reranker import TileReranker

        TileReranker.clear_cache()

        concurrent_count = {"n": 0, "max": 0}
        lock = threading.Lock()

        def slow_predict(pairs, **kwargs):
            with lock:
                concurrent_count["n"] += 1
                concurrent_count["max"] = max(concurrent_count["max"], concurrent_count["n"])
            time.sleep(0.05)  # simula forward pass
            with lock:
                concurrent_count["n"] -= 1
            return np.array([1.0] * len(pairs))

        mock_ce = MagicMock()
        mock_ce.predict = MagicMock(side_effect=slow_predict)

        with patch("tilellm.tools.reranker.CrossEncoder", return_value=mock_ce):
            with patch.object(TileReranker, "_use_fp16", False):
                with patch.object(TileReranker, "_warmup_on_load", False):
                    reranker = TileReranker("mock-model")
        reranker.model = mock_ce

        pairs = [("q", f"doc {i}") for i in range(5)]
        pair_lengths = [estimate_pair_tokens("q", f"doc {i}") for i in range(5)]

        results = []
        errors = []

        def run_predict():
            try:
                s = reranker._predict_local_with_oom_recovery(
                    pairs=pairs, pair_lengths=pair_lengths,
                    token_budget=2048, max_pairs=5
                )
                results.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_predict) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4
        # Il _gpu_lock garantisce che non ci siano più di 1 forward pass concorrente
        assert concurrent_count["max"] <= 1, (
            f"Forward pass concorrenti rilevati: max={concurrent_count['max']}"
        )
