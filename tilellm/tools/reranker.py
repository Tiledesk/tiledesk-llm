from typing import List, Any, Union, Optional, TYPE_CHECKING, Tuple
from collections import OrderedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import ConfigDict
import logging
from threading import Lock
import re

if TYPE_CHECKING:
    from tilellm.models.llm import TEIConfig

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Max-P Strategy Helper Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Fast token estimation using simple word splitting.
    Approssimazione: ~1.8 token per word (conservativa per sicurezza)

    Note: Usiamo 1.8 invece di 1.3 per essere più conservativi e evitare
    che chunk stimati come <350 token superino effettivamente il limite di 512.
    """
    # Split by whitespace and common punctuation
    words = re.findall(r'\w+|[^\w\s]', text)
    return int(len(words) * 1.8)


def chunk_text_with_overlap(text: str, max_tokens: int = 300, overlap_tokens: int = 40) -> List[str]:
    """
    Divide il testo in chunk con overlap usando strategia Max-P.

    Args:
        text: Testo da dividere
        max_tokens: Dimensione massima del chunk in token (default: 300 per sicurezza)
        overlap_tokens: Numero di token di overlap tra chunk consecutivi (default: 40)

    Returns:
        Lista di chunk di testo
    """
    # Split in words per gestire l'overlap
    words = re.findall(r'\S+', text)  # Include punctuation attached to words

    if not words:
        return [text]

    # Stima token per word (1.8 token/word, conservativa)
    tokens_per_word = 1.8
    words_per_chunk = int(max_tokens / tokens_per_word)
    overlap_words = int(overlap_tokens / tokens_per_word)

    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))

        # Se siamo alla fine, esci
        if end >= len(words):
            break

        # Muovi lo start considerando l'overlap
        start = end - overlap_words

        # Evita loop infiniti se overlap è troppo grande
        if start <= (end - words_per_chunk):
            start = end

    return chunks


def apply_maxp_chunking(documents: List[Document], max_tokens: int = 300, overlap_tokens: int = 40) -> Tuple[List[Document], List[int]]:
    """
    Applica la strategia Max-P ai documenti, dividendo quelli troppo lunghi in chunk.

    Args:
        documents: Lista di documenti originali
        max_tokens: Limite massimo di token per chunk
        overlap_tokens: Token di overlap tra chunk

    Returns:
        Tupla (chunked_documents, doc_indices) dove:
        - chunked_documents: Lista di documenti (alcuni potrebbero essere chunk)
        - doc_indices: Mapping da indice chunk a indice documento originale
    """
    chunked_docs = []
    doc_indices = []  # Map chunk index -> original doc index

    for doc_idx, doc in enumerate(documents):
        text = doc.page_content
        estimated_tokens = estimate_tokens(text)

        # Se il documento è sotto il limite, usalo così com'è
        if estimated_tokens <= max_tokens:
            chunked_docs.append(doc)
            doc_indices.append(doc_idx)
        else:
            # Dividi in chunk con overlap
            chunks = chunk_text_with_overlap(text, max_tokens, overlap_tokens)

            for chunk_text in chunks:
                # Crea un nuovo documento per ogni chunk, mantenendo i metadata originali
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "_is_chunk": True, "_original_doc_idx": doc_idx}
                )
                chunked_docs.append(chunk_doc)
                doc_indices.append(doc_idx)

    return chunked_docs, doc_indices


def aggregate_maxp_scores(chunked_scores: List[Tuple[Document, float]], doc_indices: List[int]) -> List[Tuple[Document, float]]:
    """
    Aggrega i punteggi dei chunk usando strategia Max-P (punteggio massimo per documento).

    Args:
        chunked_scores: Lista di (documento_chunk, score)
        doc_indices: Mapping da indice chunk a indice documento originale

    Returns:
        Lista di (documento_originale, max_score) aggregata
    """
    # Raggruppa punteggi per documento originale
    doc_max_scores = {}  # original_doc_idx -> (doc, max_score)

    for (chunk_doc, score), orig_idx in zip(chunked_scores, doc_indices):
        if orig_idx not in doc_max_scores:
            # Prima volta che vediamo questo documento: crea entry
            # Se è un chunk, ricostruisci il documento originale dai metadata
            if chunk_doc.metadata.get("_is_chunk"):
                # Trova il documento originale (potrebbe essere in chunked_scores)
                # Per ora usa il chunk doc ma marca che è stato processato
                original_doc = Document(
                    page_content=chunk_doc.page_content,  # Useremo il primo chunk come rappresentante
                    metadata={k: v for k, v in chunk_doc.metadata.items() if not k.startswith("_")}
                )
                doc_max_scores[orig_idx] = (original_doc, score)
            else:
                doc_max_scores[orig_idx] = (chunk_doc, score)
        else:
            # Aggiorna il punteggio se questo chunk ha score maggiore
            existing_doc, existing_score = doc_max_scores[orig_idx]
            if score > existing_score:
                doc_max_scores[orig_idx] = (existing_doc, score)

    # Converti in lista ordinata per indice originale
    result = [doc_max_scores[idx] for idx in sorted(doc_max_scores.keys())]
    return result

class TEIReranker:
    def __init__(self, config: "TEIConfig", max_tokens: int = 300, overlap_tokens: int = 40):
        """
        Initialize TEI Reranker with Max-P strategy support.

        Args:
            config: TEI configuration
            max_tokens: Maximum tokens per chunk for Max-P strategy (default: 300, conservativo)
            overlap_tokens: Overlap tokens between chunks (default: 40)
        """
        self.config = config
        self.url = config.url.rstrip("/") if config.url else ""
        self.headers = config.custom_headers if config.custom_headers else {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self.logger = logging.getLogger(__name__)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def rerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        import httpx
        import time
        if not documents:
            return []

        # ===== STEP 1: Apply Max-P Strategy =====
        # Divide long documents into chunks to avoid token limit errors
        chunked_docs, doc_indices = apply_maxp_chunking(
            documents,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens
        )

        num_chunks = len(chunked_docs)
        num_originals = len(documents)
        if num_chunks > num_originals:
            self.logger.info(f"Max-P Strategy: Split {num_originals} documents into {num_chunks} chunks "
                           f"(max_tokens={self.max_tokens}, overlap={self.overlap_tokens})")

        texts = [doc.page_content for doc in chunked_docs]

        # ===== STEP 2: Process chunks in batches =====
        all_scored_docs = []
        i = 0
        current_batch_size = batch_size

        while i < len(texts):
            batch_texts = texts[i:i + current_batch_size]
            batch_docs = chunked_docs[i:i + current_batch_size]
            self.logger.info(f"Processing batch {i//current_batch_size + 1}/{(len(texts)-1)//current_batch_size + 1} with {len(batch_texts)} documents (batch size: {current_batch_size})")
            batch_retry_count = 0
            max_retries = 3
            batch_success = False

            while batch_retry_count < max_retries and not batch_success:
                try:
                    payload = {
                        "query": query,
                        "texts": batch_texts,
                    }
                    response = httpx.post(f"{self.url}/rerank", json=payload, headers=self.headers, timeout=60)

                    if response.status_code == 413 and current_batch_size > 1:
                        # Payload too large, reduce batch size and retry
                        new_batch_size = max(1, current_batch_size // 2)
                        self.logger.warning(f"Payload too large (413) for batch size {current_batch_size}. Reducing to {new_batch_size}")
                        current_batch_size = new_batch_size
                        batch_texts = texts[i:i + current_batch_size]
                        batch_docs = chunked_docs[i:i + current_batch_size]
                        batch_retry_count += 1
                        continue

                    response.raise_for_status()
                    data = response.json()

                    for item in data:
                        idx = item["index"]
                        score = item["score"]
                        # Map batch index to chunk document
                        all_scored_docs.append((batch_docs[idx], score))

                    batch_success = True
                    i += current_batch_size

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 413 and current_batch_size > 1:
                        # Payload too large, reduce batch size and retry
                        new_batch_size = max(1, current_batch_size // 2)
                        self.logger.warning(f"Payload too large (413) for batch size {current_batch_size}. Reducing to {new_batch_size}")
                        current_batch_size = new_batch_size
                        batch_texts = texts[i:i + current_batch_size]
                        batch_docs = chunked_docs[i:i + current_batch_size]
                        batch_retry_count += 1
                        time.sleep(0.5 * batch_retry_count)  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"HTTP error calling TEI rerank: {e}")
                        raise e
                except Exception as e:
                    self.logger.error(f"Error calling TEI rerank: {e}")
                    raise e

            if not batch_success:
                raise RuntimeError(f"Failed to process batch after {max_retries} retries")

        # ===== STEP 3: Aggregate scores using Max-P =====
        # Map chunks back to original documents and take max score
        aggregated_docs = aggregate_maxp_scores(all_scored_docs, doc_indices)

        # Sort by score (descending)
        aggregated_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in aggregated_docs[:top_k]]

    async def arerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        """
        Asynchronous version of rerank_documents using httpx.AsyncClient.
        Avoids blocking the event loop with HTTP I/O operations.
        Includes Max-P strategy for handling long documents.
        """
        import httpx
        import asyncio

        if not documents:
            return []

        # ===== STEP 1: Apply Max-P Strategy =====
        # Divide long documents into chunks to avoid token limit errors
        chunked_docs, doc_indices = apply_maxp_chunking(
            documents,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens
        )

        num_chunks = len(chunked_docs)
        num_originals = len(documents)
        if num_chunks > num_originals:
            self.logger.info(f"Max-P Strategy: Split {num_originals} documents into {num_chunks} chunks "
                           f"(max_tokens={self.max_tokens}, overlap={self.overlap_tokens})")

        texts = [doc.page_content for doc in chunked_docs]

        # ===== STEP 2: Process chunks in batches =====
        all_scored_docs = []
        i = 0
        current_batch_size = batch_size

        async with httpx.AsyncClient() as client:
            while i < len(texts):
                batch_texts = texts[i:i + current_batch_size]
                batch_docs = chunked_docs[i:i + current_batch_size]
                self.logger.info(f"Processing batch {i//current_batch_size + 1}/{(len(texts)-1)//current_batch_size + 1} with {len(batch_texts)} documents (batch size: {current_batch_size})")
                batch_retry_count = 0
                max_retries = 3
                batch_success = False

                while batch_retry_count < max_retries and not batch_success:
                    try:
                        payload = {
                            "query": query,
                            "texts": batch_texts,
                        }
                        response = await client.post(
                            f"{self.url}/rerank",
                            json=payload,
                            headers=self.headers,
                            timeout=60
                        )

                        if response.status_code == 413 and current_batch_size > 1:
                            # Payload too large, reduce batch size and retry
                            new_batch_size = max(1, current_batch_size // 2)
                            self.logger.warning(f"Payload too large (413) for batch size {current_batch_size}. Reducing to {new_batch_size}")
                            current_batch_size = new_batch_size
                            batch_texts = texts[i:i + current_batch_size]
                            batch_docs = chunked_docs[i:i + current_batch_size]
                            batch_retry_count += 1
                            continue

                        response.raise_for_status()
                        data = response.json()

                        for item in data:
                            idx = item["index"]
                            score = item["score"]
                            all_scored_docs.append((batch_docs[idx], score))

                        batch_success = True
                        i += current_batch_size

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 413 and current_batch_size > 1:
                            new_batch_size = max(1, current_batch_size // 2)
                            self.logger.warning(f"Payload too large (413) for batch size {current_batch_size}. Reducing to {new_batch_size}")
                            current_batch_size = new_batch_size
                            batch_texts = texts[i:i + current_batch_size]
                            batch_docs = chunked_docs[i:i + current_batch_size]
                            batch_retry_count += 1
                            await asyncio.sleep(0.5 * batch_retry_count)  # Exponential backoff
                            continue
                        else:
                            self.logger.error(f"HTTP error calling TEI rerank: {e}")
                            raise e
                    except Exception as e:
                        self.logger.error(f"Error calling TEI rerank: {e}")
                        raise e

                if not batch_success:
                    raise RuntimeError(f"Failed to process batch after {max_retries} retries")

        # ===== STEP 3: Aggregate scores using Max-P =====
        # Map chunks back to original documents and take max score
        aggregated_docs = aggregate_maxp_scores(all_scored_docs, doc_indices)

        # Sort by score (descending)
        aggregated_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in aggregated_docs[:top_k]]


class TileReranker:
    # Cache LRU con dimensione massima di 2 modelli

    _device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    _model_cache = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)
    _cache_lock = Lock()

    def __init__(self, model_name: Union[str, "TEIConfig"] = "BAAI/bge-reranker-v2-m3",
                 max_tokens: int = 300, overlap_tokens: int = 40):
        """
        Initialize TileReranker with Max-P strategy support.

        Args:
            model_name: Model name or TEIConfig instance
            max_tokens: Maximum tokens per chunk for Max-P strategy (default: 300, conservativo)
            overlap_tokens: Overlap tokens between chunks (default: 40)
        """
        if hasattr(model_name, "provider") and model_name.provider == "tei":
             self.model_name = "tei_" + model_name.url
             self.config = model_name
        else:
            self.model_name = model_name
            self.config = None

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model = self._get_cached_model(self.model_name, self.config, max_tokens, overlap_tokens)

    @classmethod
    def _get_cached_model(cls, model_name: str, config: Optional["TEIConfig"] = None,
                         max_tokens: int = 300, overlap_tokens: int = 40):

        with cls._cache_lock:
            """Ottieni un modello dalla cache LRU"""
            if model_name not in cls._model_cache:
                if len(cls._model_cache) >= cls._max_cache_size:
                    # Rimuovi il modello meno recentemente usato
                    oldest = next(iter(cls._model_cache))
                    cls._logger.info(f"Removing old reranker from cache: {oldest}")
                    cls._model_cache.pop(oldest)

                if config and hasattr(config, "provider") and config.provider == "tei":
                    cls._logger.info("Loading new TEIReranker instance")
                    cls._model_cache[model_name] = TEIReranker(config, max_tokens, overlap_tokens)
                else:
                    if CrossEncoder is None:
                        raise ImportError("CrossEncoder is not available. Install 'ml' extras.")
                    cls._logger.info(f"Loading new Reranker model: {model_name}")
                    cls._model_cache[model_name] = CrossEncoder(model_name, device=cls._device)
            else:
                cls._logger.info(f"Using cached Reranker model: {model_name}")

            # Sposta questo modello in cima alla cache (più recente)
            model = cls._model_cache.pop(model_name)
            cls._model_cache[model_name] = model
        return model


    def rerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        if isinstance(self.model, TEIReranker):
             return self.model.rerank_documents(query, documents, top_k, batch_size)

        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        # ===== STEP 1: Apply Max-P Strategy =====
        # Divide long documents into chunks to avoid token limit errors
        chunked_docs, doc_indices = apply_maxp_chunking(
            documents,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens
        )

        num_chunks = len(chunked_docs)
        num_originals = len(documents)
        if num_chunks > num_originals:
            self._logger.info(f"Max-P Strategy (CrossEncoder): Split {num_originals} documents into {num_chunks} chunks "
                            f"(max_tokens={self.max_tokens}, overlap={self.overlap_tokens})")

        # ===== STEP 2: Rerank chunks =====
        # Prepara le coppie query-chunk
        query_doc_pairs = [(query, doc.page_content) for doc in chunked_docs]

        # Calcola i punteggi con context no_grad
        # Disabilita il calcolo dei gradienti durante l'inferenza, riducendo il consumo di memoria.
        with torch.no_grad():
            scores_tensor = self.model.predict(query_doc_pairs)

        # Converti immediatamente in lista Python e rilascia il tensore
        if isinstance(scores_tensor, torch.Tensor):
            scores = scores_tensor.cpu().numpy().tolist()  # Sposta su CPU e converte
            del scores_tensor  # Rilascia esplicitamente il riferimento al tensore
        else:
            scores = scores_tensor  # Se non è un tensore (es. CPU)

        # ===== STEP 3: Aggregate scores using Max-P =====
        scored_chunks = list(zip(chunked_docs, scores))
        aggregated_docs = aggregate_maxp_scores(scored_chunks, doc_indices)

        # Sort by score (descending)
        aggregated_docs.sort(key=lambda x: x[1], reverse=True)

        # Liberazione esplicita della cache GPU se necessario
        if self._device == 'cuda':
            torch.cuda.empty_cache()
            self._logger.info("Freed GPU memory after tensor conversion")

        return [doc for doc, _ in aggregated_docs[:top_k]]

    async def arerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        """
        Asynchronous version of rerank_documents.
        - For TEIReranker: Uses native async HTTP client
        - For CrossEncoder: Runs in thread pool to avoid blocking

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            batch_size: Batch size for processing (used by TEI reranker)

        Returns:
            List of reranked documents (top_k)
        """
        import asyncio

        # If using TEI, delegate to its native async method
        if isinstance(self.model, TEIReranker):
            return await self.model.arerank_documents(
                query=query,
                documents=documents,
                top_k=top_k,
                batch_size=batch_size
            )

        # For CrossEncoder (CPU/GPU), run in thread pool
        return await asyncio.to_thread(
            self.rerank_documents,
            query=query,
            documents=documents,
            top_k=top_k,
            batch_size=batch_size
        )

    def clear_cache(cls):
        """Clear the model cache and free up memory"""
        cls._logger.info("Clearing BGE Reranker cache")
        cls._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RerankedRetriever(BaseRetriever):
    base_retriever: Any
    reranker: TileReranker
    top_k: int
    use_reranking: bool = True
    contextualize_query : str = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        """Recupera documenti sincroni con possibile reranking"""
        # Recupera i documenti dal retriever base

        if self.contextualize_query:
            query=self.contextualize_query

        documents = self.base_retriever.invoke(
            query,config={"callbacks": run_manager.get_child()}, **kwargs
        )

        # Applica re-ranking se necessario
        if self.use_reranking and len(documents) > self.top_k:
            return self.reranker.rerank_documents(query, documents, self.top_k, batch_size=8)

        return documents[:self.top_k] if len(documents) > self.top_k else documents



    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        """Recupera documenti asincroni con possibile reranking"""
        # Recupera i documenti dal retriever base
        if self.contextualize_query:
            query=self.contextualize_query

        documents = await self.base_retriever.ainvoke(
            query,
            config={"callbacks": run_manager.get_child()}, **kwargs
        )

        # Applica re-ranking se necessario
        if self.use_reranking and len(documents) > self.top_k:
            return self.reranker.rerank_documents(query, documents, self.top_k, batch_size=8)

        return documents[:self.top_k] if len(documents) > self.top_k else documents

