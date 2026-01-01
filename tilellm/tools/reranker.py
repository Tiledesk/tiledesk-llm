from typing import List, Any, Union, Optional, TYPE_CHECKING
from collections import OrderedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import ConfigDict
import logging
from threading import Lock

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

class TEIReranker:
    def __init__(self, config: "TEIConfig"):
        self.config = config
        self.url = config.url.rstrip("/") if config.url else ""
        self.headers = config.custom_headers if config.custom_headers else {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self.logger = logging.getLogger(__name__)

    def rerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        import httpx
        import time
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        
        # Process documents in batches to avoid payload too large errors
        all_scored_docs = []
        i = 0
        current_batch_size = batch_size
        
        while i < len(texts):
            batch_texts = texts[i:i + current_batch_size]
            batch_docs = documents[i:i + current_batch_size]
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
                        batch_docs = documents[i:i + current_batch_size]
                        batch_retry_count += 1
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data:
                        idx = item["index"]
                        score = item["score"]
                        # Map batch index to original document
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
                        batch_docs = documents[i:i + current_batch_size]
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
        
        # Sort all scored documents by score (descending)
        all_scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in all_scored_docs[:top_k]]

    async def arerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        """
        Asynchronous version of rerank_documents using httpx.AsyncClient.
        Avoids blocking the event loop with HTTP I/O operations.
        """
        import httpx
        import asyncio

        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        # Process documents in batches
        all_scored_docs = []
        i = 0
        current_batch_size = batch_size

        async with httpx.AsyncClient() as client:
            while i < len(texts):
                batch_texts = texts[i:i + current_batch_size]
                batch_docs = documents[i:i + current_batch_size]
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
                            batch_docs = documents[i:i + current_batch_size]
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
                            batch_docs = documents[i:i + current_batch_size]
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

        # Sort all scored documents by score (descending)
        all_scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in all_scored_docs[:top_k]]


class TileReranker:
    # Cache LRU con dimensione massima di 2 modelli

    _device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    _model_cache = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)
    _cache_lock = Lock()

    def __init__(self, model_name: Union[str, "TEIConfig"] = "BAAI/bge-reranker-v2-m3"):
        if hasattr(model_name, "provider") and model_name.provider == "tei":
             self.model_name = "tei_" + model_name.url
             self.config = model_name
        else:
            self.model_name = model_name
            self.config = None
            
        self.model = self._get_cached_model(self.model_name, self.config)

    @classmethod
    def _get_cached_model(cls, model_name: str, config: Optional["TEIConfig"] = None):

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
                    cls._model_cache[model_name] = TEIReranker(config)
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

        # Prepara le coppie query-documento
        query_doc_pairs = [(query, doc.page_content) for doc in documents]

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

        # Ordina i documenti
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Liberazione esplicita della cache GPU se necessario
        if self._device == 'cuda':
            torch.cuda.empty_cache()
            self._logger.info("Freed GPU memory after tensor conversion")

        return [doc for doc, _ in scored_docs[:top_k]]

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

