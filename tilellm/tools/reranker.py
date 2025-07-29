from abc import ABC
from typing import List, Any
from collections import OrderedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder
import logging
import torch
from functools import lru_cache
from threading import Lock

class TileReranker:
    # Cache LRU con dimensione massima di 2 modelli

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model_cache = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)
    _cache_lock = Lock()

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._get_cached_model(model_name)

    @classmethod
    def _get_cached_model(cls, model_name: str):

        with cls._cache_lock:
            """Ottieni un modello dalla cache LRU"""
            if model_name not in cls._model_cache:
                if len(cls._model_cache) >= cls._max_cache_size:
                    # Rimuovi il modello meno recentemente usato
                    oldest = next(iter(cls._model_cache))
                    cls._logger.info(f"Removing old reranker from cache: {oldest}")
                    cls._model_cache.pop(oldest)

                cls._logger.info(f"Loading new Reranker model: {model_name}")
                cls._model_cache[model_name] = CrossEncoder(model_name, device=cls._device)
            else:
                cls._logger.info(f"Using cached Reranker model: {model_name}")

            # Sposta questo modello in cima alla cache (più recente)
            model = cls._model_cache.pop(model_name)
            cls._model_cache[model_name] = model
        return model



    def rerank_documents_old(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        Re-rank documents using re-ranker model
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        # Prepare query-document pairs for scoring
        query_doc_pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(query_doc_pairs)


        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if self._device == 'cuda':
            del scores
            torch.cuda.empty_cache()
            # debug
            self._logger.info("Freed GPU memory after batch processing")

        # Return top_k documents

        return [doc for doc, score in scored_docs[:top_k]]

    def rerank_documents(self, query: str, documents: List[Document], top_k: int) -> List[Document]:

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
            return self.reranker.rerank_documents(query, documents, self.top_k)

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
            return self.reranker.rerank_documents(query, documents, self.top_k)

        return documents[:self.top_k] if len(documents) > self.top_k else documents

