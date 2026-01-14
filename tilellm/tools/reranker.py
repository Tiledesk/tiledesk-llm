from typing import List, Any, Union, Optional, TYPE_CHECKING, Tuple, Dict
from collections import OrderedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import ConfigDict
import logging
from threading import Lock
import re

if TYPE_CHECKING:
    from tilellm.models.llm import TEIConfig, PineconeRerankerConfig

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    import torch
except ImportError:
    torch = None

try:
    from pinecone import Pinecone, PineconeAsyncio
except ImportError:
    Pinecone = None


# ============================================================================
# Pinecone Model Specifications
# ============================================================================

PINEONE_MODEL_SPECS = {
    "cohere-rerank-3.5": {
        "max_tokens_per_pair": 40000,  # 40k token limit
        "max_documents": 200,
        "supports_truncate": False,
        "supports_max_chunks_per_doc": True,
        "max_rank_fields": None,  # Multiple fields supported
        "default_parameters": {},
        "requires_maxp": False,  # 40k tokens is huge, Max-P likely not needed
        "recommended_batch_size": 50,  # Can handle larger batches
    },
    "bge-reranker-v2-m3": {
        "max_tokens_per_pair": 1024,
        "max_documents": 100,
        "supports_truncate": True,
        "supports_max_chunks_per_doc": False,
        "max_rank_fields": 1,  # Only single field
        "default_parameters": {"truncate": "END"},
        "requires_maxp": True,  # 1024 token limit may require chunking
        "recommended_batch_size": 20,
    },
    "pinecone-rerank-v0": {
        "max_tokens_per_pair": 512,
        "max_documents": 100,
        "supports_truncate": True,
        "supports_max_chunks_per_doc": False,
        "max_rank_fields": 1,  # Only single field
        "default_parameters": {"truncate": "END"},
        "requires_maxp": True,  # 512 token limit definitely requires chunking
        "recommended_batch_size": 15,
    }
}

def get_pinecone_model_spec(model_name: str) -> dict:
    """Get specification for a Pinecone reranker model."""
    # Try exact match first
    if model_name in PINEONE_MODEL_SPECS:
        return PINEONE_MODEL_SPECS[model_name]
    
    # Try partial match (e.g., with version suffix)
    for key in PINEONE_MODEL_SPECS:
        if model_name.startswith(key):
            return PINEONE_MODEL_SPECS[key]
    
    # Default to bge-reranker-v2-m3 specs (most common)
    return PINEONE_MODEL_SPECS["bge-reranker-v2-m3"]


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


class PineconeReranker:
    def __init__(self, config: "PineconeRerankerConfig", max_tokens: int = 300, overlap_tokens: int = 40):
        """
        Initialize Pinecone Reranker with adaptive strategy based on model specifications.

        Args:
            config: Pinecone reranker configuration
            max_tokens: Maximum tokens per chunk for Max-P strategy (default: 300, conservativo)
            overlap_tokens: Overlap tokens between chunks (default: 40)
        """
        if Pinecone is None:
            raise ImportError("Pinecone is not available. Install 'pinecone-client'.")
        self.config = config
        self.api_key = config.api_key.get_secret_value()
        self.name = config.name
        self.top_n = config.top_n
        self.rank_fields = config.rank_fields
        self.parameters = config.parameters
        self.logger = logging.getLogger(__name__)
        self.client = Pinecone(api_key=self.api_key)
        
        # Get model specifications
        self.model_spec = get_pinecone_model_spec(self.name)
        
        # Adjust Max-P strategy based on model capabilities
        if self.model_spec["requires_maxp"]:
            # Models with lower token limits need Max-P
            self.max_tokens = max_tokens
            self.overlap_tokens = overlap_tokens
            self.use_maxp = True
        else:
            # Models with high token limits (cohere-rerank-3.5) don't need Max-P
            # Set max_tokens to model limit for safety
            self.max_tokens = min(max_tokens, self.model_spec["max_tokens_per_pair"] // 2)
            self.overlap_tokens = overlap_tokens
            self.use_maxp = False
            self.logger.info(f"Model {self.name} has high token limit ({self.model_spec['max_tokens_per_pair']}), Max-P strategy disabled")
        
        # Validate and adjust rank_fields based on model limitations
        if self.model_spec["max_rank_fields"] is not None:
            if self.rank_fields and len(self.rank_fields) > self.model_spec["max_rank_fields"]:
                self.logger.warning(f"Model {self.name} supports only {self.model_spec['max_rank_fields']} rank field(s), "
                                  f"using first {self.model_spec['max_rank_fields']} from {self.rank_fields}")
                self.rank_fields = self.rank_fields[:self.model_spec["max_rank_fields"]]
        
        # Filter parameters based on model support
        self.filtered_parameters = self._filter_parameters(self.parameters)
        
        # Set recommended batch size for logging
        self.recommended_batch_size = self.model_spec["recommended_batch_size"]

        #apc = PineconeAsyncio(api_key=self.api_key)
        #await apc.inference.rerank()

    def _filter_parameters(self, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter parameters based on model support."""
        if parameters is None:
            parameters = {}
        
        filtered = {}
        for key, value in parameters.items():
            # Check if parameter is supported by the model
            if key == "truncate" and not self.model_spec["supports_truncate"]:
                self.logger.warning(f"Parameter '{key}' not supported for model {self.name}, skipping")
                continue
            elif key == "max_chunks_per_doc" and not self.model_spec["supports_max_chunks_per_doc"]:
                self.logger.warning(f"Parameter '{key}' not supported for model {self.name}, skipping")
                continue
            
            filtered[key] = value
        
        # Apply default parameters if not specified
        for key, value in self.model_spec["default_parameters"].items():
            if key not in filtered:
                filtered[key] = value
        
        return filtered

    def rerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        if not documents:
            return []

        # ===== STEP 1: Apply Adaptive Strategy =====
        if self.use_maxp:
            # Apply Max-P chunking for models with low token limits
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
        else:
            # For models with high token limits (cohere-rerank-3.5), use original documents
            chunked_docs = documents
            doc_indices = list(range(len(documents)))  # Each doc maps to itself
        
        # ===== STEP 2: Prepare documents for Pinecone API =====
        pinecone_docs = []
        for idx, doc in enumerate(chunked_docs):
            # Use document metadata id if available, otherwise generate a unique id
            doc_id = doc.metadata.get("id", f"doc_{hash(doc.page_content) & 0xFFFFFFFF}")
            
            # Prepare document fields for reranking
            doc_dict = {"id": doc_id}
            
            # Add rank fields with appropriate content
            if self.rank_fields:
                for field in self.rank_fields:
                    if field == "chunk_text" or field == "text":
                        doc_dict[field] = doc.page_content
                    else:
                        # Try to get field from metadata
                        doc_dict[field] = doc.metadata.get(field, "")
            else:
                # Default field
                doc_dict["text"] = doc.page_content
            
            pinecone_docs.append(doc_dict)
        
        # Determine top_n: if not set, use top_k or number of documents
        top_n = self.top_n if self.top_n is not None else top_k
        
        # ===== STEP 3: Batch processing if documents exceed model limit =====
        max_docs_per_call = self.model_spec["max_documents"]
        all_scored_chunks = []
        
        if len(pinecone_docs) <= max_docs_per_call:
            # Single batch processing
            self.logger.info(f"Pinecone rerank with model {self.name}: "
                           f"documents={len(pinecone_docs)}, top_n={top_n}, "
                           f"rank_fields={self.rank_fields}, parameters={self.filtered_parameters}")
            
            try:
                ranked_results = self.client.inference.rerank(
                    model=self.name,
                    query=query,
                    documents=pinecone_docs,
                    top_n=min(top_n, len(pinecone_docs)),
                    rank_fields=self.rank_fields if self.rank_fields else ["text"],
                    return_documents=True,
                    parameters=self.filtered_parameters
                )
            except Exception as e:
                self.logger.error(f"Error calling Pinecone rerank: {e}")
                raise e
            
            # Process results
            for item in ranked_results.data:
                idx = item.index
                score = item.score
                all_scored_chunks.append((chunked_docs[idx], score))
        else:
            # Batch processing - split into multiple API calls
            num_batches = (len(pinecone_docs) + max_docs_per_call - 1) // max_docs_per_call
            self.logger.info(f"Pinecone rerank with model {self.name}: "
                           f"documents={len(pinecone_docs)} exceeds limit {max_docs_per_call}, "
                           f"splitting into {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start = batch_idx * max_docs_per_call
                end = min(start + max_docs_per_call, len(pinecone_docs))
                batch_docs = pinecone_docs[start:end]
                batch_chunked_docs = chunked_docs[start:end]
                
                self.logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_docs)} documents")
                
                try:
                    # For batch processing, we want all results from each batch
                    batch_top_n = min(top_n, len(batch_docs))
                    ranked_results = self.client.inference.rerank(
                        model=self.name,
                        query=query,
                        documents=batch_docs,
                        top_n=batch_top_n,
                        rank_fields=self.rank_fields if self.rank_fields else ["text"],
                        return_documents=True,
                        parameters=self.filtered_parameters
                    )
                except Exception as e:
                    self.logger.error(f"Error calling Pinecone rerank on batch {batch_idx + 1}: {e}")
                    raise e
                
                # Process batch results
                for item in ranked_results.data:
                    idx = item.index
                    score = item.score
                    # Adjust index to global chunked_docs
                    global_idx = start + idx
                    all_scored_chunks.append((chunked_docs[global_idx], score))
        
        # ===== STEP 4: Aggregate and rank results =====
        if self.use_maxp:
            # Aggregate scores using Max-P strategy
            aggregated_docs = aggregate_maxp_scores(all_scored_chunks, doc_indices)
            # Sort by score (descending)
            aggregated_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in aggregated_docs[:top_k]]
        else:
            # For non-Max-P, we already have original documents
            # Sort by score (descending) and take top_k
            all_scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in all_scored_chunks[:top_k]]

    async def arerank_documents(self, query: str, documents: List[Document], top_k: int, batch_size: int = 8) -> List[Document]:
        """
        Asynchronous version of rerank_documents.
        Pinecone SDK currently does not support async natively, so we run in thread pool.
        """

        import asyncio
        return await asyncio.to_thread(
            self.rerank_documents,
            query=query,
            documents=documents,
            top_k=top_k,
            batch_size=batch_size
        )


class TileReranker:
    # Cache LRU con dimensione massima di 2 modelli

    _device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    _model_cache = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)
    _cache_lock = Lock()

    def __init__(self, model_name: Union[str, "TEIConfig", "PineconeRerankerConfig"] = "BAAI/bge-reranker-v2-m3",
                 max_tokens: int = 300, overlap_tokens: int = 40):
        """
        Initialize TileReranker with Max-P strategy support.

        Args:
            model_name: Model name or TEIConfig instance
            max_tokens: Maximum tokens per chunk for Max-P strategy (default: 300, conservativo)
            overlap_tokens: Overlap tokens between chunks (default: 40)
        """
        if not isinstance(model_name, str) and hasattr(model_name, "provider"):
            if model_name.provider == "tei":
                self.model_name = f"tei_{model_name.name}"
                self.config = model_name
            elif model_name.provider == "pinecone":
                self.model_name = f"pinecone_{model_name.name}"
                self.config = model_name
            else:
                self.model_name = str(model_name)
                self.config = None
        else:
            self.model_name = model_name
            self.config = None

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model = self._get_cached_model(self.model_name, self.config, max_tokens, overlap_tokens)

    @classmethod
    def _get_cached_model(cls, model_name: str, config: Optional[Union["TEIConfig", "PineconeRerankerConfig"]] = None,
                         max_tokens: int = 300, overlap_tokens: int = 40):

        with cls._cache_lock:
            """Ottieni un modello dalla cache LRU"""
            if model_name not in cls._model_cache:
                if len(cls._model_cache) >= cls._max_cache_size:
                    # Rimuovi il modello meno recentemente usato
                    oldest = next(iter(cls._model_cache))
                    cls._logger.info(f"Removing old reranker from cache: {oldest}")
                    cls._model_cache.pop(oldest)

                if config and hasattr(config, "provider"):
                    if config.provider == "tei":
                        cls._logger.info("Loading new TEIReranker instance")
                        cls._model_cache[model_name] = TEIReranker(config, max_tokens, overlap_tokens)
                    elif config.provider == "pinecone":
                        if Pinecone is None:
                            raise ImportError("Pinecone is not available. Install 'pinecone-client'.")
                        cls._logger.info("Loading new PineconeReranker instance")
                        cls._model_cache[model_name] = PineconeReranker(config, max_tokens, overlap_tokens)
                    else:
                        raise ValueError(f"Unknown provider: {config.provider}")
                else:
                    if CrossEncoder is None:
                        raise ImportError("CrossEncoder is not available. Install 'sentence-transformers' package or 'ml' extras.")
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
        if isinstance(self.model, PineconeReranker):
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
        if torch is not None:
            with torch.no_grad():
                scores_tensor = self.model.predict(query_doc_pairs)
        else:
            scores_tensor = self.model.predict(query_doc_pairs)

        # Converti immediatamente in lista Python e rilascia il tensore
        if torch is not None and isinstance(scores_tensor, torch.Tensor):
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
        if torch is not None and self._device == 'cuda':
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
        # If using Pinecone, delegate to its async method
        if isinstance(self.model, PineconeReranker):
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

    @classmethod
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

