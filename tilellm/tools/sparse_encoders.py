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


def _chunk_text(text: str) -> List[str]:
    if not text:
        return [""]
    
    # Generic word splitting (conservative approach)
    # Assuming avg 1.3 tokens per word, 300 words ~ 390 tokens < 512
    words = text.split()
    chunk_size = 300 
    
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

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


class TiledeskSpladeEncoder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Init of SpladeEncoder on device: {self.device}")
        self.splade = SpladeEncoder(device=self.device)
        self.logger.info("SpladeEncoder loaded")

    def encode_documents_with_batch(self, contents: List[str], batch_size: Optional[int] = 10) -> List[Dict[str, List]]:
        if not batch_size:
            self.logger.info(f"Encoding {len(contents)} documents with SpladeEncoder")
            return self.splade.encode_documents(contents)

        results = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            self.logger.info(
                f"Encoding batch [{i // batch_size + 1}/{(len(contents) - 1) // batch_size + 1}] with {len(batch)} documents")
            batch_result = self.splade.encode_documents(batch)
            results.extend(batch_result)

            # Libera memoria GPU dopo ogni batch
            if self.device == 'cuda':
                del batch_result
                torch.cuda.empty_cache()
                self.logger.debug("Freed GPU memory after batch processing")

        return results

    def encode_documents(self, contents: List[str], batch_size: Optional[int] = 10) -> List[Dict[str, List]]:
        if not contents:
            return []
            
        all_chunks = []
        doc_map = [] # (start_index, length)
        
        for text in contents:
            chunks = _chunk_text(text)
            doc_map.append((len(all_chunks), len(chunks)))
            all_chunks.extend(chunks)
            
        # Use old method to encode chunks (handles batching on GPU)
        chunk_results = self.encode_documents_with_batch(all_chunks, batch_size=batch_size)
        
        results = []
        for start, length in doc_map:
            doc_vectors = chunk_results[start : start + length]
            results.append(_merge_sparse_vectors(doc_vectors))
            
        return results

    def encode_queries(self, query: str) -> Dict[str, List]:
        self.logger.debug(f"Encoding query: '{query}'")
        return self.splade.encode_queries(query)


class TiledeskBGEM3:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_fp16_bool = True if self.device == 'cuda' else False
        self.logger.info(
            f"Init of BGEM3FlagModel ('BAAI/bge-m3') on device: {self.device}, use_fp16: {self.use_fp16_bool}")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=self.use_fp16_bool)
        self.logger.info("BGEM3FlagModel loaded.")

    def encode_documents(self, contents: List[str], batch_size: Optional[int] = 10) -> List[Dict[str, List]]:
        if not batch_size:
            self.logger.debug(f"Encoding {len(contents)} documents with BGEM3FlagModel")
            output = self.model.encode(contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            return self._convert_sparse_vectors(output['lexical_weights'])

        results = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            self.logger.info(
                f"Encoding batch [{i // batch_size + 1}/{(len(contents) - 1) // batch_size + 1}] with {len(batch)} documents")
            output = self.model.encode(batch, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            batch_result = self._convert_sparse_vectors(output['lexical_weights'])
            results.extend(batch_result)

            # Libera memoria GPU dopo ogni batch
            if self.device == 'cuda':
                del output, batch_result
                torch.cuda.empty_cache()
                self.logger.debug("Freed GPU memory after batch processing")

        return results

    def _convert_sparse_vectors(self, lexical_weights: List[Dict]) -> List[Dict[str, List]]:
        return [{
            'indices': [int(k) for k in doc_dict.keys()],
            'values': [float(doc_dict[k]) for k in doc_dict.keys()]
        } for doc_dict in lexical_weights]

    def encode_queries(self, query: str) -> Dict[str, List]:
        self.logger.debug(f"Encoding query: '{query}'")
        output = self.model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return self._convert_sparse_vectors(output['lexical_weights'])[0]


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


class TiledeskSparseEncoders:
    # LRU Cache con dimensione massima di 2 modelli
    _encoder_cache = OrderedDict()
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
    def _get_cached_encoder(cls, model_name: str, config: Optional["TEIConfig"] = None) -> Union[TiledeskSpladeEncoder, TiledeskBGEM3, TEISparseEncoder]:
        with cls._cache_lock:
            # Verifica se il modello è già in cache
            if model_name in cls._encoder_cache:
                cls._logger.info(f"Reusing cached instance of: {model_name}")
                # Sposta in fondo (più recente)
                encoder = cls._encoder_cache.pop(model_name)
                cls._encoder_cache[model_name] = encoder
                return encoder

            # Crea nuovo encoder se non in cache
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

            # Gestione LRU Cache
            if len(cls._encoder_cache) >= cls._max_cache_size:
                # Rimuovi il meno recente
                oldest = next(iter(cls._encoder_cache))
                cls._logger.info(f"Removing oldest model from cache: {oldest}")
                cls._encoder_cache.pop(oldest)

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
        cls._logger.warning("Clearing encoder cache and GPU memory")
        cls._encoder_cache.clear()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()