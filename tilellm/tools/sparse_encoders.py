import torch
from pinecone_text.sparse import SpladeEncoder
from FlagEmbedding import BGEM3FlagModel
from typing import Dict, Any, Optional, Union, List
import logging
import functools
from collections import OrderedDict


class TiledeskSpladeEncoder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Init of SpladeEncoder on device: {self.device}")
        self.splade = SpladeEncoder(device=self.device)
        self.logger.info(f"SpladeEncoder loaded")

    def encode_documents(self, contents: List[str], batch_size: Optional[int] = 10) -> List[Dict[str, List]]:
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


class TiledeskSparseEncoders:
    # LRU Cache con dimensione massima di 2 modelli
    _encoder_cache = OrderedDict()
    _max_cache_size = 2
    _logger = logging.getLogger(__name__)

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.encoder = self._get_cached_encoder(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def _get_cached_encoder(cls, model_name: str) -> Union[TiledeskSpladeEncoder, TiledeskBGEM3]:
        # Verifica se il modello è già in cache
        if model_name in cls._encoder_cache:
            cls._logger.info(f"Reusing cached instance of: {model_name}")
            # Sposta in fondo (più recente)
            encoder = cls._encoder_cache.pop(model_name)
            cls._encoder_cache[model_name] = encoder
            return encoder

        # Crea nuovo encoder se non in cache
        if model_name == "splade":
            cls._logger.info(f"Creating new SpladeEncoder instance")
            encoder = TiledeskSpladeEncoder()
        elif model_name == "bge-m3":
            cls._logger.info(f"Creating new BGEM3 instance")
            encoder = TiledeskBGEM3()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'splade' or 'bge-m3'.")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()