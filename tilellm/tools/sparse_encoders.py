import torch
from pinecone_text.sparse import SpladeEncoder
from FlagEmbedding import BGEM3FlagModel
from typing import Dict, Any, Optional, Union
import logging


class TiledeskSpladeEncoder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Init of SpladeEncoder on device: {self.device}")
        self.splade = SpladeEncoder(device=self.device)
        self.logger.info(f"SpladeEncoder loaded")

    def encode_documents(self, contents):
        # Logica di encoding specifica per SpladeEncoder
        self.logger.debug(f"Encoding {len(contents)} documents with SpladeEncoder")
        doc_sparse_vectors = self.splade.encode_documents(contents)
        return doc_sparse_vectors

    def encode_queries(self,query):
        self.logger.debug(f"Encoding query {query} documents with SpladeEncoder")
        return  self.splade.encode_queries(query)


class TiledeskBGEM3:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_fp16_bool = True if self.device == 'cuda' else False
        self.logger.info(f"Init of BGEM3FlagModel ('BAAI/bge-m3') su device: "
                         f"{self.device}, use_fp16: {self.use_fp16_bool}")
        self.model = BGEM3FlagModel('BAAI/bge-m3',
                               use_fp16=self.use_fp16_bool
                               )
        self.logger.info("BGEM3FlagModel loaded.")

    def encode_documents(self, contents):
        self.logger.debug(f"Encoding {len(contents)} documents with BGEM3FlagModel")

        output_1 = self.model.encode(contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dd = output_1['lexical_weights']
        doc_sparse_vectors = [
            {
                'indices': [int(k) for k in dd.keys()],
                'values': [float(dd[k]) for k in dd.keys()]
            }
            for dd in dd
        ]
        return doc_sparse_vectors

    def encode_queries(self, query):
        self.logger.debug(f"Encoding query '{query}' with BGEM3FlagModel.")
        query_encode = self.model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        dd = query_encode['lexical_weights']
        doc_sparse_vectors = [
            {
                'indices': [int(k) for k in dd.keys()],
                'values': [float(dd[k]) for k in dd.keys()]
            }
            for dd in dd
        ]
        return doc_sparse_vectors[0]

class TiledeskSparseEncoders:
    # --- Global Cache encoders instances ---
    _encoder_cache: Dict[str, Union[TiledeskSpladeEncoder, TiledeskBGEM3]] = {}
    _logger = logging.getLogger(__name__)

    def __init__(self, model_name):
        #self.encoder = self._get_encoder(model_name)
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = self._get_cached_encoder(model_name)
        # Il device è gestito all'interno delle classi TiledeskSpladeEncoder e TiledeskBGEM3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def _get_cached_encoder(cls, model_name: str) -> Union[TiledeskSpladeEncoder, TiledeskBGEM3]:
        """
        Ottiene un'istanza dell'encoder dalla cache o la crea se non esiste.
        Questo metodo è un @classmethod perché agisce sulla cache della classe.
        """
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._encoder_cache:
            cls._logger.info(f"Caricamento del modello sparse encoder '{model_name_lower}' per la prima volta...")
            if model_name_lower == "splade":
                encoder_instance = TiledeskSpladeEncoder()
            elif model_name_lower == "bge-m3":
                encoder_instance = TiledeskBGEM3()
            else:
                raise ValueError(
                    f"Model_name non supportato: '{model_name}'. I valori supportati sono 'splade' e 'bge-m3'.")

            cls._encoder_cache[model_name_lower] = encoder_instance
            cls._logger.info(f"Modello '{model_name_lower}' caricato e aggiunto alla cache.")
        else:
            cls._logger.info(f"Modello '{model_name_lower}' già in cache. Riutilizzo l'istanza esistente.")

        return cls._encoder_cache[model_name_lower]

    #def _get_encoder(self, model_name):
    #    if model_name == "splade":
    #        return TiledeskSpladeEncoder()
    #    elif model_name == "bge-m3":
    #        return TiledeskBGEM3()
    #    else:
    #        raise ValueError("Unsupported model_name: {}. Supported values are 'splade' and 'bge-m3'.".format(model_name))

    def encode_documents(self, contents):
        if self.encoder:
            return self.encoder.encode_documents(contents)
        else:
            raise ValueError("No encoder has been initialized.")

    def encode_queries(self, query):
        if self.encoder:
            return self.encoder.encode_queries(query)
        else:
            raise ValueError("No encoder has been initialized.")

    @classmethod
    def clear_cache(cls):
        """
        Svuota la cache dei modelli degli encoder.
        Da usare con estrema cautela in produzione.
        """
        cls._logger.warning("Svuotamento della cache dei modelli sparse encoder in corso...")
        cls._encoder_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cls._logger.warning("Cache GPU svuotata.")
        cls._logger.warning("Cache modelli sparse encoder svuotata.")