import time
from collections import OrderedDict
import logging

from typing import Any, Dict, Optional, Tuple, Callable
import asyncio

logger = logging.getLogger("GlobalCache")


class TimedCache:
    """
    Cache avanzata con scadenza basata sull'inattività.
    Gestisce automaticamente la rimozione degli oggetti dopo un periodo di inattività.
    Supporta diversi tipi di oggetti (Embeddings, Chat, Repository) con chiavi univoche.
    Gestisce la chiusura automatica delle connessioni Pinecone per i repository.
    """
    _caches: Dict[str, OrderedDict] = {}
    _timeout_seconds: float = 300  # 5 minuti
    _max_size: int = 100  # Limite massimo di oggetti per tipo
    # Policy per object_type
    _policies: Dict[str, Dict[str, Optional[int]]] = {
        # embeddings: niente TTL, nessuna chiusura su eviction (si chiudono solo su clear)
        "embedding": {
            "timeout_seconds": None,
            "max_size": 200,
            "refresh_on_access": False,
            "close_on_evict": False,
        },
        # vector store: TTL e chiusura su eviction
        "vector_store": {
            "timeout_seconds": 1800,
            "max_size": 200,
            "refresh_on_access": True,
            "close_on_evict": False,
        },
        # repository pinecone (se lo usi come object_type)
        "repository": {
            "timeout_seconds": 1800,
            "max_size": 100,
            "refresh_on_access": True,
            "close_on_evict": True,
        },
        "persistent_pinecone_client": {
            "timeout_seconds": 600,  # 2 ore
            "max_size": 50,
            "refresh_on_access": True,
            "close_on_evict": True,
        },
        "vector_store_wrapper": {
            "timeout_seconds" : 1800,
            "max_size" : 200,
            "refresh_on_access" : True,
            "close_on_evict" : True
        },
        # fallback per altri tipi
        "*": {
            "timeout_seconds": _timeout_seconds,
            "max_size": _max_size,
            "refresh_on_access": True,
            "close_on_evict": False,
        },
    }

    @classmethod
    def set_policy(cls, object_type: str, timeout_seconds: Optional[int], max_size: Optional[int] = None,
                   refresh_on_access: Optional[bool] = None, close_on_evict: Optional[bool] = None):
        pol = cls._policies.get(object_type, cls._policies["*"]).copy()
        if timeout_seconds is not None or timeout_seconds is None:
            pol["timeout_seconds"] = timeout_seconds
        if max_size is not None:
            pol["max_size"] = max_size
        if refresh_on_access is not None:
            pol["refresh_on_access"] = refresh_on_access
        if close_on_evict is not None:
            pol["close_on_evict"] = close_on_evict
        cls._policies[object_type] = pol

    @classmethod
    def _policy(cls, object_type: str) -> Dict[str, Any]:
        return cls._policies.get(object_type, cls._policies["*"])


    @staticmethod
    def _is_cached_vector_store_wrapper(obj: Any) -> bool:
        return obj.__class__.__name__ == "CachedVectorStore" or "CachedVectorStore" in obj.__class__.__name__

    @classmethod
    def _is_pinecone_repository(cls, obj: Any) -> bool:
        """Controlla se l'oggetto è un repository Pinecone che necessita chiusura"""
        class_name = obj.__class__.__name__
        return class_name in ['PineconeRepositoryPod', 'PineconeRepositoryServerless']

    @classmethod
    def _is_vector_store(cls, obj: Any) -> bool:
        """Controlla se l'oggetto è un PineconeVectorStore che necessita chiusura"""
        class_name = obj.__class__.__name__
        return class_name == 'PineconeVectorStore' or 'VectorStore' in class_name

    @classmethod
    def _is_embeddings(cls, obj: Any) -> bool:
        """Controlla se l'oggetto è un Embeddings che necessita chiusura"""
        class_name = obj.__class__.__name__
        return class_name == 'Embeddings' in class_name

    @classmethod
    def _needs_connection_cleanup(cls, obj: Any) -> bool:
        """Controlla se l'oggetto necessita chiusura connessioni"""
        if cls._is_cached_vector_store_wrapper(obj):
            return True
        return cls._is_pinecone_repository(obj) or cls._is_vector_store(obj)

    @classmethod
    async def _close_pinecone_connection(cls, obj: Any, key: Any) -> None:
        try:
            # chiudi il wrapper se presente
            if cls._is_cached_vector_store_wrapper(obj) and hasattr(obj, "close"):
                res = obj.close()
                if asyncio.iscoroutine(res):
                    await res
                else:
                    obj.close()
                logger.info(f"Wrapper Pinecone chiuso per {key}")
                return
            # Vector store
            if cls._is_vector_store(obj):
                if hasattr(obj, 'index') and hasattr(obj.index, 'close'):
                    if asyncio.iscoroutinefunction(obj.index.close):
                        await obj.index.close()
                    else:
                        obj.index.close()
                    logger.info(f"Vector store index connection chiusa per oggetto {key}")
                    return

            # Repository Pinecone
            if cls._is_pinecone_repository(obj):
                if hasattr(obj, 'close_connection'):
                    if asyncio.iscoroutinefunction(obj.close_connection):
                        await obj.close_connection()
                    else:
                        obj.close_connection()
                    logger.info(f"Connessione Repository Pinecone chiusa per oggetto {key} tramite close_connection()")
                elif hasattr(obj, 'vector_store') and hasattr(obj.vector_store, 'close'):
                    if asyncio.iscoroutinefunction(obj.vector_store.close):
                        await obj.vector_store.close()
                    else:
                        obj.vector_store.close()
                    logger.info(
                        f"Connessione Repository Pinecone chiusa per oggetto {key} tramite vector_store.close()")
                elif hasattr(obj, 'index') and hasattr(obj.index, 'close'):
                    if asyncio.iscoroutinefunction(obj.index.close):
                        await obj.index.close()
                    else:
                        obj.index.close()
                    logger.info(f"Connessione Repository Pinecone chiusa per oggetto {key} tramite index.close()")
                else:
                    logger.warning(f"Nessun metodo di chiusura trovato per Repository Pinecone {key}")

        except Exception as ex:
            logger.error(f"Errore durante la chiusura della connessione Pinecone per {key}: {ex}")

    @classmethod
    def _close_pinecone_connection_sync(cls, obj: Any, key: Any) -> None:
        try:
            if hasattr(obj, 'close_connection') and not asyncio.iscoroutinefunction(obj.close_connection):
                obj.close_connection()
                logger.info(f"Connessione Pinecone chiusa per oggetto {key} tramite close_connection()")
            elif hasattr(obj, 'vector_store') and hasattr(obj.vector_store,
                                                          'close') and not asyncio.iscoroutinefunction(
                    obj.vector_store.close):
                obj.vector_store.close()
                logger.info(f"Connessione Pinecone chiusa per oggetto {key} tramite vector_store.close()")
            elif hasattr(obj, 'index') and hasattr(obj.index, 'close') and not asyncio.iscoroutinefunction(
                    obj.index.close):
                obj.index.close()
                logger.info(f"Connessione Pinecone chiusa per oggetto {key} tramite index.close()")
        except Exception as ex:
            logger.error(f"Errore durante la chiusura sincrona della connessione Pinecone per {key}: {ex}")

    @classmethod
    async def _close_embeddings_async(cls, obj: Any, key: Any):
        try:
            close = getattr(obj, "aclose", None)
            if callable(close):
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception as ex:
            logger.error(f"Errore chiusura embeddings async per {key}: {ex}")

    @classmethod
    def _close_embeddings_sync(cls, obj: Any, key: Any):
        try:
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        except Exception as ex:
            logger.error(f"Errore chiusura embeddings sync per {key}: {ex}")

    @classmethod
    def get(
            cls,
            object_type: str,
            key: Tuple,
            constructor: Callable,
            *args,
            **kwargs
    ) -> Any:
        """Ottieni un oggetto dalla cache o crealo se non presente"""
        if object_type not in cls._caches:
            cls._caches[object_type] = OrderedDict()

        cache = cls._caches[object_type]
        now = time.time()


        pol = cls._policy(object_type)
        timeout = pol.get("timeout_seconds", cls._timeout_seconds)
        max_size = pol.get("max_size", cls._max_size)

        refresh_on_access = pol.get("refresh_on_access", True)
        close_on_evict = pol.get("close_on_evict", False)

        # Fase 1: Rimuovi gli oggetti scaduti
        expired_keys = []
        if timeout is not None:
            for k, (obj, timestamp) in list(cache.items()):
                if now - timestamp > timeout:  # Controllo scadenza
                    expired_keys.append(k)

        for k in expired_keys:
            obj, timestamp = cache[k]
            if close_on_evict and cls._needs_connection_cleanup(obj):
                cls._close_pinecone_connection_sync(obj, k)
                # Non chiudiamo embeddings qui
            del cache[k]
            logger.info(f"Rimosso oggetto scaduto: {object_type}/{k}")

        # Fase 2: Rimuovi gli oggetti meno recenti se si supera la dimensione massima
        while len(cache) > max_size:
            k, (obj, timestamp) = cache.popitem(last=False)  # Rimuove il più vecchio
            if close_on_evict and cls._needs_connection_cleanup(obj):
                cls._close_pinecone_connection_sync(obj, k)
            logger.info(f"Rimosso oggetto per superamento limite cache: {object_type}/{k}")

        # Se l'oggetto è in cache, aggiorna la posizione ma NON il timestamp
        if key in cache:
            obj, timestamp = cache[key]
            cache.move_to_end(key)  # Sposta alla fine (più recente)
            if refresh_on_access:
                cache[key] = (obj, now)
            logger.debug(f"Oggetto {object_type}/{key} recuperato dalla cache")
            return obj

        # Miss: Altrimenti crea un nuovo oggetto
        logger.info(f"Creazione nuovo oggetto {object_type}/{key}")
        obj = constructor(*args, **kwargs)
        cache[key] = (obj, now)  # Registra il timestamp di creazione

        # Controllo aggiuntivo dimensione cache dopo inserimento
        while len(cache) > max_size:
            k, (obj_to_remove, timestamp) = cache.popitem(last=False)
            if close_on_evict and cls._needs_connection_cleanup(obj_to_remove):
                cls._close_pinecone_connection_sync(obj_to_remove, k)
            logger.info(f"Rimosso oggetto per superamento limite cache (dopo inserimento): {object_type}/{k}")

        return obj

    @classmethod
    async def async_get(
            cls,
            object_type: str,
            key: Tuple,
            constructor: Callable[[], Any],
            ttl: Optional[int] = None,  # Aggiungi ttl come parametro esplicito
            *args,
            **kwargs) -> Any:

        if object_type not in cls._caches:
            cls._caches[object_type] = OrderedDict()

        cache = cls._caches[object_type]
        now = time.time()
        pol = cls._policy(object_type)

        timeout = ttl if ttl is not None else pol.get("timeout_seconds", cls._timeout_seconds)
        max_size = pol.get("max_size", cls._max_size)
        refresh_on_access = pol.get("refresh_on_access", True)
        close_on_evict = pol.get("close_on_evict", False)

        # Rimozione scaduti
        expired_keys = []
        if timeout is not None:
            for k, (obj, ts) in list(cache.items()):
                if now - ts > timeout:
                    expired_keys.append(k)

        for k in expired_keys:
            obj, ts = cache[k]
            if close_on_evict and cls._needs_connection_cleanup(obj):
                await cls._close_pinecone_connection(obj, k)
            # Non chiudiamo embeddings qui
            del cache[k]
            logger.info(f"Rimosso oggetto scaduto: {object_type}/{k}")

        # Controllo dimensione
        while len(cache) > max_size:
            k, (obj, ts) = cache.popitem(last=False)
            if close_on_evict and cls._needs_connection_cleanup(obj):
                await cls._close_pinecone_connection(obj, k)
            logger.info(f"Rimosso oggetto per superamento limite cache: {object_type}/{k}")

        # Cache hit
        if key in cache:
            obj, ts = cache[key]
            cache.move_to_end(key)
            if refresh_on_access:
                cache[key] = (obj, now)
            logger.debug(f"Oggetto {object_type}/{key} recuperato dalla cache")
            return obj

        # Miss: crea
        logger.info(f"Creazione nuovo oggetto {object_type}/{key}")
        obj = await constructor(*args, **kwargs)
        cache[key] = (obj, now)
        return obj

    @classmethod
    def clear_cache(cls, object_type: Optional[str] = None):
        import torch
        if object_type:
            if object_type in cls._caches:
                cache = cls._caches[object_type]
                # Chiudi Pinecone e, se richiesto, embeddings (solo su clear)
                for k, (obj, ts) in list(cache.items()):
                    if cls._is_pinecone_repository(obj) or cls._is_vector_store(obj):
                        cls._close_pinecone_connection_sync(obj, k)
                    if object_type == "embedding" and cls._is_embeddings(obj):
                        cls._close_embeddings_sync(obj, k)
                cache.clear()
                logger.info(f"Cache svuotata per {object_type}")
        else:
            # Clear completo
            for ot, cache in cls._caches.items():
                for k, (obj, ts) in list(cache.items()):
                    if cls._is_pinecone_repository(obj) or cls._is_vector_store(obj):
                        cls._close_pinecone_connection_sync(obj, k)
                    if ot == "embedding" and cls._is_embeddings(obj):
                        cls._close_embeddings_sync(obj, k)
                cache.clear()
            logger.info("Cache globale svuotata")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    async def async_clear_cache(cls, object_type: Optional[str] = None):
        import torch
        if object_type:
            if object_type in cls._caches:
                cache = cls._caches[object_type]
                for k, (obj, ts) in list(cache.items()):
                    if cls._is_pinecone_repository(obj) or cls._is_vector_store(obj):
                        await cls._close_pinecone_connection(obj, k)
                    if object_type == "embedding" and cls._is_embeddings(obj):
                        await cls._close_embeddings_async(obj, k)
                cache.clear()
                logger.info(f"Cache svuotata per {object_type}")
        else:
            for ot, cache in cls._caches.items():
                for k, (obj, ts) in list(cache.items()):
                    if cls._is_pinecone_repository(obj) or cls._is_vector_store(obj):
                        await cls._close_pinecone_connection(obj, k)
                    if ot == "embedding" and cls._is_embeddings(obj):
                        await cls._close_embeddings_async(obj, k)
                cache.clear()
            logger.info("Cache globale svuotata")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    async def async_remove(cls, object_type: str, key: Tuple):
        """
        Rimuove un oggetto specifico dalla cache in modo asincrono.
        Gestisce la chiusura delle connessioni (es. Pinecone, embeddings) se l'oggetto lo richiede.
        """
        if object_type in cls._caches and key in cls._caches[object_type]:
            cache = cls._caches[object_type]
            obj, _ = cache.pop(key)  # Rimuovi l'oggetto dalla cache

            # Controlla se l'oggetto rimosso richiede la chiusura di connessioni asincrone
            if cls._needs_connection_cleanup(obj):
                await cls._close_pinecone_connection(obj, key)
            elif object_type == "embedding" and cls._is_embeddings(obj):
                await cls._close_embeddings_async(obj, key)

            logger.info(f"Oggetto {object_type}/{key} rimosso con successo dalla cache.")
        else:
            logger.warning(f"Tentativo di rimuovere oggetto non trovato: {object_type}/{key}")
