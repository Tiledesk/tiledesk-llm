import time
from collections import OrderedDict
from functools import wraps
import logging
import torch
from typing import Any, Dict, Optional, Tuple, Callable, Type

logger = logging.getLogger("GlobalCache")


class TimedCache:
    """
    Cache avanzata con scadenza basata sull'inattività.
    Gestisce automaticamente la rimozione degli oggetti dopo un periodo di inattività.
    Supporta diversi tipi di oggetti (Embeddings, Chat, Repository) con chiavi univoche.
    """
    _caches: Dict[str, OrderedDict] = {}
    _timeout_seconds: float = 300  # 5 minuti
    _max_size: int = 100  # Limite massimo di oggetti per tipo

    @classmethod
    def get_old(
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

        # Pulisci cache: rimuovi oggetti scaduti o oltre il limite
        expired_keys = []
        for k, (obj, timestamp) in list(cache.items()):
            if now - timestamp > cls._timeout_seconds or len(cache) > cls._max_size:
                expired_keys.append(k)

        for k in expired_keys:
            del cache[k]
            logger.info(f"Rimosso oggetto scaduto: {object_type}/{k}")

        # Se l'oggetto è in cache, aggiorna il timestamp e restituiscilo
        if key in cache:
            obj, _ = cache[key]
            cache.move_to_end(key)  # Sposta alla fine (più recente)
            cache[key] = (obj, now)
            logger.info(f"Oggetto {object_type}/{key} recuperato dalla cache")
            #print(f"Oggetto {object_type}/{key} recuperato dalla cache")
            return obj

        # Altrimenti crea un nuovo oggetto
        #print(f"Creazione nuovo oggetto {object_type}/{key}")
        logger.info(f"Creazione nuovo oggetto {object_type}/{key}")
        obj = constructor(*args, **kwargs)
        cache[key] = (obj, now)
        return obj

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

        # Fase 1: Rimuovi gli oggetti scaduti
        expired_keys = []
        for k, (obj, timestamp) in list(cache.items()):
            if now - timestamp > cls._timeout_seconds:  # Controllo scadenza
                expired_keys.append(k)

        for k in expired_keys:
            del cache[k]
            logger.info(f"Rimosso oggetto scaduto: {object_type}/{k}")

        # Fase 2: Rimuovi gli oggetti meno recenti se si supera la dimensione massima
        while len(cache) > cls._max_size:
            k, (obj, timestamp) = cache.popitem(last=False)  # Rimuove il più vecchio
            logger.info(f"Rimosso oggetto per superamento limite cache: {object_type}/{k}")

        # Se l'oggetto è in cache, aggiorna la posizione ma NON il timestamp
        if key in cache:
            obj, timestamp = cache[key]
            cache.move_to_end(key)  # Sposta alla fine (più recente)
            logger.debug(f"Oggetto {object_type}/{key} recuperato dalla cache")
            return obj

        # Altrimenti crea un nuovo oggetto
        logger.info(f"Creazione nuovo oggetto {object_type}/{key}")
        obj = constructor(*args, **kwargs)
        cache[key] = (obj, now)  # Registra il timestamp di creazione

        # Controllo aggiuntivo dimensione cache dopo inserimento
        while len(cache) > cls._max_size:
            k, (obj, timestamp) = cache.popitem(last=False)
            logger.info(f"Rimosso oggetto per superamento limite cache (dopo inserimento): {object_type}/{k}")

        return obj


    @classmethod
    def clear_cache(cls, object_type: Optional[str] = None):
        """Svuota completamente la cache o una specifica categoria"""
        if object_type:
            if object_type in cls._caches:
                cls._caches[object_type].clear()
                logger.info(f"Cache svuotata per {object_type}")
        else:
            for cache in cls._caches.values():
                cache.clear()
            logger.info("Cache globale svuotata")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()