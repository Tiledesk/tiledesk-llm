import asyncio
import time
from typing import Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec #, PodSpec
import logging

logger = logging.getLogger(__name__)


class PersistentPineconeClient:
    """Client Pinecone persistente con gestione cache degli indici (versione sincrona)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        self._client_lock = asyncio.Lock()
        self.created_at = None
        self.last_used = None

        # Cache degli indici per questo client
        self._index_cache: Dict[str, Any] = {}
        self._index_locks: Dict[str, asyncio.Lock] = {}

    async def get_client(self):
        """Ottieni il client, creandolo se necessario"""
        self.last_used = time.time()
        if self._client is None:
            async with self._client_lock:
                if self._client is None:  # Double-check locking
                    # Usa Pinecone sincrono invece di PineconeAsyncio
                    self._client = Pinecone(api_key=self.api_key)
                    self.created_at = time.time()
                    logger.info("Nuovo client Pinecone creato")
        return self._client

    async def get_or_create_index(self, engine, emb_dimension: int):
        """Ottieni l'indice dalla cache o crealo se non esiste"""
        cache_key = engine.index_name

        # Controlla cache
        if cache_key in self._index_cache:
            index = self._index_cache[cache_key]
            # Testa la connessione
            if await self._test_index(index):
                return index
            else:
                # Rimuovi dalla cache se non funziona
                self._index_cache.pop(cache_key, None)

        # Ottieni lock per questo indice
        lock = self._index_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Double-check dopo aver ottenuto il lock
            if cache_key in self._index_cache:
                index = self._index_cache[cache_key]
                if await self._test_index(index):
                    return index
                else:
                    self._index_cache.pop(cache_key, None)

            # Crea/assicurati l'esistenza dell'indice
            client = await self.get_client()
            host = await self._ensure_index_exists(client, engine, emb_dimension)

            # Crea l'handle dell'indice
            effective_host = host or getattr(engine, "host", None)
            index = self._get_index_handle(client, engine.index_name, host=effective_host)

            # Salva in cache
            self._index_cache[cache_key] = index
            return index

    async def _ensure_index_exists(self, pc, engine, emb_dimension: int) -> Optional[str]:
        """Crea l'indice se non esiste; ritorna l'host (utile per pod), None altrimenti."""

        # Esegui operazioni sincrone in thread separato per non bloccare il loop
        def _sync_operations():
            # Check esistenza via list_indexes
            existing = pc.list_indexes()
            exists = engine.index_name in existing.names()

            if not exists:
                logger.info(f"Creating index '{engine.index_name}' ({engine.type})...")
                metric = getattr(engine, "metric", "cosine") or "cosine"

                if engine.type == "serverless":
                    cloud = getattr(engine, "cloud", "aws")
                    region = getattr(engine, "region", "us-east-1")
                    try:
                        pc.create_index(
                            name=engine.index_name,
                            dimension=emb_dimension,
                            metric=metric,
                            spec=ServerlessSpec(cloud=cloud, region=region),
                        )

                    except Exception as e:
                        # In caso di race: se "already exists", prosegui
                        if "already exists" not in str(e).lower():
                            raise
                else:
                    # Pod: prendi impostazioni da engine
                    environment = getattr(engine, "environment", "us-west1-gcp")
                    pod_type = getattr(engine, "pod_type", "p1")
                    pods = getattr(engine, "pods", 1)
                    logger.debug(f"Index creation '{engine.index_name}' ({engine.type}) ({environment}) ({pod_type}) ({pods}) Disabled.")
                    try:
                        logger.info("TENTATIVO DI CREAZIONE DI INDICE - Creazione su POD disabilitata")
                        pass


                        #pc.create_index(
                        #    name=engine.index_name,
                        #    dimension=emb_dimension,
                        #    metric=metric,
                        #    spec=PodSpec(environment=environment, pod_type=pod_type, pods=pods),
                        #)
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            raise

                # Attesa readiness
                desc = self._wait_index_ready_sync(pc, engine.index_name)
                return getattr(desc, "host", None)

            # Se esiste già, per pod ricava host (per serverless non serve)
            return self._describe_host_sync(pc, engine.index_name)

        # Esegui in thread separato per non bloccare il loop asyncio
        return await asyncio.to_thread(_sync_operations)

    def _wait_index_ready_sync(self, pc, index_name: str, timeout_s: int = 300, base_sleep: float = 1.0):
        """Attende che l'indice sia pronto (versione sincrona)"""
        start = time.perf_counter()
        sleep = base_sleep
        while True:
            desc = pc.describe_index(index_name)
            status = getattr(desc, "status", {}) or {}
            ready = status.get("ready", False)
            if ready:
                return desc
            if time.perf_counter() - start > timeout_s:
                raise TimeoutError(f"Timeout waiting for index '{index_name}' to be ready")
            time.sleep(sleep)  # Sleep sincrono
            sleep = min(sleep * 1.3, 5.0)

    def _describe_host_sync(self, pc, index_name: str) -> Optional[str]:
        """Ottieni l'host per un indice esistente (versione sincrona)"""
        try:
            desc = pc.describe_index(index_name)
            return getattr(desc, "host", None)
        except Exception:
            return None

    def _get_index_handle(self, pc, index_name: str, host: Optional[str] = None):
        """Crea l'handle dell'indice (versione sincrona)"""
        # Usa Index invece di IndexAsyncio
        if host:
            return pc.Index(index_name=index_name, host=host)
        else:
            return pc.Index(index_name=index_name)

    async def _test_index(self, index) -> bool:
        """Testa se l'indice è ancora valido"""

        def _sync_test():
            try:
                index.describe_index_stats()  # Chiamata sincrona
                return True
            except Exception:
                return False

        # Esegui in thread separato
        return await asyncio.to_thread(_sync_test)

    def invalidate_index(self, index_name: str):
        """Rimuovi un indice dalla cache"""
        self._index_cache.pop(index_name, None)
        self._index_locks.pop(index_name, None)

    def invalidate_all_indexes(self):
        """Rimuovi tutti gli indici dalla cache"""
        self._index_cache.clear()
        self._index_locks.clear()

    async def close(self):
        """Chiudi il client e pulisci le cache"""
        # Il client sincrono non ha bisogno di chiusura speciale
        logger.info("Client Pinecone chiuso (sincrono)")

        # Pulisci le cache
        self._client = None
        self.invalidate_all_indexes()

    def is_expired(self, ttl: int = 7200) -> bool:
        """Verifica se il client è scaduto"""
        if self.created_at is None:
            return True
        return (time.time() - self.created_at) > ttl

    async def close_connection(self):
        """Metodo di compatibilità per TimedCache"""
        await self.close()