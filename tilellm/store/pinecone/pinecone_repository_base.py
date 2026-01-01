import uuid
from abc import abstractmethod

import time
import asyncio

import pinecone
from langchain_core.documents import Document
from langchain_pinecone.vectorstores import PineconeVectorStore


from tilellm.models.schemas import (RepositoryQueryResult,
                                    RepositoryItems,
                                    RepositoryNamespaceResult,
                                    RepositoryItemNamespaceResult,
                                    RepositoryIdSummaryResult,
                                    RepositoryDescNamespaceResult,
                                    RepositoryNamespace
                                    )
from tilellm.models import (Engine,
                            QuestionAnswer
                            )


from typing import Dict, List, Optional, Union

import logging

from tilellm.models.llm import TEIConfig

from tilellm.shared.timed_cache import TimedCache
from tilellm.shared.utility import _hash_api_key

from tilellm.store.vector_store_repository import VectorStoreRepository
from tilellm.tools.sparse_encoders import TiledeskSparseEncoders

logger = logging.getLogger(__name__)

class CachedVectorStore:
    def __init__(self, engine, embeddings, emb_dimension):
        self.engine = engine
        self.embeddings = embeddings
        self.emb_dimension = emb_dimension
        self._pc_client = None
        self._host = None
        self._lock = asyncio.Lock()
        self._loop = None
        self._last_check = 0.0
        self._check_every_sec = 60  # throttling

    async def _ensure_client_and_host(self):
        cur_loop = asyncio.get_running_loop()
        async with self._lock:
            if self._pc_client is None or self._loop is not cur_loop:
                # (ri)crea client sul loop corrente
                if self._pc_client:
                    try:
                        await self._pc_client.close()
                    except Exception as e:
                        logger.debug(f"Exception on Session close {repr(e)}")
                        pass
                self._pc_client = pinecone.PineconeAsyncio(api_key=self.engine.apikey.get_secret_value())
                self._loop = cur_loop
                self._host = None
            if self._host is None:
                existing = await self._pc_client.list_indexes()
                if self.engine.index_name not in existing.names():
                    await self._create_index_if_not_exists()
                self._host = (await self._pc_client.describe_index(self.engine.index_name)).host

    async def get_vector_store(self):
        idx = await self.get_index()
        return PineconeVectorStore(
            index=idx,
            embedding=self.embeddings,
            text_key=self.engine.text_key,
        )

    async def get_index(self):
        await self._ensure_client_and_host()
        return self._pc_client.IndexAsyncio(name=self.engine.index_name, host=self._host)

    async def test_connection(self):
        now = time.time()
        if now - self._last_check < self._check_every_sec:
            return True
        try:
            idx = await self.get_index()
            await idx.describe_index_stats()
            self._last_check = now
            return True
        except Exception as e:
            # recovery soft se sessione chiusa
            if "Session is closed" in str(e):
                async with self._lock:
                    if self._pc_client:
                        try:
                            await self._pc_client.close()
                        except Exception as e:
                            logger.debug(f"Exception on Session close {repr(e)}")
                            pass
                    self._pc_client = None
                    self._host = None
                # retry singolo
                try:
                    idx = await self.get_index()
                    await idx.describe_index_stats()
                    self._last_check = time.time()
                    return True
                except Exception as e:
                    logger.debug(f"Exception {repr(e)}")
                    return False
            return False

    async def _create_index_if_not_exists(self):
        """Crea l'indice se non esiste"""
        logger.info(f'Creating new index {self.engine.index_name}...')

        if self.engine.type == "serverless":
            await self._pc_client.create_index(
                name=self.engine.index_name,
                dimension=self.emb_dimension,
                metric=self.engine.metric,
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        else:  # Pod type
            await self._pc_client.create_index(
                name=self.engine.index_name,
                dimension=self.emb_dimension,
                metric=self.engine.metric,
                spec=pinecone.PodSpec(
                    pod_type="p1",
                    pods=1,
                    environment="us-west4-gpc"
                )
            )

        # Attendi che l'indice sia pronto
        while not (await self._pc_client.describe_index(self.engine.index_name)).status["ready"]:
            logger.debug(f"Waiting for index {self.engine.index_name} to be ready...")
            await asyncio.sleep(1)

    async def close(self):
        if self._pc_client:
            try:
                await self._pc_client.close()
            finally:
                self._pc_client = None
                self._host = None
                self._loop = None



class PineconeRepositoryBase(VectorStoreRepository):
    sparse_enabled = False



    @abstractmethod
    async def search_community_report(self, question_answer, index, dense_vector, sparse_vector):
        pass

    async def aadd_documents(self, engine: Engine, documents: List[Document], namespace: str, embedding_model: any, sparse_encoder: Union[str, TEIConfig, None] = None, **kwargs):
        logger.info(f"Adding {len(documents)} documents to namespace '{namespace}' with hybrid embeddings.")
        
        # Ensure sparse_encoder is initialized if hybrid search is desired
        if sparse_encoder is None:
            sparse_encoder = TiledeskSparseEncoders("default") # Initialize with a default sparse encoder

        # 1. Get Pinecone Index
        emb_dimension = await self.get_embeddings_dimension(embedding_model)
        vector_store_instance = await self.create_index(engine, embedding_model, emb_dimension) # Using create_index, which handles caching
        index = await vector_store_instance.async_index # Get the actual async index object


        try:
            # 2. Clear namespace before adding new documents
            logger.info(f"Clearing namespace '{namespace}' before upserting.")
            await index.delete(delete_all=True, namespace=namespace)

            # 3. Prepare data and embeddings
            doc_batch_size = 50 # Pinecone has limits on request size. Adjust as needed.
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            for i in range(0, len(documents), doc_batch_size):
                batch_contents = contents[i: i + doc_batch_size]
                batch_metadatas = metadatas[i: i + doc_batch_size]
                batch_ids = [str(uuid.uuid4()) for _ in batch_contents] # Generate UUIDs for IDs

                # Generate dense embeddings
                dense_embeds = await embedding_model.aembed_documents(batch_contents)

                # Generate sparse embeddings
                sparse_embeds = sparse_encoder.encode_documents(batch_contents)

                # 4. Upsert to Pinecone
                vectors_to_upsert = []
                for j, content in enumerate(batch_contents):
                    # Combine original metadata with page_content for text_key and namespace for filtering
                    combined_metadata = {
                        **batch_metadatas[j],
                        engine.text_key: content, # This is crucial for text retrieval
                        "namespace": namespace # Ensure namespace is in metadata for easy filtering
                    }

                    vector = {
                        "id": batch_ids[j],
                        "values": dense_embeds[j],
                        "metadata": combined_metadata,
                        "sparse_values": sparse_embeds[j] # Add sparse vector
                    }
                    vectors_to_upsert.append(vector)
                
                await index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                logger.info(f"Upserted batch {i//doc_batch_size + 1} with {len(vectors_to_upsert)} vectors to namespace '{namespace}'.")
            
            logger.info(f"Successfully added {len(documents)} documents to namespace '{namespace}'.")
            # Return generated IDs for traceability
            return [vec["id"] for vec in vectors_to_upsert]

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            raise
        finally:
            if index:
                await index.close()
    
    @abstractmethod
    async def add_item(self, item):
        pass


    @abstractmethod
    async def add_item_hybrid(self, item):
        pass

    @abstractmethod
    async def get_chunks_from_repo(self, question_answer: QuestionAnswer): #, embedding_obj=None, embedding_dimension=None):
        pass

    async def delete_namespace(self, namespace_to_delete: RepositoryNamespace):
        """
        Delete namespace from Pinecone index
        :param namespace_to_delete:
        :return:
        """
        engine = namespace_to_delete.engine
        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )
            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)
            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            async with index as index:
                await index.delete(delete_all=True, namespace=namespace_to_delete.namespace)
        except Exception as ex:

            logger.error(ex)

            raise ex

    @abstractmethod
    async def delete_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        pass

    async def delete_chunk_id_namespace(self, engine: Engine, chunk_id: str, namespace: str):
        """
        delete chunk from pinecone
        :param engine: Engine
        :param chunk_id:
        :param namespace:
        :return:
        """

        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )
            host = pc.describe_index(engine.index_name).host

            #index = pc.Index(name=engine.index_name, host=host)
            index = pc.IndexAsyncio(name=engine.index_name, host=host)


            async with index as index:
                #describe = await index.describe_index_stats()

                await index.delete(ids=[chunk_id], namespace=namespace)
        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
        """
        Get from Pinecone all items from namespace given document id
        :param engine: Engine
        :param metadata_id:
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            async with index as index:
                describe = await index.describe_index_stats()
                # print(describe)
                logger.debug(describe)
                namespaces = describe.namespaces
                total_vectors = 1

                if namespaces:
                    if namespace in namespaces.keys():
                        total_vectors = namespaces.get(namespace).get('vector_count')

                logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")
                batch_size = min([total_vectors, 10000])
                pc_res = await index.query(
                    vector=[0] * 1536,  # [0,0,0,0......0]
                    top_k=batch_size,
                    filter={"id": {"$eq": metadata_id}},
                    namespace=namespace,
                    include_values=False,
                    include_metadata=True
                )
            matches = pc_res.get('matches')
            # from pprint import pprint
            # pprint(matches)
            # ids = [obj.get('id') for obj in matches]
            # print(type(matches[0].get('id')))
            result = []
            for obj in matches:
                result.append(RepositoryQueryResult(id=obj.get('id', ""),
                                                    metadata_id=obj.get('metadata').get('id'),
                                                    metadata_source=obj.get('metadata').get('source'),
                                                    metadata_type=obj.get('metadata').get('type'),
                                                    date=obj.get('metadata').get('date', 'Date not defined'),
                                                    text=obj.get('metadata').get(engine.text_key)
                                                    # su pod content, su Serverless text
                                                    )
                              )
            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def list_namespaces(self, engine: Engine) -> RepositoryNamespaceResult:
        """

        :param engine:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            async with index as index:
                describe = await index.describe_index_stats()

                logger.debug(describe)
                namespaces = describe.namespaces

                results = []

                for namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')
                    pc_item_namespace = RepositoryItemNamespaceResult(namespace=namespace, vector_count=total_vectors)
                    results.append(pc_item_namespace)
                    logger.debug(f"{namespace}, {total_vectors}")

            logger.debug(f"pinecone total vector in {results}")

            return RepositoryNamespaceResult(namespaces=results)

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_all_obj_namespace(self, engine: Engine, namespace: str, with_text:bool=False) -> RepositoryItems:
        """
        Query Pinecone to get all object
        :param engine: Engine
        :param namespace:
        :param with_text:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            async with index as index:
                # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
                describe = await index.describe_index_stats()

                logger.debug(describe)
                namespaces = describe.namespaces
                total_vectors = 1

                if namespaces:
                    if namespace in namespaces.keys():
                        total_vectors = namespaces.get(namespace).get('vector_count')

                logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")

                batch_size = min([total_vectors, 1000])

                pc_res = await index.query(
                    vector=[0] * 1536,  # [0,0,0,0......0]
                    top_k=batch_size,
                    # filter={"id": {"$eq": id}},
                    namespace=namespace,
                    include_values=False,
                    include_metadata=True
                )
            matches = pc_res.get('matches')
            # from pprint import pprint
            # pprint(matches)
            # ids = [obj.get('id') for obj in matches]
            # print(type(matches[0].get('id')))
            result = []

            for obj in matches:
                result.append(RepositoryQueryResult(id=obj.get('id', ""),
                                                    metadata_id=obj.get('metadata').get('id'),
                                                    metadata_source=obj.get('metadata').get('source'),
                                                    metadata_type=obj.get('metadata').get('type'),
                                                    date=obj.get('metadata').get('date', 'Date not defined'),
                                                    text=obj.get('metadata').get(engine.text_key) if with_text else None  # su pod content, su Serverless text
                                                    )
                              )
            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_desc_namespace(self, engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        """
        Query Pinecone to get all object
        :param engine: Engine
        :param namespace:
        :return: PineconeDescNamespaceResult
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            async with index as index:
                describe = await index.describe_index_stats()

                logger.debug(describe)
                namespaces = describe.namespaces
                total_vectors = 1
                description = RepositoryItemNamespaceResult(namespace=namespace, vector_count=0)
                if namespaces:
                    if namespace in namespaces.keys():
                        total_vectors = namespaces.get(namespace).get('vector_count')
                        description = RepositoryItemNamespaceResult(namespace=namespace, vector_count=total_vectors)

                logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")

                batch_size = min([total_vectors, 10000])

                pc_res = await index.query(
                    vector=[0] * 1536,  # [0,0,0,0......0]
                    top_k=batch_size,
                    # filter={"id": {"$eq": id}},
                    namespace=namespace,
                    include_values=False,
                    include_metadata=True
                )
            matches = pc_res.get('matches')
            # from pprint import pprint
            # pprint(matches)
            # ids = [obj.get('id') for obj in matches]
            # print(type(matches[0].get('id')))
            # result = []
            ids_count: Dict[str, RepositoryIdSummaryResult] = {}

            for obj in matches:
                metadata_id = obj.get('metadata').get('id')
                if metadata_id in ids_count:
                    ids_count[metadata_id].chunks_count += 1
                else:
                    ids_count[metadata_id] = RepositoryIdSummaryResult(metadata_id=metadata_id,
                                                                       source=obj.get('metadata').get('source'),
                                                                       chunks_count=1)

            res = RepositoryDescNamespaceResult(namespace_desc=description, ids=list(ids_count.values()))

            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    async def get_sources_namespace(self, engine: Engine, source: str, namespace: str) -> RepositoryItems:
        """
        Get from Pinecone all items from namespace given source
        :param engine: Engine
        :param source:
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey.get_secret_value()
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            async with index as index:
                describe = await index.describe_index_stats()
                logger.debug(describe)
                namespaces = describe.namespaces
                total_vectors = 1

                if namespaces:
                    if namespace in namespaces.keys():
                        total_vectors = namespaces.get(namespace).get('vector_count')

                logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")
                pc_res = await index.query(
                    vector=[0] * 1536,  # [0,0,0,0......0]
                    top_k=total_vectors,
                    filter={"source": {"$eq": source}},
                    namespace=namespace,
                    include_values=False,
                    include_metadata=True
                )
            matches = pc_res.get('matches')
            # from pprint import pprint
            # pprint(matches)
            # ids = [obj.get('id') for obj in matches]
            # print(type(matches[0].get('id')))
            result = []
            for obj in matches:
                result.append(RepositoryQueryResult(id=obj.get('id'),
                                                    metadata_id=obj.get('metadata').get('id'),
                                                    metadata_source=obj.get('metadata').get('source'),
                                                    metadata_type=obj.get('metadata').get('type'),
                                                    date=obj.get('metadata').get('date', 'Date not defined'),
                                                    text=obj.get('metadata').get(engine.text_key)
                                                    # su pod content, su Serverless text
                                                    )
                              )
            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    def chunk_data(data, chunk_size=256, chunk_overlap=10):
        """
        Chunk document in small pieces
        :param data:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """

        from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks

    @staticmethod
    def chunk_data_extended(data, chunk_size=256, chunk_overlap=10, **kwargs):
        """
        Chunk document in small pieces. Semantic chunking is implemented too with
        percentile, standard_deviation, interquartile, gradient

        :param data:
        :param chunk_size:
        :param chunk_overlap:
        :param kwargs:
        :return:
        """

        use_semantic_chunk = kwargs['semantic']
        if use_semantic_chunk:
            embeddings = kwargs['embeddings']
            breakpoint_threshold_type = kwargs['breakpoint_threshold_type']
            logger.info(f"Semantic chunk with {breakpoint_threshold_type}")
            from langchain_experimental.text_splitter import SemanticChunker
            text_splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type

            )
            chunks = text_splitter.split_documents(data)
        else:
            from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(data)

        return chunks

    @staticmethod
    def calc_embedding_cost(texts, embedding):
        """
        Calculate the embedding cost with OpenAI embedding
        :param texts:
        :param embedding:
        :return:
        """
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        logger.info(f'Total numer of Token: {total_tokens}')
        cost = 0
        try:
            if embedding == "text-embedding-3-large":
                cost = total_tokens / 1e6 * 0.13
            elif embedding == "text-embedding-3-small":
                cost = total_tokens / 1e6 * 0.02
            else:
                embedding = "text-embedding-ada-002"
                cost = total_tokens / 1e6 * 0.10

        except IndexError:
            embedding = "text-embedding-ada-002"
            cost = total_tokens / 1e6 * 0.10

        logger.info(f'Embedding cost $: {cost:.6f}')
        return total_tokens, cost

    @staticmethod
    async def get_embeddings_dimension(embedding):
        """
        Get embedding dimension for OpenAI embedding model
        :param embedding:
        :return:
        """
        emb_dimension = 1536
        try:
            if embedding == "text-embedding-3-large":
                emb_dimension = 3072
            elif embedding == "text-embedding-3-small":
                emb_dimension = 1536
            else:
                embedding = "text-embedding-ada-002"
                emb_dimension = 1536

        except IndexError:
            embedding = "text-embedding-ada-002"
            emb_dimension = 1536

        return emb_dimension



    @staticmethod
    async def create_index_instance(engine, embeddings, emb_dimension) -> PineconeVectorStore:
        """
        Create or return existing index. Questo metodo attualmente non è utilizzato. Viene creata un'istanza
        di PineconeAsyncio ad ogni uso.
        :param engine:
        :param embeddings:
        :param emb_dimension:
        :return:
        """
        import pinecone

        pc =  pinecone.PineconeAsyncio(
            api_key= engine.apikey.get_secret_value()
        )

        if engine.index_name in (await pc.list_indexes()).names(): #const.PINECONE_INDEX in pc.list_indexes().names():
            #logger.debug(engine.index_name)
            logger.debug(f'Index {engine.index_name} exists. Loading embeddings ... ')
            host = (await pc.describe_index(engine.index_name)).host

            async with pc as pc:
                index = pc.IndexAsyncio(name=engine.index_name,
                                 host=host)

                vector_store = PineconeVectorStore(index=index,
                                                   embedding=embeddings,
                                                   text_key=engine.text_key,
                                                   pinecone_api_key=engine.apikey.get_secret_value(),
                                                   index_name=engine.index_name
                                                   )

        else:
            logger.debug(f'Create index {engine.index_name} and embeddings ...')

            if engine.type == "serverless": #os.environ.get("PINECONE_TYPE") == "serverless":
                await pc.create_index(engine.index_name,   # const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric=engine.metric,
                                spec=pinecone.ServerlessSpec(cloud="aws",
                                                             region="us-west-2"
                                                             )
                                )
            else:
                await pc.create_index(engine.index_name, #const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric=engine.metric,
                                spec=pinecone.PodSpec(pod_type="p1",
                                                      pods=1,
                                                      environment="us-west4-gpc"
                                                      )
                                )
            while not (await pc.describe_index(engine.index_name)).status["ready"]:
                await asyncio.sleep(1)

            host = (await pc.describe_index(engine.index_name)).host
            index = pc.IndexAsyncio(name=engine.index_name, host=host)

            vector_store = PineconeVectorStore(index=index,
                                               embedding=embeddings,
                                               text_key=engine.text_key,
                                               pinecone_api_key=engine.apikey.get_secret_value(),
                                               index_name=engine.index_name
                                               )



        return vector_store

    @staticmethod
    async def create_index_sync(engine, embeddings, emb_dimension) -> PineconeVectorStore:
        """
        Questo metodo usa Pinecone Sync. Attualmente non usato
        :param engine:
        :param embeddings:
        :param emb_dimension:
        :return:
        """
        # Cache del client persistente
        client_cache_key = f"persistent_client_{str(engine.apikey)[:20]}"
        from tilellm.store.pinecone.pinecone_persistent_client import PersistentPineconeClient
        async def client_creator():
            return PersistentPineconeClient(engine.apikey.get_secret_value())

        persistent_client = await TimedCache.async_get(
            object_type="persistent_pinecone_client",
            key=client_cache_key,
            constructor=client_creator,
        )



            # Verifica se il client deve essere ricreato
        if persistent_client.is_expired():
            print("=========================> MORTO")
            await persistent_client.close()
            await TimedCache.async_remove("persistent_pinecone_client", client_cache_key)
            persistent_client = await TimedCache.async_get(
                object_type="persistent_pinecone_client",
                key=client_cache_key,
                constructor=client_creator
            )

        # Cache del vector store
        cache_key = (
            str(engine.apikey)[:20],
            engine.index_name,
            engine.type
        )

        async def vector_store_creator():
            logger.info(f"Creazione nuovo vector_store per index: {engine.index_name}")

            # Ottieni l'indice (creandolo se necessario)
            index = await persistent_client.get_or_create_index(engine, emb_dimension)

            return PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key=engine.text_key,
                pinecone_api_key=engine.apikey.get_secret_value(),
                index_name=engine.index_name
            )


        cached_vector_store = await TimedCache.async_get(
            object_type="vector_store",
            key=cache_key,
            constructor=vector_store_creator
        )

        logger.debug(f"Vector store ottenuto/creato per index: {engine.index_name}")
        return cached_vector_store

    @staticmethod
    async def create_index_cache_wrapper(engine, embeddings, emb_dimension, embedding_config_key=None, cache_suffix=None) -> CachedVectorStore:
        cache_key = (
            _hash_api_key(engine.apikey.get_secret_value()) ,
            engine.index_name,
            engine.type,
            embedding_config_key if embedding_config_key is not None else "default"
        )
        if cache_suffix is not None:
            cache_key = cache_key + (cache_suffix,)

        async def _wrapper_creator():
            return await _create_vector_store_instance(engine, embeddings, emb_dimension)

        wrapper = await TimedCache.async_get(
            object_type="vector_store_wrapper",
            key=cache_key,
            constructor=_wrapper_creator
        )

        return wrapper

    async def create_index(self, engine, embeddings, emb_dimension, embedding_config_key=None, cache_suffix=None) -> PineconeVectorStore:
        cached_vs_wrapper = await self.create_index_cache_wrapper(
            engine, embeddings, emb_dimension, embedding_config_key, cache_suffix
        )
        return await cached_vs_wrapper.get_vector_store()


async def _create_vector_store_instance(engine, embeddings, emb_dimension) -> CachedVectorStore:
    logger.info(f"Creating cached vector store wrapper for index: {engine.index_name}")
    cached_vs = CachedVectorStore(engine, embeddings, emb_dimension)
    # opzionale: warm-up client/host
    await cached_vs._ensure_client_and_host()
    return cached_vs






