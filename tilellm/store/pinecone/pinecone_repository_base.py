from abc import abstractmethod

import time
from langchain_pinecone import PineconeVectorStore

from tilellm.models.item_model import (MetadataItem,
                                       RepositoryQueryResult,
                                       RepositoryItems,
                                       IndexingResult,
                                       RepositoryNamespaceResult,
                                       RepositoryItemNamespaceResult,
                                       RepositoryIdSummaryResult,
                                       RepositoryDescNamespaceResult, Engine, RepositoryNamespace

                                       )

from tilellm.shared import const
from langchain_core.vectorstores import VectorStore
import os
from typing import List, Dict

import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryBase:
    @abstractmethod
    async def add_pc_item(self, item):
        pass

    @abstractmethod
    async def add_pc_item_hybrid(self, item):
        pass

    @staticmethod
    async def delete_pc_namespace(namespace_to_delete: RepositoryNamespace):
        """
        Delete namespace from Pinecone index
        :param namespace_to_delete:
        :return:
        """
        engine = namespace_to_delete.engine
        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey
            )
            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)
            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            delete_response = index.delete(delete_all=True, namespace=namespace_to_delete.namespace)
        except Exception as ex:

            logger.error(ex)

            raise ex

    @abstractmethod
    async def delete_pc_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        pass

    @staticmethod
    async def delete_pc_chunk_id_namespace(engine: Engine, chunk_id: str, namespace: str):
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
                api_key=engine.apikey
            )
            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)
            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX,)
            delete_response = index.delete(ids=[chunk_id], namespace=namespace)
        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def get_pc_ids_namespace(engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
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
                api_key=engine.apikey
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            describe = index.describe_index_stats()
            # print(describe)
            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1

            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')

            logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")
            batch_size = min([total_vectors, 10000])
            pc_res = index.query(
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

    @staticmethod
    async def pinecone_list_namespaces(engine: Engine) -> RepositoryNamespaceResult:
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            describe = index.describe_index_stats()

            logger.debug(describe)
            namespaces = describe.get("namespaces", {})

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

    @staticmethod
    async def get_pc_all_obj_namespace(engine: Engine, namespace: str) -> RepositoryItems:
        """
        Query Pinecone to get all object
        :param engine: Engine
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            describe = index.describe_index_stats()

            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1

            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')

            logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")

            batch_size = min([total_vectors, 1000])

            pc_res = index.query(
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
                                                    text=None  # su pod content, su Serverless text
                                                    )
                              )
            res = RepositoryItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def get_pc_desc_namespace(engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        """
        Query Pinecone to get all object
        :param engine: Engine
        :param namespace:
        :return: PineconeDescNamespaceResult
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=engine.apikey
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            describe = index.describe_index_stats()

            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1
            description = RepositoryItemNamespaceResult(namespace=namespace, vector_count=0)
            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')
                    description = RepositoryItemNamespaceResult(namespace=namespace, vector_count=total_vectors)

            logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")

            batch_size = min([total_vectors, 10000])

            pc_res = index.query(
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

    @staticmethod
    async def get_pc_sources_namespace(engine: Engine, source: str, namespace: str) -> RepositoryItems:
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
                api_key=engine.apikey
            )

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            describe = index.describe_index_stats()
            logger.debug(describe)
            namespaces = describe.get("namespaces", {})
            total_vectors = 1

            if namespaces:
                if namespace in namespaces.keys():
                    total_vectors = namespaces.get(namespace).get('vector_count')

            logger.debug(f"pinecone total vector in {namespace}: {total_vectors}")
            pc_res = index.query(
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
    async def create_pc_index(engine, embeddings, emb_dimension) -> PineconeVectorStore:
        """
        Create or return existing index
        :param engine:
        :param embeddings:
        :param emb_dimension:
        :return:
        """
        import pinecone



        pc =  pinecone.Pinecone(
            api_key= engine.apikey #const.PINECONE_API_KEY
        )

        #existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if engine.index_name in pc.list_indexes().names(): #const.PINECONE_INDEX in pc.list_indexes().names():
           # logger.debug(engine.index_name) #(const.PINECONE_TEXT_KEY)
            #logger.debug(f'Index {const.PINECONE_INDEX} exists. Loading embeddings ... ')
            logger.debug(f'Index {engine.index_name} exists. Loading embeddings ... ')
            #print(f"================== {engine.index_name}, api {engine.apikey}")
            host = pc.describe_index(engine.index_name).host

            index = pc.Index(name=engine.index_name,
                             host=host)

            vector_store = PineconeVectorStore(index=index,
                                               embedding=embeddings,
                                               text_key=engine.text_key,
                                               pinecone_api_key=engine.apikey,
                                               index_name=engine.index_name)


            #vector_store = PineconeVectorStore.from_existing_index(index_name= engine.index_name, #const.PINECONE_INDEX,
            #                                                       embedding=embeddings,
            #                                                       text_key= engine.text_key #const.PINECONE_TEXT_KEY
            #                                                       )  # text-key nuova versione è text
        else:
            #logger.debug(f'Create index {const.PINECONE_INDEX} and embeddings ...')
            logger.debug(f'Create index {engine.index_name} and embeddings ...')

            if engine.type == "serverless": #os.environ.get("PINECONE_TYPE") == "serverless":
                pc.create_index(engine.index_name,   # const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric=engine.metric,
                                spec=pinecone.ServerlessSpec(cloud="aws",
                                                             region="us-west-2"
                                                             )
                                )
            else:
                pc.create_index(engine.index_name, #const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric=engine.metric,
                                spec=pinecone.PodSpec(pod_type="p1",
                                                      pods=1,
                                                      environment="us-west4-gpc"
                                                      )
                                )
            while not pc.describe_index(engine.index_name).status["ready"]:
                time.sleep(1)

            host = pc.describe_index(engine.index_name).host
            index = pc.Index(name=engine.index_name, host=host)

            vector_store = PineconeVectorStore(index=index,
                                               embedding=embeddings,
                                               text_key=engine.text_key,
                                               pinecone_api_key=engine.apikey,
                                               index_name=engine.index_name)



        return vector_store

    @staticmethod
    def chunk_data(data, chunk_size=256, chunk_overlap=10):
        """
        Chunk document in small pieces
        :param data:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """

        from langchain.text_splitter import RecursiveCharacterTextSplitter
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
            from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    def get_embeddings_dimension(embedding):
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
