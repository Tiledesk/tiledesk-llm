from abc import abstractmethod
from tilellm.models.item_model import (MetadataItem,
                                       PineconeQueryResult,
                                       PineconeItems,
                                       PineconeIndexingResult
                                       )
from tilellm.tools.document_tool_simple import (get_content_by_url,
                                                get_content_by_url_with_bs,
                                                load_document,
                                                load_from_wikipedia
                                                )

from tilellm.shared import const
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import uuid
import os

import logging

logger = logging.getLogger(__name__)


class PineconeRepositoryBase:
    @abstractmethod
    async def add_pc_item(self, item):
        pass

    @staticmethod
    async def delete_pc_namespace(namespace: str):
        """
        Delete namespace from Pinecone index
        :param namespace:
        :return:
        """
        import pinecone
        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )
            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)
            # vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
            delete_response = index.delete(delete_all=True, namespace=namespace)
        except Exception as ex:

            logger.error(ex)

            raise ex

    @abstractmethod
    async def delete_pc_ids_namespace(self, metadata_id: str, namespace: str):
        pass

    @staticmethod
    async def get_pc_ids_namespace( metadata_id: str, namespace: str):
        """
        Get from Pinecone all items from namespace given document id
        :param metadata_id:
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)

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
                result.append(PineconeQueryResult(id=obj.get('id', ""),
                                                  metadata_id=obj.get('metadata').get('id'),
                                                  metadata_source=obj.get('metadata').get('source'),
                                                  metadata_type=obj.get('metadata').get('type'),
                                                  date=obj.get('metadata').get('date', 'Date not defined'),
                                                  text=obj.get('metadata').get(const.PINECONE_TEXT_KEY)
                                                  # su pod content, su Serverless text
                                                  )
                              )
            res = PineconeItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def pinecone_list_namespaces():
        import pinecone
        from tilellm.models.item_model import PineconeNamespaceResult, PineconeItemNamespaceResult

        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)

            describe = index.describe_index_stats()

            logger.debug(describe)
            namespaces = describe.get("namespaces", {})

            results = []

            for namespace in namespaces.keys():
                total_vectors = namespaces.get(namespace).get('vector_count')
                pc_item_namespace = PineconeItemNamespaceResult(namespace=namespace, vector_count=total_vectors)
                results.append(pc_item_namespace)
                logger.debug(f"{namespace}, {total_vectors}")

            logger.debug(f"pinecone total vector in {results}")

            return PineconeNamespaceResult(namespaces=results)

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def get_pc_all_obj_namespace(namespace: str):
        """
        Query Pinecone to get all object
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)

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
                result.append(PineconeQueryResult(id=obj.get('id', ""),
                                                  metadata_id=obj.get('metadata').get('id'),
                                                  metadata_source=obj.get('metadata').get('source'),
                                                  metadata_type=obj.get('metadata').get('type'),
                                                  date=obj.get('metadata').get('date', 'Date not defined'),
                                                  text=None  # su pod content, su Serverless text
                                                  )
                              )
            res = PineconeItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def get_pc_sources_namespace(source: str, namespace: str):
        """
        Get from Pinecone all items from namespace given source
        :param source:
        :param namespace:
        :return:
        """
        import pinecone

        try:
            pc = pinecone.Pinecone(
                api_key=const.PINECONE_API_KEY
            )

            host = pc.describe_index(const.PINECONE_INDEX).host
            index = pc.Index(name=const.PINECONE_INDEX, host=host)

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
                result.append(PineconeQueryResult(id=obj.get('id'),
                                                  metadata_id=obj.get('metadata').get('id'),
                                                  metadata_source=obj.get('metadata').get('source'),
                                                  metadata_type=obj.get('metadata').get('type'),
                                                  date=obj.get('metadata').get('date', 'Date not defined'),
                                                  text=obj.get('metadata').get(const.PINECONE_TEXT_KEY)
                                                  # su pod content, su Serverless text
                                                  )
                              )
            res = PineconeItems(matches=result)
            logger.debug(res)
            return res

        except Exception as ex:

            logger.error(ex)

            raise ex

    @staticmethod
    async def create_pc_index(embeddings, emb_dimension) -> VectorStore:
        """
        Create or return existing index
        :param embeddings:
        :param emb_dimension:
        :return:
        """
        import pinecone

        from langchain_community.vectorstores import Pinecone

        pc = pinecone.Pinecone(
            api_key=const.PINECONE_API_KEY
        )

        if const.PINECONE_INDEX in pc.list_indexes().names():
            logger.debug(const.PINECONE_TEXT_KEY)
            logger.debug(f'Index {const.PINECONE_INDEX} exists. Loading embeddings ... ')
            vector_store = Pinecone.from_existing_index(index_name=const.PINECONE_INDEX,
                                                        embedding=embeddings,
                                                        text_key=const.PINECONE_TEXT_KEY
                                                        )  # text-key nuova versione è text
        else:
            logger.debug(f'Create index {const.PINECONE_INDEX} and embeddings ...')

            if os.environ.get("PINECONE_TYPE") == "serverless":
                pc.create_index(const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric='cosine',
                                spec=pinecone.ServerlessSpec(cloud="aws",
                                                             region="us-west-2"
                                                             )
                                )
            else:
                pc.create_index(const.PINECONE_INDEX,
                                dimension=emb_dimension,
                                metric='cosine',
                                spec=pinecone.PodSpec(pod_type="p1",
                                                      pods=1,
                                                      environment="us-west4-gpc"
                                                      )
                                )

            vector_store = Pinecone.from_existing_index(index_name=const.PINECONE_INDEX,
                                                        embedding=embeddings,
                                                        text_key=const.PINECONE_TEXT_KEY
                                                        )

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
