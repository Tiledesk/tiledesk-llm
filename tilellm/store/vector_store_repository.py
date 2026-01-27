import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any

from tilellm.models.schemas import (RepositoryNamespace,
                                    RepositoryItems,
                                    RepositoryNamespaceResult,
                                    RepositoryDescNamespaceResult
                                    )
from tilellm.models import (Engine,
                            QuestionAnswer
                            )
from tilellm.models.llm import TEIConfig
from tilellm.tools.document_tools import get_content_by_url, load_document


logger = logging.getLogger(__name__)

class VectorStoreIndexingError(Exception):
    """Eccezione personalizzata per errori nel vector store"""
    def __init__(self, message: dict):
        self.message = message
        super().__init__(message)

class VectorStoreRepository(ABC):

    sparse_enabled=False

    def build_filter(self, namespace: str, filter_dict: Optional[Dict] = None) -> Any:
        """
        Build store-specific filter from namespace and optional Pinecone-style filter dict.
        Default implementation returns the filter_dict unchanged (suitable for Pinecone).
        Subclasses should override to convert to store-specific filter format.
        """
        # For Pinecone, namespace is separate parameter, filter_dict is Pinecone filter
        # Return filter_dict as-is; namespace will be handled separately
        return filter_dict

    @abstractmethod
    async def add_item(self, item):
        """
        Add item to Vector Store
        :param item:
        :return:
        """
        pass

    @abstractmethod
    async def add_item_hybrid(self, item):
        """
        Add Item to vector store. Item use dense and sparse vector
        :param item:
        :return:
        """
        pass

    @abstractmethod
    async def get_chunks_from_repo(self,question:QuestionAnswer):
        """
        Return top_k chunks form vector store.
        :param question: QuestionAnswer
        :return: list of chunks
        """
        pass


    @abstractmethod
    async def delete_namespace(self, namespace_to_delete: RepositoryNamespace):
        """
        Delete namespace from Repository index
        :param namespace_to_delete:
        :return:
        """
        pass

    @abstractmethod
    async def delete_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str):
        """
        Delete item from namespace/collection identified by metadata.id
        :param engine:
        :param metadata_id:
        :param namespace:
        :return:
        """
        pass

    @abstractmethod
    async def delete_chunk_id_namespace(self, engine: Engine, chunk_id: str, namespace: str):
        """
        delete chunk from vector store
        :param engine: Engine
        :param chunk_id:
        :param namespace:
        :return:
        """
        pass

    @abstractmethod
    async def get_ids_namespace(self, engine: Engine, metadata_id: str, namespace: str) -> RepositoryItems:
        """
        Get from namespace/collections all items from namespace given document id
        :param engine: Engine
        :param metadata_id:
        :param namespace:
        :return:
        """
        pass


    @abstractmethod
    async def list_namespaces(self,engine: Engine) -> RepositoryNamespaceResult:
        """
        Get list of all namespaces/collections from vector store
        :param engine:
        :return:
        """
        pass

    @abstractmethod
    async def get_all_obj_namespace(self,engine: Engine, namespace: str, with_text:bool) -> RepositoryItems:
        """
        Query Vector store to get all object
        :param engine: Engine
        :param namespace:
        :param with_text:
        :return:
        """
        pass

    @abstractmethod
    async def get_desc_namespace(self, engine: Engine, namespace: str) -> RepositoryDescNamespaceResult:
        """
        Query VEctore store to get all object
        :param engine: Engine
        :param namespace:
        :return: PineconeDescNamespaceResult
        """
        pass

    @abstractmethod
    async def get_sources_namespace(self, engine: Engine, source: str, namespace: str) -> RepositoryItems:
        """
        Get from Vector store all items from namespace given source
        :param engine: Engine
        :param source:
        :param namespace:
        :return:
        """
        pass

    @abstractmethod
    async def aadd_documents(self, engine: Engine, documents: list, namespace: str, embedding_model: any, sparse_encoder: Union[str, TEIConfig, None] = "splade", **kwargs):
        """
        Deletes all documents in a namespace and adds the new ones.
        Handles dense and sparse vectors for hybrid search.
        :param engine: The vector store engine configuration.
        :param documents: A list of LangChain Document objects to add.
        :param namespace: The target namespace.
        :param embedding_model: The model to generate dense embeddings.
        :param sparse_encoder: The model to generate sparse vectors.
        :return:
        """
        pass

    @abstractmethod
    async def initialize_embeddings_and_index(self, question_answer, llm_embeddings, emb_dimension=None, embedding_config_key=None, cache_suffix=None):
        pass

    @abstractmethod
    async def perform_hybrid_search(self, question_answer, index, dense_vector, sparse_vector):
        pass

    @abstractmethod
    async def search_community_report(self, question_answer, index, dense_vector, sparse_vector):
        pass

    @staticmethod
    async def fetch_documents_old(type_source, source, scrape_type, parameters_scrape_type_4,browser_headers):
        if type_source in ['url', 'txt']:
            return await get_content_by_url(source,
                                            scrape_type,
                                            parameters_scrape_type_4=parameters_scrape_type_4,
                                            browser_headers=browser_headers)

        return load_document(source, type_source)

    @staticmethod
    async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4, browser_headers):
        if type_source in ['url', 'txt', 'md']:
            documents = await get_content_by_url(source,
                                                 scrape_type,
                                                 parameters_scrape_type_4=parameters_scrape_type_4,
                                                 browser_headers=browser_headers)
        else:
            documents = load_document(source, type_source)

        # Verifica che i documenti siano validi
        if not documents:
            raise ValueError(f"No documents retrieved from the source: {source} (source type: {type_source})")

        # Verifica che ci sia almeno un documento con contenuto non vuoto
        has_content = False
        for doc in documents:
            if doc and doc.page_content and doc.page_content.strip():
                has_content = True
                break

        if not has_content:
            raise ValueError(f"Documents retrieved but source content is empty: {source} (source type: {type_source})")

        return documents

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