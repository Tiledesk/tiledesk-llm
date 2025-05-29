from abc import ABC, abstractmethod

from langchain_core.vectorstores import VectorStore

from tilellm.models.item_model import RepositoryNamespace, Engine, RepositoryItems, RepositoryNamespaceResult, \
    RepositoryDescNamespaceResult, QuestionAnswer


class VectorStoreRepository(ABC):

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
    async def get_all_obj_namespace(self,engine: Engine, namespace: str) -> RepositoryItems:
        """
        Query Vector store to get all object
        :param engine: Engine
        :param namespace:
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


