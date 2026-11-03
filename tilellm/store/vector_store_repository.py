import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List

from langchain_core.documents import Document

from tilellm.models.schemas import (RepositoryNamespace,
                                    RepositoryItems,
                                    RepositoryNamespaceResult,
                                    RepositoryDescNamespaceResult
                                    )
from tilellm.models import (Engine,
                            QuestionAnswer
                            )
from tilellm.models.llm import TEIConfig
from tilellm.tools.document_tools import get_content_by_url, load_document, handle_regex_custom_chunk


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


    async def get_chunks_by_index(self, engine: Engine, namespace: str, doc_id: str,
                                  chunk_indexes: List[int]) -> List[Document]:
        """
        Only the chunks `chunk_indexes` (metadata.chunk_index) of document `doc_id` —
        a targeted fetch, unlike get_by_doc_id which pages through the whole document.
        Used to re-attach a retrieved chunk's neighbours when a table row was split
        across consecutive chunks. Order of the result is not guaranteed.

        Default: no neighbours available (backends that don't override it behave
        exactly as before).
        """
        return []

    @abstractmethod
    async def get_by_doc_id(self, engine: Engine, namespace:str, doc_id: str):
        """
        Get chunks from doc_id and namespace
        :param engine
        :param namespace:
        :param doc_id:
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

    # ponytail: shared copies of the per-backend chunking helpers. Pinecone/Qdrant/Milvus keep
    # their own (identical) overrides; new backends inherit these. Migrate the others when touched.
    async def prepare_item_chunks(self, item, embedding_obj):
        """fetch -> chunk -> optional situated context for an ItemSingle. Returns (chunks, sc_token_usage)."""
        from tilellm.models import MetadataItem
        from tilellm.modules.ingestion.text_processor import process_auto_detected_text

        if item.type in ['url', 'pdf', 'docx', 'txt', 'md', 'xlsx', 'xls', 'csv']:
            documents = await self.fetch_documents(item.type, item.source, item.scrape_type,
                                                   item.parameters_scrape_type_4, item.browser_headers)
            chunks = await self.chunk_documents(item, documents, embedding_obj)
        elif item.type == 'regex_custom':
            documents = await self.fetch_documents(item.type, item.source, item.scrape_type,
                                                   item.parameters_scrape_type_4, item.browser_headers,
                                                   chunk_regex=item.chunk_regex)
            base = MetadataItem(id=item.id, source=item.source, type=item.type,
                                embedding=str(item.embedding)).model_dump()
            if item.tags:
                base["tags"] = item.tags
            from tilellm.tools.document_tools import _extract_file_name
            base["file_name"] = _extract_file_name(item.source or "")  # as Qdrant/Pinecone
            base.setdefault("page", 1)
            chunks = [Document(page_content=d.page_content, metadata={**d.metadata, **base}) for d in documents]
        else:
            metadata = MetadataItem(id=item.id, source=item.source, type=item.type,
                                    embedding=item.embedding, namespace=item.namespace).model_dump()
            if item.tags:
                metadata["tags"] = item.tags
            chunks = await process_auto_detected_text(
                content=item.content, source=item.source, doc_id=item.id, chunk_size=item.chunk_size,
                chunk_overlap=item.chunk_overlap, metadata=metadata, semantic_chunk=item.semantic_chunk,
                embeddings=embedding_obj)

        sc_token_usage = None
        if getattr(item, 'situated_context', None) and chunks:
            try:
                from tilellm.shared.situated_context import enrich_chunks_with_situated_context, build_llm_from_item
                situated_llm = await build_llm_from_item(item)
                if situated_llm:
                    sc_result = await enrich_chunks_with_situated_context(
                        chunks, situated_llm, profile=item.situated_context.profile,
                        custom_prompt=item.situated_context.custom_prompt,
                        metadata_extraction_prompt=item.situated_context.metadata_extraction_prompt)
                    chunks, sc_token_usage = sc_result.documents, sc_result.token_usage
            except Exception as sc_err:
                logger.warning(f"Situated context enrichment failed, continuing without: {sc_err}")
        return chunks, sc_token_usage

    @staticmethod
    def process_document_metadata(document, metadata):
        import datetime
        document.metadata.update(metadata)
        document.metadata['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
        for key, value in document.metadata.items():
            if isinstance(value, list) and all(v is None for v in value):
                document.metadata[key] = [""]
            elif value is None:
                document.metadata[key] = ""
        return document

    async def chunk_documents(self, item, documents, embeddings):
        from tilellm.models.llm import TableOptions
        from tilellm.modules.ingestion.table_chunker import split_table_document
        from tilellm.tools.document_tools import _extract_file_name
        table_opts = getattr(item, 'table_options', None) or TableOptions()

        chunks = []
        for document in documents:
            document.metadata["id"] = item.id
            document.metadata["source"] = item.source
            document.metadata["type"] = item.type
            document.metadata["embedding"] = str(item.embedding)
            document.metadata["namespace"] = item.namespace
            if item.tags:
                document.metadata["tags"] = item.tags
            if not document.metadata.get("file_name"):
                document.metadata["file_name"] = _extract_file_name(item.source or "")
            if "page" not in document.metadata:
                document.metadata["page"] = 1
            processed = self.process_document_metadata(document, document.metadata)
            if table_opts.enable and processed.metadata.get("element_type") == "table":
                chunks.extend(split_table_document(processed, strategy=table_opts.strategy,
                                                   max_table_chars=table_opts.max_table_chars,
                                                   header_repeat=table_opts.header_repeat))
            else:
                chunks.extend(self.chunk_data_extended(
                    data=[processed], chunk_size=item.chunk_size, chunk_overlap=item.chunk_overlap,
                    semantic=item.semantic_chunk, embeddings=embeddings,
                    breakpoint_threshold_type=item.breakpoint_threshold_type))
        return chunks

    @staticmethod
    def chunk_data_extended(data, chunk_size=256, chunk_overlap=10, **kwargs):
        if kwargs.get('semantic'):
            from langchain_experimental.text_splitter import SemanticChunker
            splitter = SemanticChunker(kwargs['embeddings'], breakpoint_threshold_type=kwargs['breakpoint_threshold_type'])
        else:
            from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(data)

    @staticmethod
    async def fetch_documents_old(type_source, source, scrape_type, parameters_scrape_type_4,browser_headers):
        if type_source in ['url', 'txt']:
            return await get_content_by_url(source,
                                            scrape_type,
                                            parameters_scrape_type_4=parameters_scrape_type_4,
                                            browser_headers=browser_headers)

        return load_document(source, type_source)

    @staticmethod
    async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4, browser_headers,
                              chunk_regex=None, crawl_options=None):
        from tilellm.tools import document_tools
        return await document_tools.fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4,
                                                    browser_headers, chunk_regex=chunk_regex,
                                                    crawl_options=crawl_options)

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