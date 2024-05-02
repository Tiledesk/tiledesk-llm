from tilellm.models.item_model import (MetadataItem,
                                       PineconeQueryResult,
                                       PineconeItems,
                                       PineconeIndexingResult
                                       )
from tilellm.tools.document_tool_simple import get_content_by_url, get_content_by_url_with_bs
from tilellm.shared import const
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import os

import logging

logger = logging.getLogger(__name__)


async def add_pc_item(item):
    """
        Add items to name
        space into Pinecone index
        :param item:
        :return:
        """
    logger.info(item)
    metadata_id = item.id
    source = item.source
    type_source = item.type
    content = item.content
    gpt_key = item.gptkey
    embedding = item.embedding
    namespace = item.namespace
    try:
        await delete_pc_ids_namespace(metadata_id=metadata_id, namespace=namespace)
    except Exception as ex:
        logger.error(ex)
        pass

    emb_dimension = get_embeddings_dimension(embedding)

    # default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536
    oai_embeddings = OpenAIEmbeddings(api_key=gpt_key, model=embedding)
    vector_store = await create_pc_index(embeddings=oai_embeddings, emb_dimension=emb_dimension)

    chunks = []
    total_tokens = 0
    cost = 0

    if type_source == 'url':
        documents = get_content_by_url(source)
        for document in documents:
            document.metadata["id"] = metadata_id
            document.metadata["source"] = source
            document.metadata["type"] = type_source
            document.metadata["embedding"] = embedding

            for key, value in document.metadata.items():
                if isinstance(value, list) and all(item is None for item in value):
                    document.metadata[key] = [""]
                elif value is None:
                    document.metadata[key] = ""

            chunks.extend(chunk_data(data=[document]))



        # from pprint import pprint
        # pprint(documents)
        logger.debug(documents)

        a = vector_store.from_documents(chunks,
                                        embedding=oai_embeddings,
                                        index_name=const.PINECONE_INDEX,
                                        namespace=namespace,
                                        text_key=const.PINECONE_TEXT_KEY)

        total_tokens, cost = calc_embedding_cost(chunks, embedding)
        logger.info(len(chunks), total_tokens, cost)
        # from pprint import pprint
        # pprint(documents)
    elif type_source == 'urlbs':
        doc_array = get_content_by_url_with_bs(source)
        chunks = list()
        for doc in doc_array:
            metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
            document = Document(page_content=doc, metadata=metadata.dict())
            chunks.append(document)
        # chunks.extend(chunk_data(data=documents))
        total_tokens, cost = calc_embedding_cost(chunks, embedding)
        a = vector_store.from_documents(chunks,
                                        embedding=oai_embeddings,
                                        index_name=const.PINECONE_INDEX,
                                        namespace=namespace,
                                        text_key=const.PINECONE_TEXT_KEY)

    else:
        metadata = MetadataItem(id=metadata_id, source=source, type=type_source, embedding=embedding)
        document = Document(page_content=content, metadata=metadata.dict())

        chunks.extend(chunk_data(data=[document]))
        total_tokens, cost = calc_embedding_cost(chunks, embedding)
        a = vector_store.from_documents(chunks,
                                        embedding=oai_embeddings,
                                        index_name=const.PINECONE_INDEX,
                                        namespace=namespace,
                                        text_key=const.PINECONE_TEXT_KEY)

    pinecone_result = PineconeIndexingResult(id=metadata_id, chunks=len(chunks), total_tokens=total_tokens,
                                             cost=f"{cost:.6f}")
    # {"id": f"{id}", "chunks": f"{len(chunks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}
    return pinecone_result


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


async def delete_pc_ids_namespace(metadata_id: str, namespace: str):
    """
    Delete from pinecone items
    :param metadata_id:
    :param namespace:
    :return:
    """
    import pinecone
    from langchain_community.vectorstores import Pinecone

    try:
        pc = pinecone.Pinecone(
            api_key=const.PINECONE_API_KEY
        )
        # index_host = "https://tilellm-s9kvboq.svc.apw5-4e34-81fa.pinecone.io"#os.environ.get("PINECONE_INDEX_HOST")
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

        logger.debug(total_vectors)
        pc_res = index.query(
            vector=[0] * 1536,  # [0,0,0,0......0]
            top_k=total_vectors,
            filter={"id": {"$eq": metadata_id}},
            namespace=namespace,
            include_values=False,
            include_metadata=False
        )
        matches = pc_res.get('matches')

        ids = [obj.get('id') for obj in matches]
        if not ids:
            raise IndexError(f"Empty list for {metadata_id} and namespace {namespace}")

        index.delete(
            ids=ids,
            namespace=namespace)

    except Exception as ex:

        logger.error(ex)

        raise ex


async def get_pc_ids_namespace(metadata_id: str, namespace: str):
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

        pc_res = index.query(
            vector=[0] * 1536,  # [0,0,0,0......0]
            top_k=total_vectors,
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
                                              text=obj.get('metadata').get(const.PINECONE_TEXT_KEY)
                                              #su pod content, su Serverless text
                                              )
                          )
        res = PineconeItems(matches=result)
        logger.debug(res)
        return res

    except Exception as ex:

        logger.error(ex)

        raise ex


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

        pc_res = index.query(
            vector=[0] * 1536,  # [0,0,0,0......0]
            top_k=total_vectors,
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
                                              text=None  # su pod content, su Serverless text
                                              )
                          )
        res = PineconeItems(matches=result)
        logger.debug(res)
        return res

    except Exception as ex:

        logger.error(ex)

        raise ex


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


async def create_pc_index(embeddings, emb_dimension):
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
        # FIXME Cercare una soluzione megliore per gestire creazione di indici. Anche in produzione si potrebbe pensare di usare serverless...

        if os.environ.get("ENVIRON") == "dev":
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
