from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader
from tilellm.models.item_model import MetadataItem, PineconeQueryResult, PineconeItems
from tilellm.tools.document_tool_simple import get_content_by_url
from tilellm.shared import const
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import os


import logging

logger = logging.getLogger(__name__)


def add_pc_item(item):
        
        logger.info(item)
        id = item.id
        source = item.source
        type_source = item.type
        content = item.content
        gptkey = item.gptkey
        embedding = item.embedding
        namespace = item.namespace
        try:
            delete_pc_ids_namespace(id=id,namespace=namespace)
        except:
            pass

        emb_dimension = get_embeddings_dimension(embedding)

        oai_embeddings = OpenAIEmbeddings(api_key=gptkey, model=embedding) #default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536 
        vector_store = create_pc_index(embeddings=oai_embeddings, emb_dimension= emb_dimension)

        chuncks=[]
        total_tokens=0
        cost=0

        if type_source=='url':
            documents = get_content_by_url(source)
            for document in documents:
                document.metadata["id"]=id
                document.metadata["source"] = source
                document.metadata["type"] = type_source
                document.metadata["embedding"] = embedding


                chuncks.extend(chunk_data(data=[document]))

            a = vector_store.from_documents(chuncks, embedding=oai_embeddings, index_name=const.PINECONE_INDEX,
                                            namespace=namespace, text_key=const.PINECONE_TEXT_KEY)
            total_tokens, cost = calc_embedding_cost(chuncks, embedding)
            logger.info(len(chuncks), total_tokens, cost)
            #from pprint import pprint
            #pprint(documents)

        else:
            metadata = MetadataItem(id=id, source=source, type=type_source, embedding=embedding)
            document = Document(page_content=content, metadata=metadata.dict())
        
            chuncks.extend(chunk_data(data=[document]))
            total_tokens, cost = calc_embedding_cost(chuncks, embedding)
            a = vector_store.from_documents(chuncks, embedding=oai_embeddings, index_name=const.PINECONE_INDEX, namespace=namespace, text_key=const.PINECONE_TEXT_KEY)
        
        return {"id": f"{id}", "chunks": f"{len(chuncks)}", "total_tokens": f"{total_tokens}", "cost": f"{cost:.6f}"}

def delete_pc_namespace(namespace:str):
    import pinecone

    try:
        pc = pinecone.Pinecone(
            api_key=const.PINECONE_API_KEY
        ) 
        #index_host = "https://tilellm-s9kvboq.svc.apw5-4e34-81fa.pinecone.io"#os.environ.get("PINECONE_INDEX_HOST")
        host = pc.describe_index(const.PINECONE_INDEX).host
        index = pc.Index(name=const.PINECONE_INDEX, host=host)
        #vector_store = Pinecone.from_existing_index(const.PINECONE_INDEX, )
        delete_response = index.delete(delete_all = True, namespace=namespace)
        
    except Exception as ex:
        
        logger.error(ex) 
        
        raise ex

def delete_pc_ids_namespace(id:str, namespace:str):
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
            filter={"id": {"$eq": id}},
            namespace=namespace,
            include_values=False,
            include_metadata=False
        )
        matches = pc_res.get('matches')
        ids = [obj.get('id') for obj in matches]

        delete_response = index.delete(
            ids=ids,
            namespace=namespace)
    except Exception as ex:

        logger.error(ex)

        raise ex

def get_pc_ids_namespace(id:str, namespace:str):
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
            filter={"id": {"$eq": id}},
            namespace=namespace,
            include_values=False,
            include_metadata=True
        )
        matches = pc_res.get('matches')
        from pprint import pprint
        pprint(matches)
        #ids = [obj.get('id') for obj in matches]
        print(type(matches[0].get('id')))
        result = []
        for obj in matches:
            result.append(PineconeQueryResult(id = obj.get('id',""),
                                              metadata_id = obj.get('metadata').get('id'),
                                              metadata_source = obj.get('metadata').get('source'),
                                              metadata_type =  obj.get('metadata').get('type'),
                                              text = obj.get('metadata').get(const.PINECONE_TEXT_KEY) #su pod content, su Serverless text
                                              )
                          )
        res = PineconeItems(matches=result)
        logger.debug(res)
        return res

    except Exception as ex:

        logger.error(ex)

        raise ex

def get_pc_sources_namespace(source:str, namespace:str):
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

def create_pc_index(embeddings, emb_dimension):
     
    import pinecone
    from langchain_community.vectorstores import Pinecone

    
    pc = pinecone.Pinecone(
        api_key=const.PINECONE_API_KEY
    )

        
    if const.PINECONE_INDEX in pc.list_indexes().names():
        print(const.PINECONE_TEXT_KEY)
        print(f'Index {const.PINECONE_INDEX} esiste. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name=const.PINECONE_INDEX,
                                                    embedding=embeddings,
                                                    text_key=const.PINECONE_TEXT_KEY
                                                   ) #text-key nuova versione è text
    else:
        print(f'Creazione di index {const.PINECONE_INDEX} e embeddings ...', end='')
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

def chunk_data(data, chunk_size=256,chunk_overlap=10):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def calc_embedding_cost(texts, embedding):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Numero totale di Token: {total_tokens}')
    cost=0
    try:
        if embedding=="text-embedding-3-large":
            cost = total_tokens / 1e6 * 0.13
        elif embedding=="text-embedding-3-small":
            cost = total_tokens / 1e6 * 0.02
        else:
            embedding = "text-embedding-ada-002"
            cost = total_tokens / 1e6 * 0.10
    
    except IndexError:
        embedding = "text-embedding-ada-002"
        cost = total_tokens / 1e6 * 0.10

    print(f'Costo degli Embedding $: {cost:.6f}')
    return total_tokens, cost

def get_embeddings_dimension(embedding):
    emb_dimension = 1536
    try: 
        if embedding=="text-embedding-3-large":
            emb_dimension = 3072
        elif embedding=="text-embedding-3-small":
            emb_dimension = 1536
        else:
            embedding = "text-embedding-ada-002"
            emb_dimension = 1536
    
    except IndexError:
        embedding = "text-embedding-ada-002"
        emb_dimension = 1536
    
    return emb_dimension 
