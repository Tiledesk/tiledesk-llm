
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredURLLoader

#"https://help.tiledesk.com/mychatbots/articles/il-pnrr-per-la-ricerca-e-linnovazione/"
def get_content_by_url(url:str):
    urls =[url]
    loader = UnstructuredURLLoader(
        urls=urls, mode="elements", strategy="fast",
    )
    docs = loader.load()

    #from pprint import pprint
    #pprint(docs)

    return docs