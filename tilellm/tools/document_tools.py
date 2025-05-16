import time

import requests
import logging
import asyncio


from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import AsyncChromiumLoader


from playwright.async_api import async_playwright

from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document
from playwright.sync_api import sync_playwright


logger = logging.getLogger(__name__)


# "https://help.tiledesk.com/mychatbots/articles/il-pnrr-per-la-ricerca-e-linnovazione/"
async def get_content_by_url(url: str, scrape_type: int,  **kwargs) -> list[Document]:
    """
    Get content by url! parse html page and extract content.
    If scrape_type=0 Unstructured analyze the page and extract some useful information about page, like UL, Title etc.
    If scrape_type=1, extract all the content.
    If scape_type=2 is used playwright.
    If scape_type=3 is used AsyncChromiumLoader and the html is transformed in text
    If scape_type=4 is used AsyncChromiumLoader and BS4 in order to select the html element to extract
    :param url: str representing url
    :param scrape_type: 0|1|2!3!4
    :return: list[Document]
    """
    try:
        urls = [url]
        if scrape_type == 0:
            loader = UnstructuredURLLoader(
                urls=urls, mode="elements", strategy="fast", continue_on_failure=False,
                headers={'user-agent': 'Mozilla/5.0'}
            )
            docs = await loader.aload()

        elif scrape_type == 1:
            loader = UnstructuredURLLoader(
                urls=urls, mode="single", continue_on_failure=False,
                headers={'user-agent': 'Mozilla/5.0'}
            )
            docs = await loader.aload()
        elif scrape_type == 2:


            params_type_4 = kwargs.get("parameters_scrape_type_4")
            docs= await scrape_page(url, params_type_4)
            #loop = asyncio.new_event_loop()
            #queue = Queue()
            #scraping_thread = threading.Thread(target=run_scraping_in_thread, args=(loop, queue, url, params_type_4))
            ##scraping_thread.start()
            #scraping_thread.join()  # This will wait for the thread to finish
            #if not queue.empty():
            #    docs = queue.get()
            #else:
            #    docs =[]

        elif scrape_type == 3:
            loader = AsyncChromiumLoader(urls=urls, user_agent='Mozilla/5.0')
            docs = await loader.aload()
            from langchain_community.document_transformers import Html2TextTransformer
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            docs = docs_transformed
        else:
            params_type_4 = kwargs.get("parameters_scrape_type_4")
            loader = AsyncChromiumLoader(urls=urls, user_agent='Mozilla/5.0')
            docs = await loader.aload()

            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(docs,
                                                                  tags_to_extract=params_type_4.tags_to_extract,
                                                                  unwanted_tags=params_type_4.unwanted_tags,
                                                                  unwanted_classnames=params_type_4.unwanted_classnames,
                                                                  remove_lines=params_type_4.remove_lines,
                                                                  remove_comments=params_type_4.remove_comments
                                                                  )
            docs = docs_transformed
            # print(f"=== DOCS BS4 {docs}")

        for doc in docs:
            doc.metadata = clean_metadata(doc.metadata)

        # from pprint import pprint
        # pprint(docs)

        return docs
    except Exception as ex:
        raise ex

def run_scraping_in_thread(loop, queue, *args, **kwargs):
    asyncio.set_event_loop(loop)
    docs = loop.run_until_complete(scrape_page(*args, **kwargs))
    queue.put(docs)

async def scrape_page(url, params_type_4, time_sleep=2):
    logger.info("Starting scraping...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent="Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36",
                                      java_script_enabled=True)

        await page.goto(url=url)

        await page.wait_for_load_state()
        time.sleep(params_type_4.time_sleep)
        results = await page.content()
        await browser.close()

        metadata = {"source": url}
        doc = Document(page_content=results, metadata=metadata)
        docs = [doc]
        #logger.info(docs)
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs,
            tags_to_extract=params_type_4.tags_to_extract,
            unwanted_tags=params_type_4.unwanted_tags,
            unwanted_classnames=params_type_4.unwanted_classnames,
            remove_lines=params_type_4.remove_lines,
            remove_comments=params_type_4.remove_comments
        )
        docs = docs_transformed
        #for doc in docs:
        #    doc.metadata = clean_metadata(doc.metadata)
        return docs

def scrape_page_new(url, params_type_4):
    logger.info("Starting scraping...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36",
                                      java_script_enabled=True)

        page.goto(url=url, wait_until="load")
        results = page.content()
        browser.close()

        metadata = {"source": url}
        doc = Document(page_content=results, metadata=metadata)
        docs = [doc]

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs,
            tags_to_extract=params_type_4.tags_to_extract,
            unwanted_tags=params_type_4.unwanted_tags,
            unwanted_classnames=params_type_4.unwanted_classnames,
            remove_lines=params_type_4.remove_lines,
            remove_comments=params_type_4.remove_comments
        )
        docs = docs_transformed
        for doc in docs:
            doc.metadata = clean_metadata(doc.metadata)
        return docs



def load_document(url: str, type_source: str):
    # import os
    # name, extension = os.path.splitext(file)

    if type_source == 'pdf':
        from langchain_community.document_loaders import (PyPDFLoader ,
                                                          #UnstructuredPDFLoader,
                                                          #OnlinePDFLoader
                                                          )
        logger.info(f'Loading {url}')
        """
        import requests
        import tempfile
        
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Scrivi il contenuto scaricato nel file temporaneo
            temp_file.write(response.content)

            # Ottieni il percorso del file temporaneo
            file_path = temp_file.name
        #
        loader = UnstructuredPDFLoader(file_path=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            model_name="yolox")
        """
        loader = PyPDFLoader(url)
    elif type_source == 'docx':
        from langchain_community.document_loaders import Docx2txtLoader
        logger.info(f'Loading {url}')
        loader = Docx2txtLoader(url)
    elif type_source == 'txt':
        from langchain_community.document_loaders import TextLoader
        logger.info(f'Loading {url}')
        loader = TextLoader(url)
    else:
        logger.info('Document format is not supported!')
        return None

    data = loader.load()
    # from pprint import pprint
    # pprint(data)
    return data


def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def get_content_by_url_with_bs(url: str):
    html = requests.get(url)
    # urls = [url]
    # Load HTML
    # loader = await AsyncChromiumLoader(urls)
    # html = loader.load()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html.content, 'html.parser')

    # Estrai tutte le sezioni h1
    h1_tags = soup.find_all('h1')

    testi = []
    for index, h1_tag in enumerate(h1_tags):
        # Trova il tag <table> successivo al tag <h1>
        next_a = h1_tag.find_next_sibling('a')
        next_desc = h1_tag.find_next_sibling('h2')


        next_table = h1_tag.find_next_sibling('table')

        # Se esiste, estrai le righe (tr) all'interno della tabella
        testo_tabella =""
        if next_table:
            rows = next_table.find_all('tr')
            # Stampa il contenuto delle righe
            for row in rows:
                # Estrai i td
                tds = row.find_all('td')
                # Se ci sono almeno due td
                if len(tds) >= 2:
                    # Stampa il testo del primo td, i due punti e il testo del secondo td
                    testo_tabella+=f"  {tds[0].get_text(strip=True)}: {tds[1].get_text(strip=True)}"

        testo_doc = f"Product: {h1_tag.text}, URL: {next_a['href']} description: {next_desc.text}.  Measurements: {testo_tabella}"
        testi.append(testo_doc)

        # Aggiungi una riga vuota tra i segmenti
        # if index < len(h1_tags) - 1:
        #    print()  # Stampa una riga vuota tra i segmenti


    return testi


def is_valid_value(value):
    if isinstance(value, (str, int, float, bool)):
        return True
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        return True
    return False


def clean_metadata(dictionary):
    return {k: v for k, v in dictionary.items() if is_valid_value(v)}






