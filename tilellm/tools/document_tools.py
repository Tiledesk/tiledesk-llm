import time
import uuid
import logging
import asyncio
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup, Comment

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    AsyncChromiumLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain_community.document_transformers.beautiful_soup_transformer import get_navigable_strings
from langchain_core.documents import Document
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


# "https://help.tiledesk.com/mychatbots/articles/il-pnrr-per-la-ricerca-e-linnovazione/"
async def get_content_by_url(url: str, scrape_type: int,  **kwargs) -> list[Document]:
    """
    Get content by url! parse html page and extract content.
    If scrape_type=0 Unstructured analyze the page and extract some useful information about page, like UL, Title etc.
    If scrape_type=1, extract all the content.
    If scape_type=2 is used playwright and BS4 in order to select the html element to extract.
    If scape_type=3 is used AsyncChromiumLoader and the html is transformed in text
    If scape_type=4 is used AsyncChromiumLoader and BS4 in order to select the html element to extract
    If scape_type=5 is used playwright and BS4 in order to select the html element to extract and class.
    :param url: str representing url
    :param scrape_type: 0|1|2!3!4
    :return: list[Document]
    """
    urls = [url]
    params_type_4 = kwargs.get("parameters_scrape_type_4")
    browser_headers = kwargs.get("browser_headers")


    try:
        if scrape_type == 0:
            return await handle_unstructured_loader(
                urls,
                mode="elements",
                strategy="fast",
                browser_headers = browser_headers
            )

        elif scrape_type == 1:
            return await handle_unstructured_loader(
                urls,
                mode="single",
                browser_headers = browser_headers
            )
        elif scrape_type == 2:
            return await handle_playwright_scrape(url, params_type_4, browser_headers = browser_headers)
        elif scrape_type == 5:
            return await handle_playwright_scrape_complex(url, params_type_4, browser_headers = browser_headers)
        elif scrape_type == 3:
            return await handle_chromium_loader(
                urls,
                transformer=Html2TextTransformer(),
                params=params_type_4,
                browser_headers=browser_headers
            )
        else:
            return await handle_chromium_loader(
                urls,
                transformer=BeautifulSoupTransformer(),
                params=params_type_4,
                transform_kwargs={
                    "tags_to_extract": params_type_4.tags_to_extract,
                    "unwanted_tags": params_type_4.unwanted_tags,
                    "unwanted_classnames": params_type_4.unwanted_classnames,
                    "remove_lines": params_type_4.remove_lines,
                    "remove_comments": params_type_4.remove_comments
                }
            )
    except Exception as ex:
        logger.error(f"Errore nel metodo principale ({scrape_type}): {str(ex)}")
        return await robust_fallback(url, params_type_4, browser_headers=browser_headers)


async def handle_unstructured_loader(urls: list, mode: str, strategy: str = None, browser_headers = dict) -> list[Document]:
    """Gestisce il caricamento con UnstructuredURLLoader"""
    loader_args = {
        "urls": urls,
        "continue_on_failure": False,
        "headers": browser_headers
    }

    if strategy:
        loader_args["strategy"] = strategy
        loader_args["mode"] = mode
    else:
        loader_args["mode"] = mode
    logger.info(f"loader args for UnstructuredLoader {loader_args}")
    loader = UnstructuredURLLoader(**loader_args)
    docs = await loader.aload()
    return clean_documents_metadata(docs)

async def handle_playwright_scrape(url: str, params: object, browser_headers: dict) -> list[Document]:
    """Gestisce lo scraping con Playwright"""
    docs = await scrape_page(url, params, browser_headers = browser_headers)
    return clean_documents_metadata(docs)

async def handle_playwright_scrape_complex(url: str, params: object,browser_headers: dict) -> list[Document]:
    """Gestisce lo scraping con Playwright"""
    docs = await scrape_page_complex(url, params, browser_headers = browser_headers)
    return clean_documents_metadata(docs)


async def handle_chromium_loader(
        urls: list,
        transformer: object = None,
        params: object = None,
        transform_kwargs: dict = None,
        browser_headers :dict= None
) -> list[Document]:
    """Gestisce AsyncChromiumLoader con trasformazione opzionale"""
    loader = AsyncChromiumLoader(
        urls=urls,
        user_agent=browser_headers["user-agent"]
    )
    docs = await loader.aload()

    # Controllo del contenuto minimo
    if not docs or any(len(doc.page_content.strip()) < 50 for doc in docs):
        raise ValueError("Contenuto insufficiente o vuoto")

    if transformer and transform_kwargs:

        docs = transformer.transform_documents(docs, **transform_kwargs)
    elif transformer:
        docs = transformer.transform_documents(docs)

    return clean_documents_metadata(docs)


async def scrape_page_fallback_selectors(url, browser_headers:dict=None):
    """Fallback con selettori più permissivi"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
                                          args=[
                                              "--disable-blink-features=AutomationControlled",
                                              "--no-sandbox"
                                          ])
        page = await browser.new_page(extra_http_headers=browser_headers,
                                      java_script_enabled=True)

        await page.goto(url=url, wait_until="networkidle", timeout=60000)
        time.sleep(2)  # Tempo ridotto per fallback

        results = await page.content()
        await browser.close()

        # Selettori di fallback più permissivi
        fallback_selectors = get_fallback_selectors()

        transformed_content = custom_html_transform(
            results,
            selectors_to_extract=fallback_selectors,
            unwanted_tags=["script", "style", "nav", "header", "footer", "aside"],
            unwanted_classnames=["advertisement", "ads", "sidebar", "popup"],
            remove_lines=True,
            remove_comments=True
        )

        metadata = {"source": url, "scraping_method": "playwright_fallback"}
        doc = Document(page_content=transformed_content, metadata=metadata)
        return [doc]

def get_fallback_selectors():
    """Restituisce selettori di fallback più permissivi"""
    return [
        # Contenuto principale
        "main", "article", "[role='main']",
        # Div con classi comuni per contenuto
        "div.content", "div.main", "div.post", "div.article",
        "div.entry-content", "div.post-content", "div.article-content",
        # Sezioni generiche
        "section",
        # Paragrafi e testi
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        # Lista e elementi di testo
        "ul", "ol", "li", "blockquote",
        # Fallback finale - quasi tutto tranne elementi indesiderati
        "div:not(.sidebar):not(.advertisement):not(.ads):not(.popup)"
    ]

async def robust_fallback(url: str, params: object = None, browser_headers:dict=None) -> list[Document]:
    """Meccanismo di fallback a più livelli"""
    logger.warning(f"Attivazione fallback per URL: {url}")

    try:
        # Primo fallback: metodo alternativo sincrono
        logger.info("Tentativo fallback 1: scrape asincrono")
        return clean_documents_metadata(await scrape_page_fallback_selectors(url, browser_headers=browser_headers))
    except Exception as e:
        logger.error(f"Fallback 1 fallito: {str(e)}")

    try:
        # Secondo fallback: Playwright diretto
        logger.info("Tentativo fallback 2: Playwright")
        return await handle_playwright_scrape(url, params, browser_headers=browser_headers)
    except Exception as e:
        logger.error(f"Fallback 2 fallito: {str(e)}")

    try:
        # Terzo fallback: Chromium con timeout aumentato
        logger.info("Tentativo fallback 3: Chromium rinforzato")
        loader = AsyncChromiumLoader(
            urls=[url],
            user_agent=browser_headers["user-agent"],
            headless=True
        )
        docs = await loader.aload()
        return clean_documents_metadata(docs)
    except Exception as e:
        logger.error(f"Fallback 3 fallito: {str(e)}")

    # Fallback finale: documento vuoto
    logger.error("Tutti i fallback falliti, restituisco documento vuoto")
    return clean_documents_metadata([Document(
        page_content="",
        metadata={"source": url}
    )])


def clean_documents_metadata(docs: list[Document]) -> list[Document]:
    """Pulisce i metadati per tutti i documenti"""
    for doc in docs:
        doc.metadata = clean_metadata(doc.metadata)
    return docs

async def scrape_page(url, params_type_4, browser_headers:dict, time_sleep=2):
    logger.info("Starting scraping...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True,
                                              args=[
                                                  "--disable-blink-features=AutomationControlled",
                                                  "--no-sandbox"
                                              ]
                                              )
            page = await browser.new_page(extra_http_headers=browser_headers, #user_agent="Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36",
                                          java_script_enabled=True)

            # Registra un listener per l'evento 'request'
            #page.on("request", handle_request)


            await page.goto(url=url, wait_until="networkidle", timeout=120000)

            await page.wait_for_load_state("networkidle")

            time.sleep(params_type_4.time_sleep)

            results = await page.content()
            await browser.close()

            metadata = {"source": url}
            doc = Document(page_content=results, metadata=metadata)
            docs = [doc]

            bs_transformer = BeautifulSoupTransformer()
            return bs_transformer.transform_documents(
                docs,
                tags_to_extract=params_type_4.tags_to_extract,
                unwanted_tags=params_type_4.unwanted_tags,
                unwanted_classnames=params_type_4.unwanted_classnames,
                remove_lines=params_type_4.remove_lines,
                remove_comments=params_type_4.remove_comments
            )

    except Exception as e:
        logger.error(f"Playwright scrape failed: {str(e)}")
        raise


async def scrape_page_complex(url, params_type_4, browser_headers:dict, time_sleep=2):
    logger.info("Starting scraping...")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True,
                                              args=[
                                                  "--disable-blink-features=AutomationControlled",
                                                  "--no-sandbox"
                                              ]
                                              )
            page = await browser.new_page(extra_http_headers=browser_headers, #user_agent="Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36",
                                          java_script_enabled=True)

            page.on("request", handle_request)
            await page.goto(url=url)
            await page.wait_for_load_state()

            time.sleep(params_type_4.time_sleep)

            results = await page.content()
            await browser.close()

            # Processamento personalizzato con BeautifulSoup
            transformed_content = custom_html_transform(
                results,
                selectors_to_extract=params_type_4.tags_to_extract,
                unwanted_tags=getattr(params_type_4, 'unwanted_tags', []),
                unwanted_classnames=getattr(params_type_4, 'unwanted_classnames', []),
                remove_lines=getattr(params_type_4, 'remove_lines', True),
                remove_comments=getattr(params_type_4, 'remove_comments', True)
            )

            metadata = {"source": url}
            doc = Document(page_content=transformed_content, metadata=metadata)
            return [doc]

    except Exception as e:
        logger.error(f"Playwright scrape failed: {str(e)}")
        raise


def custom_html_transform(html_content, selectors_to_extract=None, unwanted_tags=None,
                          unwanted_classnames=None, remove_lines=True, remove_comments=True):
    """
    Trasforma il contenuto HTML usando selettori CSS personalizzati

    Args:
        html_content (str): Contenuto HTML da processare
        selectors_to_extract (list): Lista di selettori CSS (es. ["div.class", "p#id", "section"])
        unwanted_tags (list): Tag da rimuovere
        unwanted_classnames (list): Classi da rimuovere
        remove_lines (bool): Se rimuovere linee vuote
        remove_comments (bool): Se rimuovere commenti HTML
    """
    import re
    if not selectors_to_extract:
        selectors_to_extract = ["*"]  # Seleziona tutto se non specificato

    soup = BeautifulSoup(html_content, 'html.parser')

    # Rimuovi commenti HTML se richiesto
    if remove_comments:
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

    # Rimuovi tag indesiderati
    if unwanted_tags:
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                #element.extract()
                element.decompose()

    # Rimuovi elementi con classi indesiderate
    if unwanted_classnames:
        for class_name in unwanted_classnames:
            for element in soup.find_all(class_=class_name):
                element.decompose()

    # Estrai elementi basandosi sui selettori CSS
    extracted_elements = []

    text_parts: List[str] = []

    # Raccogli tutti gli elementi che corrispondono ai selettori
    elements_to_process = []

    for selector in selectors_to_extract:
        try:
            # Usa select() per selettori CSS (supporta sia tag semplici che complessi)
            elements = soup.select(selector)
            elements_to_process.extend(elements)
        except Exception as e:
            # Se il selettore CSS fallisce, prova come tag semplice (fallback)
            try:
                elements = soup.find_all(selector)
                elements_to_process.extend(elements)
            except Exception:
                # Se anche questo fallisce, continua con il prossimo selettore
                continue

    # Rimuovi duplicati mantenendo l'ordine di apparizione nel DOM
    seen = set()
    unique_elements = []
    for element in elements_to_process:
        # Usa id dell'oggetto per identificare univocamente l'elemento
        element_id = id(element)
        if element_id not in seen:
            seen.add(element_id)
            unique_elements.append(element)

    # Ordina gli elementi in base alla loro posizione nel DOM
    # per mantenere l'ordine naturale del documento
    def get_element_position(element):
        """Calcola la posizione di un elemento nel DOM"""
        position = 0
        for sibling in element.parent.children if element.parent else []:
            if sibling == element:
                break
            position += 1
        return position

    # Ordina per posizione nel DOM se hanno lo stesso genitore
    # altrimenti mantieni l'ordine di scoperta
    unique_elements.sort(key=lambda x: (
        str(x.parent) if x.parent else "",
        get_element_position(x)
    ))

    # Estrai il testo da tutti gli elementi
    for element in unique_elements:
        # Extract all navigable strings recursively from this element.
        text_parts += get_navigable_strings(
            element, remove_comments=remove_comments
        )

        # To avoid duplicate text, remove all descendants from the soup.
        element.decompose()

    result = " ".join(text_parts)

    # Rimuovi linee vuote se richiesto
    if remove_lines:
        result = re.sub(r'\n\s*\n', '\n', result)
        result = result.strip()

    return result

def load_document(url: str, type_source: str):
    # import os
    # name, extension = os.path.splitext(file)

    if type_source == 'pdf':

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
        logger.info(f'Loading {url}')
        loader = Docx2txtLoader(url)
    elif type_source == 'txt':
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


async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4, browser_headers):
    if type_source in ['url', 'txt']:
        return await get_content_by_url(source,
                                        scrape_type,
                                        parameters_scrape_type_4=parameters_scrape_type_4,
                                        browser_headers=browser_headers)
    return load_document(source, type_source)


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


async def handle_request(request):
    """Callback function to handle network requests asynchronously."""
    if request.is_navigation_request() and request.resource_type == "document":
        logger.info(f"URL della richiesta principale: {request.url}")
        logger.info("Header inviati:")
        headers = await request.all_headers()
        for name, value in headers.items():
            logger.info(f"  {name}: {value}")