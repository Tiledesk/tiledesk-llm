import time
import uuid
import logging
import asyncio
from datetime import datetime

import requests

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    AsyncChromiumLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain_core.documents import Document
from nltk.help import brown_tagset
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

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
    urls = [url]
    params_type_4 = kwargs.get("parameters_scrape_type_4")

    try:
        if scrape_type == 0:
            return await handle_unstructured_loader(
                urls,
                mode="elements",
                strategy="fast"
            )

        elif scrape_type == 1:
            return await handle_unstructured_loader(
                urls,
                mode="single"
            )
        elif scrape_type == 2:
            return await handle_playwright_scrape(url, params_type_4)

        elif scrape_type == 3:
            return await handle_chromium_loader(
                urls,
                transformer=Html2TextTransformer(),
                params=params_type_4
            )
        elif scrape_type == 5:
            return await robust_fallback(url, scrape_type, params_type_4)
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
        logger.error(f"Errore nel metodo principale ({scrape_type}): {str(e)}")
        return await robust_fallback(url, scrape_type, params_type_4)


async def handle_unstructured_loader(urls: list, mode: str, strategy: str = None) -> list[Document]:
    """Gestisce il caricamento con UnstructuredURLLoader"""
    loader_args = {
        "urls": urls,
        "continue_on_failure": False,
        "headers": {'user-agent': 'Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36'}
    }

    if strategy:
        loader_args["strategy"] = strategy
        loader_args["mode"] = mode
    else:
        loader_args["mode"] = mode

    loader = UnstructuredURLLoader(**loader_args)
    docs = await loader.aload()
    return clean_documents_metadata(docs)

async def handle_playwright_scrape(url: str, params: object) -> list[Document]:
    """Gestisce lo scraping con Playwright"""
    docs = await scrape_page(url, params)
    return clean_documents_metadata(docs)


async def handle_chromium_loader(
        urls: list,
        transformer: object = None,
        params: object = None,
        transform_kwargs: dict = None
) -> list[Document]:
    """Gestisce AsyncChromiumLoader con trasformazione opzionale"""
    loader = AsyncChromiumLoader(
        urls=urls,
        user_agent='Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36'
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


async def robust_fallback(url: str, scrape_type: int, params: object) -> list[Document]:
    """Meccanismo di fallback a più livelli"""
    logger.warning(f"Attivazione fallback per URL: {url}")

    try:
        # Primo fallback: metodo alternativo sincrono
        logger.info("Tentativo fallback 1: scrape asincrono")
        return clean_documents_metadata([Document(
            page_content=await fallback_scrape(url),
            metadata={"source": url}
        )])
    except Exception as e:
        logger.error(f"Fallback 1 fallito: {str(e)}")

    try:
        # Secondo fallback: Playwright diretto
        logger.info("Tentativo fallback 2: Playwright")
        return await handle_playwright_scrape(url, params)
    except Exception as e:
        logger.error(f"Fallback 2 fallito: {str(e)}")

    try:
        # Terzo fallback: Chromium con timeout aumentato
        logger.info("Tentativo fallback 3: Chromium rinforzato")
        loader = AsyncChromiumLoader(
            urls=[url],
            user_agent='Mozilla/5.0 AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36',
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ]
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

def run_scraping_in_thread(loop, queue, *args, **kwargs):
    asyncio.set_event_loop(loop)
    docs = loop.run_until_complete(scrape_page(*args, **kwargs))
    queue.put(docs)

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

            try:
                # 1. Navigazione iniziale. 'load' è un buon punto di partenza.
                logger.debug("Navigating to page and waiting for 'load' state...")
                await page.goto(url=url, wait_until="load", timeout=60000)

                # 2. CICLO DI ATTESA DINAMICA (sostituisce il time.sleep)
                logger.debug("Waiting for DOM to stabilize...")
                previous_html_size = 0
                # Eseguiamo il controllo per un massimo di 5 volte (es. 5 * 2 secondi = 10 secondi max)
                # per evitare loop infiniti su pagine che cambiano sempre (es. con un timer).
                for _ in range(5):
                    await page.wait_for_timeout(2000)  # Attendi 2 secondi tra un controllo e l'altro

                    current_html_size = len(await page.content())

                    # Se la dimensione non è cambiata, la pagina è stabile.
                    if current_html_size == previous_html_size:
                        logger.debug("DOM is stable. Proceeding.")
                        break

                    # Altrimenti, aggiorna la dimensione e continua a controllare.
                    previous_html_size = current_html_size
                    logger.debug(f"DOM is still changing... current size: {current_html_size}")
                else:  # Questo `else` si attiva solo se il `for` loop finisce senza `break`
                    logger.warning("DOM did not stabilize after several checks. Proceeding anyway.")

            except PlaywrightTimeoutError:
                logger.warning(f"Timeout reached while loading {url}. Proceeding with captured content.")

            #time.sleep(params_type_4.time_sleep)

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

def scrape_page_new(url, params_type_4):
    logger.info("Starting scraping...")
    with async_playwright() as p:
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


async def fallback_scrape(url: str) -> str | None:
    from requests_html import AsyncHTMLSession
    from requests.exceptions import RequestException
    # Pyppeteer può lanciare un suo TimeoutError, è bene gestirlo
    from pyppeteer.errors import TimeoutError as PyppeteerTimeoutError

    """
    Esegue lo scraping di un URL renderizzando il JavaScript.
    Gestisce correttamente la sessione, la chiusura delle risorse e gli errori.

    Args:
        url: L'URL della pagina da analizzare.

    Returns:
        Il testo HTML renderizzato della pagina, o None in caso di errore.
    """
    logger.info(f"Avvio fallback scrape per l'URL: {url}")
    session = AsyncHTMLSession()
    try:
        # 1. 'await' è necessario per eseguire la richiesta asincrona
        response = await session.get(url, timeout=30)

        # 2. Controlla se la richiesta ha avuto successo (status code 2xx)
        response.raise_for_status()

        # 'arender' esegue il browser headless (Chromium)
        await response.html.arender(timeout=60)

        return response.html.text

    except (RequestException, PyppeteerTimeoutError) as e:
        logger.error(f"Errore durante lo scraping di {url}: {e}")
        return None  # Restituisce None in caso di fallimento

    finally:
        # 3. È fondamentale chiudere la sessione per terminare il processo del browser
        await session.close()

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


async def fetch_documents(type_source, source, scrape_type, parameters_scrape_type_4):
    if type_source in ['url', 'txt']:
        return await get_content_by_url(source,
                                        scrape_type,
                                        parameters_scrape_type_4=parameters_scrape_type_4)
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
