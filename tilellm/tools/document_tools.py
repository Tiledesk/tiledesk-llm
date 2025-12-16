import time

import logging

from typing import List, Optional, Sequence, Any

import requests
from bs4 import BeautifulSoup, Comment

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain_community.document_transformers.beautiful_soup_transformer import get_navigable_strings
from langchain_core.documents import Document
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


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
    browser_headers: Optional[dict] = kwargs.get("browser_headers")
    
    # Default browser headers if not provided
    if not browser_headers:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    # At this point browser_headers is guaranteed to be a dict
    assert isinstance(browser_headers, dict)
    
    # Validate parameters for scrape_type 4
    if scrape_type == 4 and params_type_4 is None:
        raise ValueError("parameters_scrape_type_4 is required for scrape_type=4")
    
    # Validate parameters for scrape_type 2, 5
    if scrape_type in [2, 5] and params_type_4 is None:
        raise ValueError("parameters_scrape_type_4 is required for scrape_type=2 or 5")


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
        elif scrape_type == 4:
            # params_type_4 is guaranteed to be not None due to earlier validation
            if params_type_4 is None:
                raise ValueError("parameters_scrape_type_4 is required for scrape_type=4")
            assert params_type_4 is not None
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
        else:
            raise ValueError(f"Unsupported scrape_type: {scrape_type}")
    except Exception as ex:
        logger.error(f"Errore nel metodo principale ({scrape_type}): {str(ex)}")
        # Se l'errore è relativo a CAPTCHA, non procedere con fallback
        if "CAPTCHA" in str(ex) or "bloccata" in str(ex) or "troppo breve" in str(ex):
            raise
        return await robust_fallback(url, params_type_4, browser_headers=browser_headers)


async def handle_unstructured_loader(urls: list, mode: str, strategy: Optional[str] = None, browser_headers: Optional[dict] = None) -> list[Document]:
    """Gestisce il caricamento con UnstructuredURLLoader"""
    #print(urls)
    if browser_headers is None or browser_headers is dict:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
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
    try:
        docs = await loader.aload()
        return clean_documents_metadata(docs)
    finally:
        if hasattr(loader, 'close'):
            loader.close()
        elif hasattr(loader, 'aclose'):
            await loader.aclose()

async def handle_playwright_scrape(url: str, params: object, browser_headers: Optional[dict] = None) -> list[Document]:
    """Gestisce lo scraping con Playwright"""
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    docs = await scrape_page(url, params, browser_headers = browser_headers)
    return clean_documents_metadata(docs)

async def handle_playwright_scrape_complex(url: str, params: object, browser_headers: Optional[dict] = None) -> list[Document]:
    """Gestisce lo scraping con Playwright"""
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    docs = await scrape_page_complex(url, params, browser_headers = browser_headers)
    return clean_documents_metadata(docs)


async def handle_chromium_loader(
        urls: list,
        transformer: Optional[Any] = None,
        params: Optional[object] = None,
        transform_kwargs: Optional[dict] = None,
        browser_headers: Optional[dict] = None
) -> list[Document]:
    """Gestisce scraping con Playwright e trasformazione opzionale (sostituisce AsyncChromiumLoader)"""
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    docs = []
    for url in urls:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox"
                ]
            )
            try:
                page = await browser.new_page(
                    extra_http_headers=browser_headers,
                    java_script_enabled=True
                )
                await page.goto(url=url, wait_until="load", timeout=60000)
                html = await page.content()
                metadata = {"source": url}
                doc = Document(page_content=html, metadata=metadata)
                docs.append(doc)
            finally:
                await browser.close()
    
    # Controllo del contenuto minimo
    if not docs or any(len(doc.page_content.strip()) < 50 for doc in docs):
        raise ValueError("Contenuto insufficiente o vuoto")

    if transformer and transform_kwargs:
        docs = transformer.transform_documents(docs, **transform_kwargs)
    elif transformer:
        docs = transformer.transform_documents(docs)

    return clean_documents_metadata(docs)


async def scrape_page_fallback_selectors(url, browser_headers: Optional[dict] = None):
    """Fallback con selettori più permissivi"""
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
                                          args=[
                                              "--disable-blink-features=AutomationControlled",
                                              "--no-sandbox"
                                          ])
        try:
            page = await browser.new_page(extra_http_headers=browser_headers,
                                          java_script_enabled=True)

            await page.goto(url=url, wait_until="networkidle", timeout=60000)
            time.sleep(2)  # Tempo ridotto per fallback

            results = await page.content()
        finally:
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

async def robust_fallback(url: str, params: Optional[object] = None, browser_headers: Optional[dict] = None) -> list[Document]:
    """Meccanismo di fallback a più livelli"""
    logger.warning(f"Attivazione fallback per URL: {url}")
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    assert browser_headers is not None

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
        # Terzo fallback: Playwright diretto (sostituisce AsyncChromiumLoader)
        logger.info("Tentativo fallback 3: Playwright rinforzato")
        return await handle_chromium_loader(
            urls=[url],
            transformer=None,
            params=None,
            transform_kwargs=None,
            browser_headers=browser_headers
        )
    except Exception as e:
        logger.error(f"Fallback 3 fallito: {str(e)}")

    # Fallback finale: solleva eccezione
    error_msg = f"Tutti i fallback falliti per l'URL: {url}. Impossibile recuperare contenuto."
    logger.error(error_msg)
    raise ValueError(error_msg)


def clean_documents_metadata(docs: Sequence[Document]) -> list[Document]:
    """Pulisce i metadati per tutti i documenti"""
    for doc in docs:
        doc.metadata = clean_metadata(doc.metadata)
    return list(docs)

async def scrape_page(url, params_type_4, browser_headers: Optional[dict] = None, time_sleep=2):
    logger.info("Starting scraping...")
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
                                            args=[
                                                "--disable-blink-features=AutomationControlled",
                                                "--no-sandbox"
                                            ]
                                            )
        try:
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
        finally:
            await browser.close()





async def scrape_page_complex(url, params_type_4, browser_headers: Optional[dict] = None, time_sleep=2):
    logger.info("Starting scraping...")
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    import asyncio
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth, ALL_EVASIONS_DISABLED_KWARGS
    try:
        # **1. Configura Stealth con tutte le evasioni attivate**
        stealth = Stealth(
            # Disabilita le evasioni problematiche se necessario
            # **{**ALL_EVASIONS_DISABLED_KWARGS, "navigator_languages": True}
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-setuid-sandbox",
                    "--disable-accelerated-2d-canvas",
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-extensions",
                    "--disable-default-apps",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ]
            )
            try:
                # **2. Context con parametri umani-realistici**
                context = await browser.new_context(
                    extra_http_headers=browser_headers,
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                    locale="it-IT",
                    timezone_id="Europe/Rome",
                    permissions=["geolocation"],  # Simula permessi reali
                )

                # **3. APPLICA STEALTH AL CONTESTO**
                await stealth.apply_stealth_async(context)

                page = await context.new_page()

                # **4. Intercetta e logga i redirect**
                async def check_redirect(route, request):
                    if "captcha-delivery.com" in request.url or "geo.captcha-delivery.com" in request.url:
                        logger.error(f"❌ CAPTCHA DETECTED! Request: {request.url}")
                        # **Scegli una strategia:**
                        await route.abort()  # Blocca la richiesta
                        # Oppure continua per analizzare la risposta:
                        # await route.continue_()
                    else:
                        await route.continue_()

                await page.route("**/*", check_redirect)

                # **5. Logga ogni navigazione per debug**
                def log_navigation(frame):
                    logger.info(f"🌐 Navigated to: {frame.url}")
                    if "captcha-delivery.com" in frame.url:
                        logger.error("🚫 Siamo stati rediretti al CAPTCHA!")

                page.on("framenavigated", log_navigation)

                # **6. Vai alla pagina con timeout e strategia di attesa**
                try:
                    await page.goto(
                        url=url,
                        wait_until="domcontentloaded",  # Più veloce di "networkidle"
                        timeout=30000
                    )
                except Exception as e:
                    logger.error(f"Errore durante la navigazione: {e}")
                    raise ValueError(f"Navigazione fallita per l'URL: {url}. Errore: {e}")

                # **7. Attendi elementi specifici del sito target**
                try:
                    # **SOSTITUISCI** con un selettore REALE della pagina target
                    # Esempio: await page.wait_for_selector("main, article, .content", timeout=10000)
                    await page.wait_for_selector("body", timeout=5000)  # Fallback generico
                except Exception as e:
                    logger.warning(f"⚠️ Elementi target non trovati: {e}")

                # **8. CORRETTO: Sleep asincrono**
                await asyncio.sleep(params_type_4.time_sleep)

                # **9. VERIFICA FINALE della URL e contenuto**
                current_url = page.url
                if "captcha-delivery.com" in current_url:
                    logger.error("🚫 Pagina CAPTCHA rilevata, scraping annullato.")
                    raise ValueError("Pagina bloccata da CAPTCHA. Impossibile procedere con lo scraping.")

                # **10. Ottieni contenuto e chiudi**
                results = await page.content()
                logger.error(f"Contenuto {results}")
                # **11. Processa e valida il contenuto**
                transformed_content = custom_html_transform(
                    results,
                    selectors_to_extract=params_type_4.tags_to_extract,
                    unwanted_tags=getattr(params_type_4, 'unwanted_tags', []),
                    unwanted_classnames=getattr(params_type_4, 'unwanted_classnames', []),
                    remove_lines=getattr(params_type_4, 'remove_lines', True),
                    remove_comments=getattr(params_type_4, 'remove_comments', True)
                )

                # **12. Doppio check: il contenuto è valido?**
                if "captcha-delivery.com" in transformed_content or len(transformed_content.strip()) < 500:
                    error_msg = "Contenuto bloccato da CAPTCHA o troppo breve (meno di 500 caratteri). Impossibile procedere con lo scraping."
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                metadata = {"source": url}
                doc = Document(page_content=transformed_content, metadata=metadata)
                return [doc]
            finally:
                await browser.close()

    except Exception as e:
        logger.error(f"Playwright scrape failed: {str(e)}")
        raise

async def scrape_page_complex_old(url, params_type_4, browser_headers: Optional[dict] = None, time_sleep=2):
    logger.info("Starting scraping...")
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True,
                                              args=[
                                                  "--disable-blink-features=AutomationControlled",
                                                  "--no-sandbox",
                                                  "--disable-dev-shm-usage",
                                                  "--disable-web-security",
                                                  "--disable-features=IsolateOrigins,site-per-process",
                                                  "--disable-setuid-sandbox",
                                                  "--disable-accelerated-2d-canvas",
                                                  "--disable-gpu",
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
        documents = await get_content_by_url(source,
                                        scrape_type,
                                        parameters_scrape_type_4=parameters_scrape_type_4,
                                        browser_headers=browser_headers)
    else:
        documents = load_document(source, type_source)
    
    # Verifica che i documenti siano validi
    if not documents:
        raise ValueError(f"Nessun documento recuperato dalla sorgente: {source} (tipo: {type_source})")
    
    # Verifica che ci sia almeno un documento con contenuto non vuoto
    has_content = False
    for doc in documents:
        if doc and doc.page_content and doc.page_content.strip():
            has_content = True
            break
    
    if not has_content:
        raise ValueError(f"Documenti recuperati ma contenuto vuoto dalla sorgente: {source} (tipo: {type_source})")
    
    return documents


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