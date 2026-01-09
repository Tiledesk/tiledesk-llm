import time

import logging

from typing import List, Optional, Sequence, Any

import requests
from bs4 import BeautifulSoup, Comment

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
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
    If scape_type=3 is used AsyncChromiumLoader and the html is transforme@inject_llm_chat_asyncd in text
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

    # Check if the URL is a markdown file
    if url.lower().endswith('.md'):
        logger.info(f"Detected .md file, using handle_unstructured_loader for: {url}")
        return await handle_unstructured_loader(
            urls,
            mode="single",
            browser_headers=browser_headers
        )

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
        logger.error(f"Error in the main method get_content_by_url ({scrape_type}): {str(ex)}")
        # If the error is related to a CAPTCHA, Do not proceed with fallback
        if "CAPTCHA" in str(ex) or "blocked" in str(ex) or "too short" in str(ex):
            raise
        return await robust_fallback(url, params_type_4, browser_headers=browser_headers)


async def handle_unstructured_loader(urls: list, mode: str, strategy: Optional[str] = None, browser_headers: Optional[dict] = None) -> list[Document]:
    """Manages loading with UnstructuredURLLoader"""
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
        if hasattr(loader, 'aclose'):
            await loader.aclose()
        elif hasattr(loader, 'close'):
            loader.close()

async def handle_playwright_scrape(url: str, params: object, browser_headers: Optional[dict] = None) -> list[Document]:
    """Handles scraping with Playwright"""
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    docs = await scrape_page(url, params, browser_headers = browser_headers)
    return clean_documents_metadata(docs)

async def handle_playwright_scrape_complex(url: str, params: object, browser_headers: Optional[dict] = None) -> list[Document]:
    """Handles scraping with Playwright"""
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
    """Handles scraping with Playwright and optional transformation (replace AsyncChromiumLoader)"""
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
    if not docs or any(len(doc.page_content.strip()) < 20 for doc in docs):
        raise ValueError("Insufficient or empty content")

    if transformer and transform_kwargs:
        docs = transformer.transform_documents(docs, **transform_kwargs)
    elif transformer:
        docs = transformer.transform_documents(docs)

    return clean_documents_metadata(docs)


async def scrape_page_fallback_selectors(url, browser_headers: Optional[dict] = None):
    """Fallback with more permissive selectors"""
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
    """Multi-level Fallback Mechanism"""
    logger.warning(f"Fallback activation for URL: {url}")
    if browser_headers is None:
        browser_headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    assert browser_headers is not None

    try:
        # First fallback: alternative sync method
        logger.info("Fallback attempt 1: async scrape")
        return clean_documents_metadata(await scrape_page_fallback_selectors(url, browser_headers=browser_headers))
    except Exception as e:
        logger.error(f"Fallback 1 fail: {str(e)}")

    try:
        # Second fallback: Playwright
        logger.info("Fallback attempt 2: Playwright")
        return await handle_playwright_scrape(url, params, browser_headers=browser_headers)
    except Exception as e:
        logger.error(f"Fallback 2 fail: {str(e)}")

    try:
        # Third fallback: Playwright (in place of AsyncChromiumLoader)
        logger.info("Fallback attempt 3: reinforced Playwright")
        return await handle_chromium_loader(
            urls=[url],
            transformer=None,
            params=None,
            transform_kwargs=None,
            browser_headers=browser_headers
        )
    except Exception as e:
        logger.error(f"Fallback 3 failed: {str(e)}")

    # Final Fallback: arise exception
    error_msg = f"All fallbacks failed for URL: {url}. Unable to retrieve content."
    logger.error(error_msg)
    raise ValueError(error_msg)


def clean_documents_metadata(docs: Sequence[Document]) -> list[Document]:
    """Clean metadata for all documents"""
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
            page = await browser.new_page(extra_http_headers=browser_headers,
                                          java_script_enabled=True)

            try:
                # 1. Initial navigation. 'load' is a good starting point.
                logger.debug("Navigating to page and waiting for 'load' state...")
                await page.goto(url=url, wait_until="load", timeout=60000)

                # 2. DYNAMIC WAIT LOOP (replaces (time.sleep)
                logger.debug("Waiting for DOM to stabilize...")
                previous_html_size = 0
                # Perform the check-up to 5 times (e.g.,5 × 2 seconds = 10 seconds max)
                # to avoid infinite loops on pages that constantly change (e.g., with a timer).
                for _ in range(5):
                    await page.wait_for_timeout(2000)  # Attendi 2 secondi tra un controllo e l'altro

                    current_html_size = len(await page.content())

                    # If the size hasn't changed, the page is stable.
                    if current_html_size == previous_html_size:
                        logger.debug("DOM is stable. Proceeding.")
                        break

                    # Otherwise, update the size and keep checking.
                    previous_html_size = current_html_size
                    logger.debug(f"DOM is still changing... current size: {current_html_size}")
                else:  # This `else` fires only if the `for` loop ends without a `break`
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
        # **1. Configure Stealth with all evasions enabled**
        stealth = Stealth(
            # Disable problematic evasions if necessary.
            **{**ALL_EVASIONS_DISABLED_KWARGS, "navigator_languages": True}
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
                # 2 **Context with human‑realistic parameters.**
                context = await browser.new_context(
                    extra_http_headers=browser_headers,
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                    locale="it-IT",
                    timezone_id="Europe/Rome",
                    permissions=["geolocation"],  # Simulate real permissions
                )

                # **3. Apply STEALTH to the Contest**
                await stealth.apply_stealth_async(context)

                page = await context.new_page()

                # **4. Intercept and log the redirect**
                async def check_redirect(route, request):
                    if "captcha-delivery.com" in request.url or "geo.captcha-delivery.com" in request.url:
                        logger.error(f"❌ CAPTCHA DETECTED! Request: {request.url}")
                        # ** Choose a strategy:**
                        await route.abort()  # Block the request
                        # Otherwise continue in order to analyze the response:
                        # await route.continue_()
                    else:
                        await route.continue_()

                await page.route("**/*", check_redirect)

                # **5. Log each navigation for debug**
                def log_navigation(frame):
                    logger.info(f"🌐 Navigated to: {frame.url}")
                    if "captcha-delivery.com" in frame.url:
                        logger.error("🚫 redirect to CAPTCHA!")

                page.on("framenavigated", log_navigation)

                # **6. Navigate to the page using a timeout and a waiting strategy.**
                try:
                    await page.goto(
                        url=url,
                        wait_until="domcontentloaded",  # Faster than "networkidle"
                        timeout=30000
                    )
                except Exception as e:
                    #logger.error(f"Navigation error: {e}")
                    raise ValueError(f"Navigation failed for URL: {url}. Error: {e}")

                # **7. Wait for specific elements on the target site.**
                try:
                    # **Replace ** with a real selector from the target page.
                    # Es: await page.wait_for_selector("main, article, .content", timeout=10000)
                    await page.wait_for_selector("body", timeout=5000)  # Fallback generico
                except Exception as e:
                    logger.warning(f"⚠️ Target elements not found: {e}")

                # **8. CORRECT: async Sleep**
                await asyncio.sleep(params_type_4.time_sleep)

                # **9. URL and Content FINAL CHECK**
                current_url = page.url
                if "captcha-delivery.com" in current_url:
                    logger.error("🚫 CAPTCHA page detected, scraping canceled.")
                    raise ValueError("Page blocked by CAPTCHA. Unable to proceed with scraping.")

                # **10. Retrieve the content and close.**
                results = await page.content()
                logger.error(f"Content {results}")
                # **11. Process and validate the content.**
                transformed_content = custom_html_transform(
                    results,
                    selectors_to_extract=params_type_4.tags_to_extract,
                    unwanted_tags=getattr(params_type_4, 'unwanted_tags', []),
                    unwanted_classnames=getattr(params_type_4, 'unwanted_classnames', []),
                    remove_lines=getattr(params_type_4, 'remove_lines', True),
                    remove_comments=getattr(params_type_4, 'remove_comments', True)
                )

                # **12. Double check: is the content valid?**
                if "captcha-delivery.com" in transformed_content or len(transformed_content.strip()) < 50:
                    error_msg = "Content blocked by CAPTCHA or too short (less than 20 characters). Unable to proceed with scraping."
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)

                metadata = {"source": url}
                doc = Document(page_content=transformed_content, metadata=metadata)
                print(doc)
                return [doc]
            finally:
                await browser.close()

    except Exception as e:
        logger.error(f"Playwright scrape failed: {str(e)}")
        raise

async def scrape_page_complex_a(url, params_type_4, browser_headers: Optional[dict] = None, time_sleep=2):
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
            page = await browser.new_page(extra_http_headers=browser_headers,
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
    Transform HTML content using custom CSS selectors

    Args:
        html_content (str): HTML content
        selectors_to_extract (list): list of CSS selectors (es. ["div.class", "p#id", "section"])
        unwanted_tags (list): Tag to remove
        unwanted_classnames (list): Class to remove
        remove_lines (bool): remove blank line
        remove_comments (bool): remove HTML comments
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
    # extracted_elements = []

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
                logger.error(f"Playwright scrape failed: {str(e)}")
                elements = soup.find_all(selector)
                elements_to_process.extend(elements)
            except Exception:
                # Se anche questo fallisce, continua con il prossimo selettore
                logger.error(f"Playwright scrape failed: {str(e)}")
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
        loader = PyPDFLoader(url)
    elif type_source == 'docx':
        logger.info(f'Loading {url}')
        loader = Docx2txtLoader(url)
    elif type_source == 'txt':
        logger.info(f'Loading {url}')
        loader = TextLoader(url)
    elif type_source == 'md':
        logger.info(f'Loading {url}')
        loader = UnstructuredMarkdownLoader(url)
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
        logger.info(f": {request.url}")
        logger.info("Sent Headers. ")
        headers = await request.all_headers()
        for name, value in headers.items():
            logger.info(f"  {name}: {value}")