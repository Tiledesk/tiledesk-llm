
from langchain_community.document_loaders import BSHTMLLoader

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import AsyncChromiumLoader
import requests


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

def get_content_by_url_with_bs(url:str):
    html = requests.get(url)
    #urls = [url]
    # Load HTML
    #loader = await AsyncChromiumLoader(urls)
    #html = loader.load()

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
        #if index < len(h1_tags) - 1:
        #    print()  # Stampa una riga vuota tra i segmenti


    return testi
