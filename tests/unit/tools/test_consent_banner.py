"""
Cookie-consent banners are the same on every page of a site: when their text
(or their cookie table) wins the extraction, the page is indexed as the banner
and, on a crawl, every later page looks like a duplicate of it. They are removed
before extraction, for type=url and type=crawl alike.
"""
from tilellm.tools.document_tools import _strip_consent, _trafilatura_documents

URL = "https://www.example.org/servizi"
CONTENT = ("Il servizio anagrafe rilascia certificati di residenza e stato di famiglia "
           "su appuntamento presso gli sportelli del quartiere. ") * 6
BANNER = ("Se preferisci, puoi selezionare qui sotto quali tipologie di cookies ritieni "
          "accettabili per la navigazione su questo sito. ") * 4
COOKIE_TABLE = ("<table><tr><th>Nome</th><th>Dominio</th><th>Scadenza</th></tr>"
                "<tr><td>_pk_id</td><td>example.org</td><td>13 mesi</td></tr>"
                "<tr><td>SESSID</td><td>example.org</td><td>sessione</td></tr></table>")


def _page(body_attrs="", banner_attrs='id="cookie-consent" class="cc-window"'):
    return (f"<html><head><title>Servizi</title></head><body {body_attrs}>"
            f"<div {banner_attrs}><p>{BANNER}</p>{COOKIE_TABLE}</div>"
            f"<main><article><h1>Servizi</h1><p>{CONTENT}</p></article></main></body></html>")


def _text(docs):
    return " ".join(d.page_content for d in docs)


def test_banner_and_its_table_are_not_indexed():
    docs = _trafilatura_documents(_page(), URL)
    assert "anagrafe" in _text(docs)
    assert "cookies" not in _text(docs) and "_pk_id" not in _text(docs)


def test_known_consent_providers_are_removed():
    for attrs in ('id="iubenda-cs-banner"', 'id="onetrust-consent-sdk"', 'id="CybotCookiebotDialog"',
                  'class="cmp-container"', 'class="gdpr-notice"',
                  'id="cc--main" class="c--anim"',  # vanilla-cookieconsent (common on Italian PA sites)
                  'id="cc_div" class="cc_div"', 'class="cc-window cc-banner"'):  # Osano cookieconsent
        text = _text(_trafilatura_documents(_page(banner_attrs=attrs), URL))
        assert "cookies" not in text and "_pk_id" not in text, attrs


def test_page_wrapper_with_a_consent_class_is_kept():
    # <body class="cookie-consent-open"> must not take the whole page with it
    docs = _trafilatura_documents(_page(body_attrs='class="cookie-consent-open"'), URL)
    assert "anagrafe" in _text(docs)


def test_html_without_banner_is_unchanged():
    html = f"<html><body><article><p>{CONTENT}</p></article></body></html>"
    assert _strip_consent(html) == html


def test_div_wrapping_the_page_with_a_consent_class_is_kept():
    html = _page().replace("<body >", '<body><div class="page cookie-banner-visible">').replace(
        "</body>", "</div></body>")
    assert "anagrafe" in _text(_trafilatura_documents(html, URL))
