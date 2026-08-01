"""
L01 products-consistency check — direction **L01 → PDF**.

For every product listed in the structured L01 workbook (the OE's offered products),
verify a PDF technical sheet in the namespace supports it. Match on product **code**,
fallback to normalized **name**. Deterministic (no LLM): retrieval + substring match.

Public API:
  - parse_l01(xlsx_bytes)            -> List[L01Product]
  - fetch_l01(url)                   -> bytes            (httpx, size-guarded)
  - reconcile_l01_to_pdf(products, retrieve, concurrency) -> L01CheckResult
  - run_l01_check(xlsx_url, retrieve, concurrency)        -> L01CheckResult

`retrieve` is an async callable `(query: str) -> (chunks: List[str], metadata: List[dict])`,
so this module stays decoupled from the vector-store repository (testable in isolation).
"""
import asyncio
import io
import logging
from typing import Awaitable, Callable, List, Optional, Tuple

import httpx
import openpyxl

from tilellm.modules.compliance_checker.logic import _normalize
from tilellm.modules.compliance_checker.models_v2 import (
    L01CheckResult,
    L01Product,
    L01ReconItem,
)

logger = logging.getLogger(__name__)

MAX_L01_XLSX_SIZE = 10 * 1024 * 1024  # 10 MB

RetrieveFn = Callable[[str], Awaitable[Tuple[List[str], List[dict]]]]

# Header keywords (accent/case-insensitive) used to locate the code and name columns.
# Matched as PREFIXES, not exact values: the real PA "tracciato" ships headers like
# "CODIFICA ARTICOLO OPERATORE ECONOMICO" / "DENOMINAZIONE ARTICOLO OPERATORE ECONOMICO",
# which no exact-match set can recognise — and the untyped fallback then picks the
# "Lotto" column as the product code for every row.
_CODE_HEADERS = ("codifica", "codice", "code", "sku", "articolo", "cod", "id")
_NAME_HEADERS = ("denominazione", "descrizione", "nome", "prodotto", "description", "name", "product")


def _match_header(header: str, keywords: Tuple[str, ...]) -> bool:
    return any(header.startswith(k) for k in keywords)


def parse_l01(content: bytes) -> List[L01Product]:
    """
    Parse the first sheet of an L01 workbook into a list of products.

    Tolerant to schema: locates the code/name columns by header keyword; if neither
    is recognized, falls back to first column = code, remaining columns joined = name.
    Fully-empty rows are skipped.
    """
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    header = [_normalize(str(c)) if c is not None else "" for c in rows[0]]
    code_idx = next((i for i, h in enumerate(header) if _match_header(h, _CODE_HEADERS)), None)
    # Only the FIRST name column: joining every name-ish column ("DENOMINAZIONE ..."
    # plus "NOME COMMERCIALE MODELLO") yields a string that matches no PDF chunk.
    name_idx = next((i for i, h in enumerate(header) if _match_header(h, _NAME_HEADERS)), None)
    name_idxs = [name_idx] if name_idx is not None else []

    products: List[L01Product] = []
    for raw in rows[1:]:
        cells = ["" if c is None else str(c).strip() for c in raw]
        if not any(cells):
            continue
        if code_idx is not None or name_idxs:
            # ponytail: header-driven mapping; add per-column typing when L01 schema is fixed
            code = cells[code_idx].strip() if code_idx is not None and code_idx < len(cells) else None
            name = " ".join(cells[i] for i in name_idxs if i < len(cells)).strip()
        else:
            # fallback: first column = code, the rest = name
            code = cells[0].strip() or None
            name = " ".join(cells[1:]).strip()
        products.append(L01Product(code=code or None, name=name))
    return products


async def fetch_l01(url: str) -> bytes:
    """Download the L01 xlsx (size-guarded)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
    if len(content) > MAX_L01_XLSX_SIZE:
        raise ValueError(
            f"Il file L01 è troppo grande ({len(content)} byte > {MAX_L01_XLSX_SIZE} byte massimi)."
        )
    logger.debug("fetch_l01: downloaded %d bytes from %s", len(content), url)
    return content


def _match_product(product: L01Product, chunks: List[str]) -> Tuple[Optional[str], int]:
    """Return (matched_by, chunk_index) — 'code' first, then 'name'; (None, -1) if absent."""
    code_n = _normalize(product.code) if product.code else ""
    name_n = _normalize(product.name) if product.name else ""
    normalized = [_normalize(c) for c in chunks]

    if code_n:
        for i, c in enumerate(normalized):
            if code_n in c:
                return "code", i
    if name_n:
        for i, c in enumerate(normalized):
            if name_n in c:
                return "name", i
    return None, -1


async def reconcile_l01_to_pdf(
    products: List[L01Product],
    retrieve: RetrieveFn,
    concurrency: int = 3,
) -> L01CheckResult:
    """
    For each L01 product, retrieve candidate PDF chunks from the namespace and check
    whether the product code (or, as fallback, its name) is present. Returns an
    always-`used=True` result (the check ran).
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(product: L01Product) -> L01ReconItem:
        async with semaphore:
            query = f"{product.code or ''} {product.name}".strip()
            try:
                chunks, metadata = await retrieve(query)
            except Exception as e:  # retrieval failure → treat as absent, flag in log
                logger.warning("L01 retrieval failed for %r: %s", query, e)
                chunks, metadata = [], []
            matched_by, idx = _match_product(product, chunks or [])
            doc = ""
            text = ""
            if idx >= 0:
                meta = metadata[idx] if idx < len(metadata or []) else {}
                doc = meta.get("file_name", meta.get("source", "")) if isinstance(meta, dict) else ""
                text = chunks[idx][:500]
            return L01ReconItem(
                code=product.code,
                name=product.name,
                present_in_pdf=matched_by is not None,
                matched_by=matched_by,
                evidence_document=doc,
                evidence_text=text,
            )

    items = list(await asyncio.gather(*[_one(p) for p in products]))
    matched = sum(1 for it in items if it.present_in_pdf)
    missing = [it.code or it.name for it in items if not it.present_in_pdf]
    return L01CheckResult(
        used=True,
        l01_products_total=len(items),
        matched=matched,
        missing_count=len(missing),
        missing=missing,
        items=items,
    )


async def run_l01_check(
    xlsx_url: str,
    retrieve: RetrieveFn,
    concurrency: int = 3,
) -> L01CheckResult:
    """Fetch + parse the L01 workbook, then reconcile against the PDF namespace."""
    content = await fetch_l01(xlsx_url)
    products = parse_l01(content)
    return await reconcile_l01_to_pdf(products, retrieve, concurrency=concurrency)
