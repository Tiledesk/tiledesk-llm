import base64
import io
import logging
import os
import asyncio
from typing import List, Optional

import pandas as pd
import fitz  # PyMuPDF
import aiohttp
from fastapi import HTTPException
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from pdf2image import convert_from_bytes
from PIL import Image

from tilellm.modules.conversion.models.convertion import ConvertedFile

logger = logging.getLogger(__name__)
# ============================================================================
# HELPER FUNCTIONS - Funzioni di utility
# ============================================================================

async def _download_file_from_url(url: str) -> bytes:
    """
    Scarica un file da una URL HTTP/HTTPS usando aiohttp.

    Args:
        url: URL del file da scaricare (http:// o https://)

    Returns:
        bytes: Contenuto del file scaricato

    Raises:
        HTTPException: Se il download fallisce
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download file from URL. Status: {response.status}"
                    )
                return await response.read()
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unexpected error downloading file: {str(e)}")


def _decode_file_content(file_content: str) -> bytes:
    """
    Decodifica il contenuto del file da vari formati.

    Args:
        file_content: Può essere:
            - Base64 puro (senza prefisso)
            - Data URI (data:application/pdf;base64,...)

    Returns:
        bytes: Contenuto decodificato
    """
    # Se è un data URI, rimuovi il prefisso
    if file_content.startswith("data:"):
        # Formato: data:mime/type;base64,<base64_data>
        parts = file_content.split(",", 1)
        if len(parts) == 2:
            file_content = parts[1]

    # Decodifica base64
    return base64.b64decode(file_content)


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Codifica un'immagine PIL in base64 (formato PNG).

    Args:
        image: Immagine PIL da codificare

    Returns:
        str: Stringa base64 dell'immagine in formato PNG
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


async def _get_file_bytes(file_content: str) -> bytes:
    """
    Ottiene i bytes del file dal contenuto fornito.
    Gestisce URL (http://, https://) e base64 (con o senza data: prefix).

    Args:
        file_content: URL o contenuto base64 del file

    Returns:
        bytes: Contenuto del file
    """
    # Se è una URL HTTP/HTTPS, scarica il file
    if file_content.startswith(("http://", "https://")):
        return await _download_file_from_url(file_content)

    # Altrimenti è base64 o data URI
    return _decode_file_content(file_content)


# ============================================================================
# CORE FUNCTIONS - Logica di conversione riutilizzabile
# ============================================================================

class ConvertedSheet:
    """Rappresenta un foglio Excel convertito in CSV."""
    def __init__(self, sheet_name: str, csv_content: str):
        self.sheet_name = sheet_name
        self.csv_content = csv_content


def _process_xlsx_to_csv_core(file_name: str, file_bytes: bytes) -> List[ConvertedSheet]:
    """
    Core function: Converte un file XLSX in CSV.
    Restituisce solo il testo CSV, senza codifica base64.

    Returns:
        List[ConvertedSheet]: Lista di fogli convertiti con il loro contenuto CSV come testo
    """
    try:
        # Legge il file XLSX direttamente dai byte in memoria
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets_data = []

        # Itera su ogni foglio del file Excel
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Converte il DataFrame in una stringa CSV
            csv_content = df.to_csv(index=False, encoding='utf-8')

            sheets_data.append(ConvertedSheet(sheet_name, csv_content))

        return sheets_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore durante l'elaborazione del file XLSX: {e}")


def _process_pdf_to_text_core(file_name: str, file_bytes: bytes) -> str:
    """
    Core function: Estrae il testo da un file PDF.
    Restituisce solo il testo estratto, senza codifica base64.

    Returns:
        str: Il testo estratto dal PDF
    """
    try:
        # Apre il PDF dai byte in memoria usando PyMuPDF
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""

        # Estrae il testo da ogni pagina
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text("text")

        pdf_document.close()

        return full_text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore durante l'elaborazione del file PDF: {e}")


async def _pdf_to_images_core(file_name: str, file_bytes: bytes, dpi: int = 300) -> List[str]:
    """
    Core function: Converte un file PDF in una lista di immagini PNG base64.
    Tutte le operazioni bloccanti vengono eseguite in thread separati.

    Args:
        file_name: Nome del file PDF
        file_bytes: Contenuto del PDF in bytes
        dpi: Risoluzione DPI per la conversione (default 300)

    Returns:
        List[str]: Lista di immagini PNG codificate in base64, una per pagina

    Raises:
        Exception: Per errori durante la conversione o codifica
    """
    try:
        logger.info(f"Avvio conversione PDF '{file_name}'. DPI: {dpi}, Dimensione: {len(file_bytes)} bytes")

        # --- 1. Conversione PDF -> Immagini (Operazione bloccante) ---
        logger.info("Conversione PDF in immagini con Poppler...")
        images = await asyncio.to_thread(
            convert_from_bytes,
            pdf_file=file_bytes,
            dpi=dpi,
            fmt='png',
            thread_count=4  # Utilizza 4 thread per Poppler
        )
        page_count = len(images)
        logger.info(f"Conversione completata. Generate {page_count} immagini.")

        # --- 2. Codifica Immagini -> Base64 (Operazione bloccante parallela) ---
        logger.info(f"Codifica di {page_count} immagini in PNG Base64...")
        encoding_tasks = [
            asyncio.to_thread(encode_image_to_base64, image) for image in images
        ]
        images_base64 = await asyncio.gather(*encoding_tasks)
        logger.info("Codifica immagini completata con successo.")

        return images_base64

    except Exception as e:
        logger.error(f"Errore durante la conversione PDF o codifica immagini: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Errore durante l'elaborazione del PDF: {str(e)}")

# ============================================================================
# API FUNCTIONS - Per l'endpoint REST (mantiene firma ConversionRequest)
# ============================================================================


def process_xlsx_to_csv(file_name: str, file_bytes: bytes) -> List[ConvertedFile]:
    """
    API wrapper: converte XLSX in CSV e codifica in base64.
    Mantiene compatibilità con l'API REST esistente.
    """
    # Ottieni i fogli convertiti (solo testo)
    converted_sheets = _process_xlsx_to_csv_core(file_name, file_bytes)

    # Costruisci i ConvertedFile con codifica base64
    result = []
    original_name_without_ext = os.path.splitext(file_name)[0]

    for sheet in converted_sheets:
        csv_bytes = sheet.csv_content.encode('utf-8')
        base64_encoded_csv = base64.b64encode(csv_bytes).decode('utf-8')
        new_file_name = f"{original_name_without_ext}_{sheet.sheet_name}.csv"

        result.append(
            ConvertedFile(
                FileName=new_file_name,
                FileExt="csv",
                FileSize=len(csv_bytes),
                File=base64_encoded_csv,
                FileContent=sheet.csv_content,
            )
        )

    return result



def process_pdf_to_text(file_name: str, file_bytes: bytes) -> List[ConvertedFile]:
    """
    API wrapper: estrae testo da PDF e codifica in base64.
    Mantiene compatibilità con l'API REST esistente.
    """
    # Ottieni il testo estratto
    full_text = _process_pdf_to_text_core(file_name, file_bytes)

    # Costruisci il ConvertedFile con codifica base64
    text_bytes = full_text.encode('utf-8')
    base64_encoded_text = base64.b64encode(text_bytes).decode('utf-8')
    original_name_without_ext = os.path.splitext(file_name)[0]
    new_file_name = f"{original_name_without_ext}.txt"

    return [
        ConvertedFile(
            FileName=new_file_name,
            FileExt="txt",
            FileSize=len(text_bytes),
            File=base64_encoded_text,
            FileContent=full_text,
        )
    ]


async def process_pdf_to_images(file_name: str, file_bytes: bytes, dpi: int = 300) -> List[ConvertedFile]:
    """
    API wrapper: converte PDF in immagini PNG e codifica in base64.
    Mantiene compatibilità con l'API REST esistente.

    Args:
        file_name: Nome del file PDF
        file_bytes: Contenuto del PDF in bytes
        dpi: Risoluzione DPI per la conversione (default 300)

    Returns:
        List[ConvertedFile]: Lista di file convertiti, uno per pagina
    """
    # Ottieni le immagini base64
    images_base64 = await _pdf_to_images_core(file_name, file_bytes, dpi)

    # Costruisci i ConvertedFile per ogni pagina
    result = []
    original_name_without_ext = os.path.splitext(file_name)[0]

    for idx, img_base64 in enumerate(images_base64, start=1):
        # Decodifica per ottenere la dimensione effettiva
        img_bytes = base64.b64decode(img_base64)
        new_file_name = f"{original_name_without_ext}_page_{idx}.png"

        result.append(
            ConvertedFile(
                FileName=new_file_name,
                FileExt="png",
                FileSize=len(img_bytes),
                File=img_base64,
                FileContent=None,  # Le immagini non hanno contenuto testuale
            )
        )

    return result


# ============================================================================
# LANGCHAIN TOOL WRAPPERS - Schema semplificato per LLM
# ============================================================================

class PDFToTextInput(BaseModel):
    """Input schema for PDF to text conversion tool."""
    file_content: str = Field(
        ...,
        description="The PDF file content. Can be: 1) Base64-encoded data, 2) Data URI (data:application/pdf;base64,...), 3) HTTP/HTTPS URL to download the PDF"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Optional: The PDF filename (e.g., 'document.pdf'). If not provided, a default name will be used"
    )


class XLSXToCSVInput(BaseModel):
    """Input schema for XLSX to CSV conversion tool."""
    file_content: str = Field(
        ...,
        description="The Excel file content. Can be: 1) Base64-encoded data, 2) Data URI (data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,...), 3) HTTP/HTTPS URL to download the Excel file"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Optional: The Excel filename (e.g., 'data.xlsx'). If not provided, a default name will be used"
    )


class PDFToImagesInput(BaseModel):
    """Input schema for PDF to images conversion tool."""
    file_content: str = Field(
        ...,
        description="The PDF file content. Can be: 1) Base64-encoded data, 2) Data URI (data:application/pdf;base64,...), 3) HTTP/HTTPS URL to download the PDF"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Optional: The PDF filename (e.g., 'document.pdf'). If not provided, a default name will be used"
    )
    dpi: int = Field(
        default=300,
        description="Resolution in DPI (default 300, optimal for OCR). Valid range: 1-600",
        gt=0,
        le=600
    )


class PdfOutput(BaseModel):
    """
    Modello per il risultato della conversione: una lista di immagini.
    """
    images_base64: List[str] = Field(
        ...,
        description="Lista di immagini PNG codificate in base64, una per ogni pagina."
    )
    page_count: int = Field(
        ...,
        description="Il numero di pagine convertite."
    )

@tool(args_schema=PDFToTextInput)
async def convert_pdf_to_text_tool(file_content: str, file_name: Optional[str] = None) -> str:
    """
    Extracts plain text from a PDF file and returns it as readable text.

    This tool accepts PDF files in multiple formats:
    - Base64-encoded data (raw base64 string)
    - Data URI format (data:application/pdf;base64,<data>)
    - HTTP/HTTPS URL (the tool will download the file automatically)

    Use this tool when you need to:
    - Read the content of PDF documents
    - Extract text data from PDF files for analysis
    - Convert PDF documents into searchable text format
    - Process PDF files from URLs or base64 data

    The tool processes all pages and returns the complete text content.

    Args:
        file_content: The PDF file as base64, data URI, or HTTP/HTTPS URL
        file_name: Optional filename (e.g., 'report.pdf'). If not provided, defaults to 'document.pdf'

    Returns:
        Extracted text content from the PDF file

    Example:
        # From URL
        convert_pdf_to_text_tool("https://example.com/document.pdf")

        # From base64
        convert_pdf_to_text_tool("JVBERi0xLjQK...")

        # From data URI
        convert_pdf_to_text_tool("data:application/pdf;base64,JVBERi0xLjQK...")
    """
    try:
        # Ottieni i bytes del file (gestisce URL, base64, data URI)
        file_bytes = await _get_file_bytes(file_content)

        # Usa il nome file fornito o un default
        if not file_name:
            file_name = "document.pdf"

        # Usa la funzione core che restituisce direttamente il testo
        text_content = _process_pdf_to_text_core(file_name, file_bytes)

        if text_content:
            return text_content
        return "No text could be extracted from the PDF."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


@tool(args_schema=XLSXToCSVInput)
async def convert_xlsx_to_csv_tool(file_content: str, file_name: Optional[str] = None) -> str:
    """
    Converts an Excel (XLSX) file into CSV format, extracting data from all sheets.

    This tool accepts Excel files in multiple formats:
    - Base64-encoded data (raw base64 string)
    - Data URI format (data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,<data>)
    - HTTP/HTTPS URL (the tool will download the file automatically)

    Use this tool when you need to:
    - Read data from Excel spreadsheets
    - Extract tabular data from XLSX files
    - Convert Excel worksheets into CSV format for analysis
    - Process Excel files from URLs or base64 data

    Each sheet in the workbook is converted to CSV format. The tool returns
    all sheets with their data in a readable format.

    Args:
        file_content: The Excel file as base64, data URI, or HTTP/HTTPS URL
        file_name: Optional filename (e.g., 'data.xlsx'). If not provided, defaults to 'spreadsheet.xlsx'

    Returns:
        Summary of all sheets with their CSV content

    Example:
        # From URL
        convert_xlsx_to_csv_tool("https://example.com/data.xlsx")

        # From base64
        convert_xlsx_to_csv_tool("UEsDBBQABgAI...")

        # From data URI
        convert_xlsx_to_csv_tool("data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,UEsDBBQABgAI...")
    """
    try:
        # Ottieni i bytes del file (gestisce URL, base64, data URI)
        file_bytes = await _get_file_bytes(file_content)

        # Usa il nome file fornito o un default
        if not file_name:
            file_name = "spreadsheet.xlsx"

        # Usa la funzione core che restituisce direttamente il testo CSV
        converted_sheets = _process_xlsx_to_csv_core(file_name, file_bytes)

        # Costruisci un output leggibile per l'LLM
        output_parts = []
        for idx, sheet in enumerate(converted_sheets, 1):
            output_parts.append(f"Sheet {idx}: {sheet.sheet_name}")
            output_parts.append(f"Content:\n{sheet.csv_content}\n")

        return "\n".join(output_parts)
    except Exception as e:
        return f"Error processing XLSX: {str(e)}"


@tool(args_schema=PDFToImagesInput)
async def convert_pdf_to_images_tool(file_content: str, file_name: Optional[str] = None, dpi: int = 300) -> PdfOutput:
    """
    Converts a PDF file into PNG images, one image per page, and returns them as base64-encoded strings.

    This tool accepts PDF files in multiple formats:
    - Base64-encoded data (raw base64 string)
    - Data URI format (data:application/pdf;base64,<data>)
    - HTTP/HTTPS URL (the tool will download the file automatically)

    Use this tool when you need to:
    - Extract visual content from PDF documents
    - Convert PDF pages into images for OCR processing
    - Generate image previews of PDF documents
    - Process PDFs with complex layouts or graphics

    The tool processes all pages and returns each page as a separate PNG image
    encoded in base64 format. The images can be used for further processing
    or analysis.

    Args:
        file_content: The PDF file as base64, data URI, or HTTP/HTTPS URL
        file_name: Optional filename (e.g., 'document.pdf'). If not provided, defaults to 'document.pdf'
        dpi: Resolution in DPI (default 300, optimal for OCR). Valid range: 1-600

    Returns:
        PdfOutput: Object containing images_base64 (list of base64 PNG strings) and page_count

    Example:
        # From URL
        convert_pdf_to_images_tool("https://example.com/document.pdf")

        # From URL with custom DPI
        convert_pdf_to_images_tool("https://example.com/document.pdf", dpi=150)

        # From base64
        convert_pdf_to_images_tool("JVBERi0xLjQK...")

        # From data URI
        convert_pdf_to_images_tool("data:application/pdf;base64,JVBERi0xLjQK...")
    """
    try:
        # Ottieni i bytes del file (gestisce URL, base64, data URI)
        file_bytes = await _get_file_bytes(file_content)

        # Usa il nome file fornito o un default
        if not file_name:
            file_name = "document.pdf"

        # Usa la funzione core che restituisce le immagini base64
        images_base64 = await _pdf_to_images_core(file_name, file_bytes, dpi)
        print(len(images_base64))


        # Restituisci l'oggetto PdfOutput
        return PdfOutput(
            images_base64=images_base64,
            page_count=len(images_base64)
        ).model_dump()
    except Exception as e:
        logger.error(f"Error in convert_pdf_to_images_tool: {e}", exc_info=True)
        # In caso di errore, restituisci un PdfOutput vuoto o rilancia l'eccezione
        raise HTTPException(status_code=400, detail=f"Error processing PDF to images: {str(e)}")

