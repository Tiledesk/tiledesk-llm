# modules/conversion/controllers.py
import base64
from binascii import Error
from typing import List

import aiohttp
from fastapi import APIRouter, HTTPException

from tilellm.modules.conversion.models.convertion import ConvertedFile, ConversionRequest, ConversionType
from tilellm.modules.conversion.services.conversion_service import process_xlsx_to_csv, process_pdf_to_text

# 1. Crea il router per questo modulo
router = APIRouter(
    prefix="/api",
    tags=["Conversion"] # Tag per la documentazione OpenAPI (Swagger)
)


@router.post("/convert", response_model=List[ConvertedFile], tags=["Conversion"])
async def convert_file(request: ConversionRequest):
    """
    Converte un file fornito come stringa Base64 o URL.

    - **xlsx_to_csv**: Converte ogni foglio di un file XLSX in un file CSV separato.
    - **pdf_to_text**: Estrae il contenuto testuale da un file PDF.
    """
    # Verifica se file_content è una URL o base64
    if request.is_url():
        try:
            # Scarica il file dalla URL
            async with aiohttp.ClientSession() as session:
                async with session.get(request.file_content) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Impossibile scaricare il file dalla URL. Status: {response.status}"
                        )
                    file_bytes = await response.read()
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Errore durante il download del file: {str(e)}")
    else:
        try:
            # Decodifica la stringa Base64 per ottenere i byte del file
            file_bytes = base64.b64decode(request.file_content)
        except (Error, ValueError):
            raise HTTPException(status_code=400, detail="Contenuto del file non valido: la stringa Base64 non è corretta.")

    if request.conversion_type == ConversionType.XLSX_TO_CSV:
        return process_xlsx_to_csv(request.file_name, file_bytes)

    elif request.conversion_type == ConversionType.PDF_TO_TEXT:
        return process_pdf_to_text(request.file_name, file_bytes)

    # Questo non dovrebbe mai accadere grazie alla validazione Pydantic, ma è una sicurezza in più.
    raise HTTPException(status_code=400, detail="Tipo di conversione non supportato.")