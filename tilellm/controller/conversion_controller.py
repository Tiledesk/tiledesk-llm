import base64
import io
import os
from typing import List, Optional

import pandas as pd
import fitz  # PyMuPDF
from fastapi import HTTPException


from tilellm.models.convertion import ConvertedFile


def process_xlsx_to_csv(file_name: str, file_bytes: bytes) -> List[ConvertedFile]:
    """
    Converte un file XLSX in una lista di file CSV, uno per ogni foglio.
    """
    try:
        # Legge il file XLSX direttamente dai byte in memoria
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets_data = []
        original_name_without_ext = os.path.splitext(file_name)[0]

        # Itera su ogni foglio del file Excel
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Converte il DataFrame in una stringa CSV
            csv_output = df.to_csv(index=False, encoding='utf-8')
            csv_bytes = csv_output.encode('utf-8')

            # Codifica il risultato in Base64
            base64_encoded_csv = base64.b64encode(csv_bytes).decode('utf-8')

            # Crea il nuovo nome del file
            new_file_name = f"{original_name_without_ext}_{sheet_name}.csv"

            sheets_data.append(
                ConvertedFile(
                    FileName=new_file_name,
                    FileExt="csv",
                    FileSize=len(csv_bytes),
                    File=base64_encoded_csv,
                    FileContent=csv_output,
                )
            )
        return sheets_data
    except Exception as e:
        # Se il file non è un XLSX valido o c'è un altro errore
        raise HTTPException(status_code=400, detail=f"Errore durante l'elaborazione del file XLSX: {e}")


def process_pdf_to_text(file_name: str, file_bytes: bytes) -> List[ConvertedFile]:
    """
    Estrae il testo da un file PDF e lo restituisce come singolo file di testo.
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

        # Codifica il testo estratto in Base64
        text_bytes = full_text.encode('utf-8')
        base64_encoded_text = base64.b64encode(text_bytes).decode('utf-8')

        # Prepara il file di output
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
    except Exception as e:
        # Se il file non è un PDF valido o c'è un altro errore
        raise HTTPException(status_code=400, detail=f"Errore durante l'elaborazione del file PDF: {e}")