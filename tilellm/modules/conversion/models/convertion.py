from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ConversionType(str, Enum):
    """
    Enum per i tipi di conversione supportati.
    """
    XLSX_TO_CSV = "xlsx_to_csv"
    PDF_TO_TEXT = "pdf_to_text"
    PDF_TO_IMAGES= "pdf_to_images"

class ConversionRequest(BaseModel):
    """
    Modello per la richiesta di conversione in input.
    """
    file_name: str = Field(..., description="Il nome del file originale, inclusa l'estensione (es. 'documento.xlsx').")
    file_content: str = Field(..., description="Il contenuto del file codificato in Base64 oppure una URL (http/https).")
    conversion_type: ConversionType = Field(..., description="Il tipo di conversione da effettuare.")

    def is_url(self) -> bool:
        """Verifica se file_content è una URL."""
        return self.file_content.startswith(('http://', 'https://'))

class ConvertedFile(BaseModel):
    """
    Modello per rappresentare un singolo file convertito in output.
    """
    FileName: str = Field(..., description="Il nome del file generato.")
    FileExt: str = Field(..., description="L'estensione del file generato (es. 'csv' o 'txt').")
    FileSize: Optional[int] = Field(None, description="La dimensione in byte del file generato.")
    File: str = Field(..., description="Il contenuto del file generato, codificato in Base64.")
    FileContent: Optional[str] = Field(None, description="Il contenuto del file generato come testo semplice (es. stringa CSV o testo estratto).")