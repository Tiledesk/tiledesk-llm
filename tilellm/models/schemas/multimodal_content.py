from typing import List, Union, Literal, Dict, Any

from pydantic import BaseModel, Field
import base64


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def to_langchain_format(self):
        return {"type": "text", "text": self.text}


# Modello per il contenuto immagine
class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: Union[str, bytes]  # URL, base64 string, o bytes
    mime_type: str = Field(default="image/jpeg")
    detail: str = Field(default="auto")

    def to_langchain_format(self) -> Dict[str, Any]:
        """Converte in formato dizionario per LangChain."""
        url = ""
        if isinstance(self.source, bytes):
            base64_image = base64.b64encode(self.source).decode('utf-8')
            url = f"data:{self.mime_type};base64,{base64_image}"
        elif self.source.strip().startswith(('http://', 'https://')):
            url = self.source
        else: # Assumiamo sia una stringa base64
            url = f"data:{self.mime_type};base64,{self.source}"

        image_data = {"url": url}
        if self.detail != "auto":
            image_data["detail"] = self.detail

        #return {
        #    "type": "image",
        #    "source_type": "base64",
        #    "data": self.source,
        #    "mime_type": self.mime_type,
        #}
        return {"type": "image_url", "image_url": image_data}

    def model_dump(self, **kwargs):
        """Override per convertire bytes in base64 string per JSON"""
        data = super().model_dump(**kwargs)
        if isinstance(data['source'], bytes):
            data['source'] = base64.b64encode(data['source']).decode('utf-8')
        return data

    class Config:
        json_schema_extra = {
            "example": {
                "type": "image",
                "source": "https://example.com/image.jpg",
                "mime_type": "image/jpeg",
                "detail": "auto"
            }
        }


class DocumentContent(BaseModel):
    """Per PDF e altri documenti (Anthropic Claude)"""
    type: Literal["document"] = "document"
    source: Union[str, bytes]
    mime_type: str = Field(default="application/pdf")

    def to_langchain_format(self) -> dict:
        if isinstance(self.source, bytes):
            base64_doc = base64.b64encode(self.source).decode('utf-8')
        else:  # Assumiamo sia già una stringa base64
            base64_doc = self.source

        # Nota: Alcuni modelli come Claude usano "image" anche per i PDF.
        # Se il tuo modello lo richiede, cambia "type" in "image".
        # Altrimenti, se hai un tool specifico, puoi usare un tipo custom.
        # Per ora, usiamo una struttura simile a quella delle immagini.
        return {
            "type": "image",  # Claude 3 gestisce i PDF come input "image"
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": base64_doc,
            },
        }

    def model_dump(self, **kwargs):
        """Override per convertire bytes in base64 string per JSON"""
        data = super().model_dump(**kwargs)
        if isinstance(data['source'], bytes):
            data['source'] = base64.b64encode(data['source']).decode('utf-8')
        return data

    class Config:
        json_schema_extra = {
            "example": {
                "type": "document",
                "source": "<base64_encoded_pdf>",
                "mime_type": "application/pdf"
            }
        }

MultimodalContent = Union[TextContent, ImageContent, DocumentContent]