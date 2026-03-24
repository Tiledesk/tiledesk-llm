import os
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

try:
    import docx
except ImportError:
    docx = None

logger = logging.getLogger(__name__)

class StructuredDocxLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if docx is None:
            raise ImportError(
                "python-docx is not installed. Please install it with `pip install python-docx`."
            )
        
        # docx.Document works with file paths or file-like objects.
        # If it's a URL, we might need to download it first, 
        # but load_document in document_tools.py seems to expect a path/URL that the loader can handle.
        # Actually, PyPDFLoader and others in langchain often handle URLs if they use requests internally,
        # but python-docx needs a local file or a stream.
        
        # If the file_path is a URL, we should download it.
        local_path = self.file_path
        is_temp = False
        if self.file_path.startswith(("http://", "https://")):
            import requests
            import tempfile
            response = requests.get(self.file_path, timeout=60)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(response.content)
                local_path = tmp.name
                is_temp = True
        
        try:
            doc = docx.Document(local_path)
            documents = []
            heading_stack = []

            # We want to maintain order. Docx paragraphs and tables are in a certain order.
            # python-docx doesn't easily give a single iterator for both in document order.
            # But we can iterate through the document's XML elements if needed.
            # For simplicity, let's start with paragraphs then tables as per the plan, 
            # but maybe try to be better.
            
            # Simple implementation as per plan:
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                    
                if para.style and para.style.name.startswith('Heading'):
                    try:
                        # style name is often 'Heading 1', 'Heading 2' etc.
                        parts = para.style.name.split(' ')
                        if len(parts) >= 2 and parts[1].isdigit():
                            level = int(parts[1])
                            heading_stack = heading_stack[:level-1] + [para.text.strip()]
                    except Exception:
                        pass
                    
                    # Also add the heading itself as a document
                    documents.append(Document(
                        page_content=para.text.strip(),
                        metadata={
                            "source": self.file_path,
                            "heading_path": " > ".join(heading_stack),
                            "style": para.style.name if para.style else "Normal",
                            "type": "docx"
                        }
                    ))
                else:
                    documents.append(Document(
                        page_content=para.text.strip(),
                        metadata={
                            "source": self.file_path,
                            "heading_path": " > ".join(heading_stack),
                            "style": para.style.name if para.style else "Normal",
                            "type": "docx"
                        }
                    ))

            for i, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text.strip() for cell in row.cells])
                
                # Convert table to markdown or string
                if table_data:
                    df = pd.DataFrame(table_data)
                    table_text = df.to_markdown(index=False)
                    documents.append(Document(
                        page_content=table_text,
                        metadata={
                            "source": self.file_path,
                            "heading_path": " > ".join(heading_stack),
                            "type": "docx",
                            "element_type": "table",
                            "table_index": i
                        }
                    ))
            
            return documents
        finally:
            if is_temp and os.path.exists(local_path):
                os.remove(local_path)

class ExcelLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        local_path = self.file_path
        is_temp = False
        if self.file_path.startswith(("http://", "https://")):
            import requests
            import tempfile
            response = requests.get(self.file_path, timeout=60)
            response.raise_for_status()
            suffix = ".xlsx" if ".xlsx" in self.file_path.lower() else ".xls"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                local_path = tmp.name
                is_temp = True
        
        try:
            documents = []
            # Read all sheets
            excel_file = pd.ExcelFile(local_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if df.empty:
                    continue
                
                # Option 1: One document per sheet (as markdown)
                # Option 2: Chunking if too large.
                # The plan says: "Sheet with > 100 righe -> chunking verticale (gruppi di 20-50 righe con header)"
                
                col_names = list(df.columns.astype(str))
                file_type = "xlsx" if local_path.endswith(".xlsx") else "xls"
                rows_per_chunk = 50
                if len(df) > 100:
                    for i in range(0, len(df), rows_per_chunk):
                        chunk_df = df.iloc[i:i+rows_per_chunk]
                        table_text = chunk_df.to_markdown(index=False)
                        documents.append(Document(
                            page_content=table_text,
                            metadata={
                                "source": self.file_path,
                                "sheet_name": sheet_name,
                                "col_names": ", ".join(col_names),
                                "row_range": f"{i}-{i+len(chunk_df)}",
                                "type": file_type,
                            }
                        ))
                else:
                    table_text = df.to_markdown(index=False)
                    documents.append(Document(
                        page_content=table_text,
                        metadata={
                            "source": self.file_path,
                            "sheet_name": sheet_name,
                            "col_names": ", ".join(col_names),
                            "type": file_type,
                        }
                    ))
            return documents
        finally:
            if is_temp and os.path.exists(local_path):
                os.remove(local_path)

class CSVLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        local_path = self.file_path
        is_temp = False
        if self.file_path.startswith(("http://", "https://")):
            import requests
            import tempfile
            response = requests.get(self.file_path, timeout=60)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(response.content)
                local_path = tmp.name
                is_temp = True
        
        try:
            df = pd.read_csv(local_path)
            if df.empty:
                return []

            col_names = list(df.columns.astype(str))
            documents = []
            rows_per_chunk = 50
            if len(df) > 100:
                for i in range(0, len(df), rows_per_chunk):
                    chunk_df = df.iloc[i:i+rows_per_chunk]
                    table_text = chunk_df.to_markdown(index=False)
                    documents.append(Document(
                        page_content=table_text,
                        metadata={
                            "source": self.file_path,
                            "col_names": ", ".join(col_names),
                            "row_range": f"{i}-{i+len(chunk_df)}",
                            "type": "csv",
                        }
                    ))
            else:
                table_text = df.to_markdown(index=False)
                documents.append(Document(
                    page_content=table_text,
                    metadata={
                        "source": self.file_path,
                        "col_names": ", ".join(col_names),
                        "type": "csv",
                    }
                ))
            return documents
        finally:
            if is_temp and os.path.exists(local_path):
                os.remove(local_path)
