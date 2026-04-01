import os
import pytest
import pandas as pd
from tilellm.tools.document_tools import load_document, fetch_documents

@pytest.mark.asyncio
async def test_fetch_documents_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
    df.to_csv(csv_file, index=False)
    
    docs = await fetch_documents(
        type_source="csv",
        source=str(csv_file),
        scrape_type=0,
        parameters_scrape_type_4=None,
        browser_headers={}
    )
    
    assert len(docs) == 1
    assert "val1" in docs[0].page_content
    assert docs[0].metadata["type"] == "csv"
    assert docs[0].metadata["file_name"] == "test.csv"

@pytest.mark.asyncio
async def test_fetch_documents_xlsx(tmp_path):
    excel_file = tmp_path / "test.xlsx"
    df = pd.DataFrame({"col1": ["val1"]})
    df.to_excel(excel_file, index=False)
    
    docs = await fetch_documents(
        type_source="xlsx",
        source=str(excel_file),
        scrape_type=0,
        parameters_scrape_type_4=None,
        browser_headers={}
    )
    
    assert len(docs) == 1
    assert "val1" in docs[0].page_content
    assert docs[0].metadata["type"] == "xlsx"

@pytest.mark.asyncio
async def test_fetch_documents_docx(tmp_path):
    import docx
    docx_file = tmp_path / "test.docx"
    doc = docx.Document()
    doc.add_paragraph("Integrated test content")
    doc.save(docx_file)
    
    docs = await fetch_documents(
        type_source="docx",
        source=str(docx_file),
        scrape_type=0,
        parameters_scrape_type_4=None,
        browser_headers={}
    )
    
    assert any("Integrated test content" in d.page_content for d in docs)
    assert any(d.metadata["type"] == "docx" for d in docs)
