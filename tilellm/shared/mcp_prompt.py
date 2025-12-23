from langchain_core.prompts import PromptTemplate

# Template for Base64 Content Management (used in both simple and advanced agents)
MCP_BASE64_MANAGEMENT_TEMPLATE = PromptTemplate.from_template(
    """
IMPORTANT: Base64 Content Management

Large base64-encoded images are automatically extracted and replaced with references
to avoid context overflow. You will see references like:
- [IMAGE_REF:base64_ref_1:length=52341]

When you need to analyze image content:
1. Use the invoke_multimodal_llm tool
2. Pass the reference: images_base64=["<base64_ref_1>"]
3. The tool will automatically resolve and analyze it

Currently available: {storage_count} references
"""
)

# Template for Document Processing Header
MCP_DOC_HEADER_TEMPLATE = PromptTemplate.from_template(
    """=== DOCUMENT PROCESSING INSTRUCTIONS ===

You have {doc_count} document(s) attached and {tool_count} tool(s) available.

DOCUMENTS PROVIDED:
{doc_list}"""
)

# Template for Document Processing Instructions and Examples
MCP_DOC_INSTRUCTIONS_TEMPLATE = PromptTemplate.from_template(
    """HOW TO PROCESS DOCUMENTS:
1. You will see document references like:
   - [DOCUMENT_doc_1: application/pdf, URL=https://example.com/file.pdf] (URL type)
   - [DOCUMENT_doc_2: application/pdf, 52341 bytes] (BASE64 type)

2. Check the MCP tool's parameters to understand what it accepts:
   - If tool has 'url' parameter → use it for URL documents
   - If tool has 'pdf_base64' or 'file_data' → use it for BASE64 documents

3. CRITICAL - How to retrieve and pass document data:
   a) Look up the document ID in the 'DOCUMENTS PROVIDED' section above
   b) Check if it's Type: URL or Type: BASE64
   c) For URL documents:
      - Find the URL listed in 'DOCUMENTS PROVIDED'
      - Pass it directly to the tool's 'url' parameter (e.g., url='https://...')
      - DO NOT try to download or convert it - the tool handles this!
   d) For BASE64 documents:
      - The base64 data is in storage (you don't see it to avoid context overflow)
      - Pass the document ID reference to the tool (the system resolves it automatically)
      - Or use placeholder like 'pdf_base64=<base64_data_from_storage> '

EXAMPLES:
  Example 1 - URL Document:
    User: 'Convert the PDF to images'
    You see: [DOCUMENT_doc_1: application/pdf, URL=https://pdfobject.com/pdf/sample.pdf]
    Tool param: 'url' (accepts URL)
    ✓ CORRECT: convert_pdf_to_images(url='https://pdfobject.com/pdf/sample.pdf')
    ✗ WRONG: convert_pdf_to_images(pdf_base64='...')  ← Don't download it yourself!

  Example 2 - BASE64 Document:
    User: 'Extract text from PDF'
    You see: [DOCUMENT_doc_2: application/pdf, 52341 bytes]
    Tool param: 'pdf_base64' (accepts base64)
    ✓ CORRECT: extract_text(pdf_base64='<base64_data_from_storage>')

Note: The system automatically manages base64 data to avoid context overflow.
"""
)

# Template for Internal Tool Instructions
MCP_INTERNAL_TOOL_TEMPLATE = PromptTemplate.from_template(
    """INTERNAL TOOL AVAILABLE:
  • invoke_multimodal_llm: Analyzes images/documents with vision capabilities
  Use this after converting documents to images for visual analysis.
"""
)
