#!/usr/bin/env python3
"""
Unit tests for conversion services.
Test core functions in isolation with mocked dependencies.
"""
import pytest
import base64
import io
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from fastapi import HTTPException

from tilellm.modules.conversion.services.conversion_service import (
    _process_xlsx_to_csv_core,
    _process_pdf_to_text_core,
    _pdf_to_images_core,
    _get_file_bytes,
    _download_file_from_url,
    _decode_file_content,
    encode_image_to_base64,
    ConvertedSheet
)


class TestConversionServiceCore:
    """Test core conversion functions with mocked dependencies."""
    
    def test_encode_image_to_base64(self):
        """Test encoding PIL image to base64."""
        from PIL import Image
        # Create a simple test image
        img = Image.new('RGB', (10, 10), color='red')
        result = encode_image_to_base64(img)
        
        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    def test_decode_file_content_base64(self):
        """Test decoding plain base64 content."""
        test_data = b"test data"
        encoded = base64.b64encode(test_data).decode()
        result = _decode_file_content(encoded)
        assert result == test_data
    
    def test_decode_file_content_data_uri(self):
        """Test decoding data URI with base64."""
        test_data = b"test data"
        encoded = base64.b64encode(test_data).decode()
        data_uri = f"data:application/pdf;base64,{encoded}"
        result = _decode_file_content(data_uri)
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_download_file_from_url_success(self):
        """Test downloading file from URL with mocked aiohttp."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"file content")
        
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await _download_file_from_url("http://example.com/file.pdf")
            assert result == b"file content"
    
    @pytest.mark.asyncio
    async def test_download_file_from_url_failure(self):
        """Test URL download failure raises HTTPException."""
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(HTTPException) as exc:
                await _download_file_from_url("http://example.com/notfound.pdf")
            assert exc.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_get_file_bytes_url(self):
        """Test getting file bytes from URL."""
        with patch('tilellm.modules.conversion.services.conversion_service._download_file_from_url',
                  AsyncMock(return_value=b"url content")) as mock_download:
            result = await _get_file_bytes("http://example.com/file.pdf")
            assert result == b"url content"
            mock_download.assert_called_once_with("http://example.com/file.pdf")
    
    @pytest.mark.asyncio
    async def test_get_file_bytes_base64(self):
        """Test getting file bytes from base64 string."""
        test_data = b"test content"
        encoded = base64.b64encode(test_data).decode()
        
        result = await _get_file_bytes(encoded)
        assert result == test_data
    
    @pytest.mark.asyncio 
    async def test_get_file_bytes_data_uri(self):
        """Test getting file bytes from data URI."""
        test_data = b"test content"
        encoded = base64.b64encode(test_data).decode()
        data_uri = f"data:application/pdf;base64,{encoded}"
        
        result = await _get_file_bytes(data_uri)
        assert result == test_data


class TestXLSXToCSVConversion:
    """Test XLSX to CSV conversion functions."""
    
    def test_process_xlsx_to_csv_core_success(self):
        """Test successful XLSX to CSV conversion."""
        # Create mock Excel file bytes
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'X': ['a', 'b'], 'Y': ['c', 'd']})
        
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df1.to_excel(writer, sheet_name='Sheet1', index=False)
                df2.to_excel(writer, sheet_name='Sheet2', index=False)
            excel_bytes = output.getvalue()
        
        result = _process_xlsx_to_csv_core("test.xlsx", excel_bytes)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(sheet, ConvertedSheet) for sheet in result)
        assert result[0].sheet_name == 'Sheet1'
        assert result[1].sheet_name == 'Sheet2'
        
        # Verify CSV content
        csv1 = result[0].csv_content
        csv2 = result[1].csv_content
        assert 'A,B' in csv1
        assert '1,2' in csv1
        assert 'X,Y' in csv2
        assert 'a,c' in csv2
    
    def test_process_xlsx_to_csv_core_invalid_file(self):
        """Test XLSX conversion with invalid file raises HTTPException."""
        invalid_bytes = b"not an excel file"
        
        with pytest.raises(HTTPException) as exc:
            _process_xlsx_to_csv_core("test.xlsx", invalid_bytes)
        assert exc.value.status_code == 400


class TestPDFToTextConversion:
    """Test PDF to text conversion functions."""
    
    def test_process_pdf_to_text_core_success(self):
        """Test successful PDF text extraction."""
        mock_fitz = Mock()
        mock_document = Mock()
        mock_page = Mock()
        
        # Mock fitz.open
        mock_fitz.open = Mock(return_value=mock_document)
        mock_document.__len__ = Mock(return_value=1)
        mock_document.load_page = Mock(return_value=mock_page)
        mock_page.get_text = Mock(return_value="Extracted text content")
        mock_document.close = Mock()
        
        with patch.dict('sys.modules', {'fitz': mock_fitz}):
            result = _process_pdf_to_text_core("test.pdf", b"pdf bytes")
            
            assert result == "Extracted text content"
            mock_document.close.assert_called_once()
    
    def test_process_pdf_to_text_core_error(self):
        """Test PDF text extraction error raises HTTPException."""
        mock_fitz = Mock()
        mock_fitz.open = Mock(side_effect=Exception("PDF error"))
        
        with patch.dict('sys.modules', {'fitz': mock_fitz}):
            with pytest.raises(HTTPException) as exc:
                _process_pdf_to_text_core("test.pdf", b"pdf bytes")
            assert exc.value.status_code == 400


class TestPDFToImagesConversion:
    """Test PDF to images conversion functions."""
    
    @pytest.mark.asyncio
    async def test_pdf_to_images_core_success(self):
        """Test successful PDF to images conversion."""
        # Mock pdf2image
        mock_image = Mock()
        mock_image.save = Mock()
        
        mock_convert_from_bytes = Mock(return_value=[mock_image, mock_image])
        
        # Mock PIL image and encode function
        with patch('tilellm.modules.conversion.services.conversion_service.convert_from_bytes',
                  mock_convert_from_bytes):
            with patch('tilellm.modules.conversion.services.conversion_service.encode_image_to_base64',
                      side_effect=['base64_1', 'base64_2']):
                
                result = await _pdf_to_images_core("test.pdf", b"pdf bytes", dpi=300)
                
                assert isinstance(result, list)
                assert len(result) == 2
                assert result == ['base64_1', 'base64_2']
                mock_convert_from_bytes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pdf_to_images_core_error(self):
        """Test PDF to images conversion error raises HTTPException."""
        mock_convert_from_bytes = Mock(side_effect=Exception("Conversion error"))
        
        with patch('tilellm.modules.conversion.services.conversion_service.convert_from_bytes',
                  mock_convert_from_bytes):
            with pytest.raises(HTTPException) as exc:
                await _pdf_to_images_core("test.pdf", b"pdf bytes", dpi=300)
            assert exc.value.status_code == 400


# Test for LangChain tool wrappers
class TestLangChainTools:
    """Test LangChain tool wrapper functions."""
    
    @pytest.mark.asyncio
    async def test_convert_pdf_to_text_tool_success(self):
        """Test PDF to text tool success."""
        from tilellm.modules.conversion.services.conversion_service import convert_pdf_to_text_tool
        
        with patch('tilellm.modules.conversion.services.conversion_service._get_file_bytes',
                  AsyncMock(return_value=b"pdf bytes")):
            with patch('tilellm.modules.conversion.services.conversion_service._process_pdf_to_text_core',
                      return_value="Extracted text"):
                result = await convert_pdf_to_text_tool("base64_content", "test.pdf")
                assert result == "Extracted text"
    
    @pytest.mark.asyncio
    async def test_convert_pdf_to_text_tool_error(self):
        """Test PDF to text tool error returns error message."""
        from tilellm.modules.conversion.services.conversion_service import convert_pdf_to_text_tool
        
        with patch('tilellm.modules.conversion.services.conversion_service._get_file_bytes',
                  AsyncMock(side_effect=Exception("PDF error"))):
            result = await convert_pdf_to_text_tool("base64_content", "test.pdf")
            assert "Error processing PDF" in result
    
    @pytest.mark.asyncio
    async def test_convert_xlsx_to_csv_tool_success(self):
        """Test XLSX to CSV tool success."""
        from tilellm.modules.conversion.services.conversion_service import convert_xlsx_to_csv_tool
        from tilellm.modules.conversion.services.conversion_service import ConvertedSheet
        
        mock_sheets = [
            ConvertedSheet("Sheet1", "col1,col2\nval1,val2"),
            ConvertedSheet("Sheet2", "colA,colB\nvalA,valB")
        ]
        
        with patch('tilellm.modules.conversion.services.conversion_service._get_file_bytes',
                  AsyncMock(return_value=b"excel bytes")):
            with patch('tilellm.modules.conversion.services.conversion_service._process_xlsx_to_csv_core',
                      return_value=mock_sheets):
                result = await convert_xlsx_to_csv_tool("base64_content", "test.xlsx")
                assert "Sheet 1: Sheet1" in result
                assert "col1,col2" in result
                assert "Sheet 2: Sheet2" in result
    
    @pytest.mark.asyncio
    async def test_convert_pdf_to_images_tool_success(self):
        """Test PDF to images tool success."""
        from tilellm.modules.conversion.services.conversion_service import convert_pdf_to_images_tool
        
        with patch('tilellm.modules.conversion.services.conversion_service._get_file_bytes',
                  AsyncMock(return_value=b"pdf bytes")):
            with patch('tilellm.modules.conversion.services.conversion_service._pdf_to_images_core',
                      AsyncMock(return_value=['img1_base64', 'img2_base64'])):
                result = await convert_pdf_to_images_tool("base64_content", "test.pdf", dpi=300)
                assert 'images_base64' in result
                assert result['images_base64'] == ['img1_base64', 'img2_base64']
                assert result['page_count'] == 2