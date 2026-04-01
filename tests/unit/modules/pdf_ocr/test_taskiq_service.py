#!/usr/bin/env python3
"""
Unit tests for PDF OCR Taskiq and Redis Queue services.
Tests the new TaskiqService (URL-only, MinIO-independent) and the legacy RedisQueueService.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import base64

from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest, PDFScrapingStatus
from tilellm.modules.pdf_ocr.services.taskiq_service import TaskiqService, get_taskiq_service
from tilellm.modules.pdf_ocr.services.job_service import PDFJobService, get_job_service


def _create_test_request(
    file_name="test.pdf",
    file_content="https://example.com/test.pdf",  # Default to URL
    namespace="test",
    job_id=None,
    use_docling=True
):
    """Helper to create a valid PDFScrapingRequest with required fields."""
    from tilellm.models.vector_store import Engine
    
    # Create minimal engine config
    engine = Engine(
        name="qdrant",
        host="localhost",
        port=6333,
        deployment="local"
    )
    
    return PDFScrapingRequest(
        id=job_id or "test-job-id",
        file_name=file_name,
        file_content=file_content,
        namespace=namespace,
        engine=engine,
        use_docling=use_docling
    )


class TestTaskiqService:
    """Test TaskiqService functionality."""

    def test_initialization(self):
        """Test TaskiqService initialization."""
        service = TaskiqService()
        assert hasattr(service, 'taskiq_available')
        assert hasattr(service, 'enabled')
        
    @pytest.mark.asyncio
    async def test_submit_with_url(self, mocker):
        """Test submitting a PDF with URL content."""
        mock_task = AsyncMock()
        mock_task.kiq = AsyncMock()
        mocker.patch('tilellm.modules.task_executor.tasks.process_pdf_document_task', mock_task)
        
        service = TaskiqService()
        service.taskiq_available = True
        service.enabled = True
        
        request = _create_test_request(
            file_content="https://example.com/test.pdf",
            job_id="test-job-456"
        )
        
        result = await service.submit(
            job_id="test-job-456",
            request=request
        )
        
        assert result['job_id'] == "test-job-456"
        assert 'message' in result
        assert mock_task.kiq.called
        
    @pytest.mark.asyncio
    async def test_submit_rejects_base64(self):
        """Test that submitting Base64 content raises ValueError."""
        service = TaskiqService()
        service.taskiq_available = True
        service.enabled = True
        
        request = _create_test_request(
            file_content=base64.b64encode(b"fake pdf content").decode('utf-8'),
            job_id="test-job-base64"
        )
        
        with pytest.raises(ValueError, match="must be a URL"):
            await service.submit(job_id="test-job-base64", request=request)
            
    @pytest.mark.asyncio
    async def test_submit_raises_if_not_available(self):
        """Test that submit raises RuntimeError if Taskiq is not available."""
        service = TaskiqService()
        service.taskiq_available = False
        service.enabled = True
        
        request = _create_test_request(job_id="test")
        
        with pytest.raises(RuntimeError, match="Taskiq is not available"):
            await service.submit(job_id="test", request=request)
            
    @pytest.mark.asyncio
    async def test_submit_raises_if_not_enabled(self):
        """Test that submit raises RuntimeError if Taskiq is not enabled."""
        service = TaskiqService()
        service.taskiq_available = True
        service.enabled = False
        
        request = _create_test_request(job_id="test")
        
        with pytest.raises(RuntimeError, match="Taskiq is not enabled"):
            await service.submit(job_id="test", request=request)
            
    @pytest.mark.asyncio
    async def test_process_synchronously(self, mocker):
        """Test synchronous processing fallback."""
        # Mock the processing function where it's used
        mock_process = AsyncMock(return_value={"status": "success", "doc_id": "test"})
        mocker.patch('tilellm.modules.pdf_ocr.logic.process_pdf_document_with_embeddings', mock_process)
        
        service = TaskiqService()
        
        request = _create_test_request(
            file_content="https://example.com/test.pdf",
            job_id="test-job-sync"
        )
        
        result = await service.process_synchronously(request=request)
        
        assert result['job_id'] == "test-job-sync"
        assert 'message' in result
        assert mock_process.called
        
    def test_is_available(self):
        """Test is_available method."""
        service = TaskiqService()
        
        service.taskiq_available = True
        service.enabled = True
        assert service.is_available() == True
        
        service.taskiq_available = False
        service.enabled = True
        assert service.is_available() == False
        
        service.taskiq_available = True
        service.enabled = False
        assert service.is_available() == False


class TestTaskiqServiceSingleton:
    """Test TaskiqService singleton pattern."""
    
    def test_get_taskiq_service_singleton(self):
        """Test that get_taskiq_service returns a singleton."""
        # Clear the singleton
        import tilellm.modules.pdf_ocr.services.taskiq_service as mod
        mod._taskiq_service = None
        
        service1 = get_taskiq_service()
        service2 = get_taskiq_service()
        
        assert service1 is service2


class TestPDFJobService:
    """Test PDFJobService functionality."""

    def test_register_task(self):
        """Test registering a Taskiq task."""
        service = PDFJobService()

        task_id = "taskiq-uuid-1234"
        doc_id = "my-doc-id"
        service.register_task(task_id=task_id, doc_id=doc_id, file_name="test.pdf")

        job = service.get_job(task_id)
        assert job is not None
        assert job.file_name == "test.pdf"
        assert job.status == PDFScrapingStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_status_response(self, mocker):
        """Test getting status response."""
        service = PDFJobService()

        task_id = "taskiq-uuid-5678"
        service.register_task(task_id=task_id, doc_id="doc-abc", file_name="test.pdf")
        job_id = task_id
        
        # Mock result_backend to return not ready (processing)
        mock_result_backend = AsyncMock()
        mock_result_backend.is_result_ready = AsyncMock(return_value=False)
        mocker.patch.object(service, '_get_result_backend', return_value=mock_result_backend)
        
        response = await service.get_status_response(job_id)
        
        assert response is not None
        assert response.job_id == job_id
        assert response.status == PDFScrapingStatus.PROCESSING
        
    @pytest.mark.asyncio
    async def test_get_status_response_not_found(self, mocker):
        """Test getting status response for non-existent job."""
        service = PDFJobService()
        
        # Mock result_backend to be unavailable
        mocker.patch.object(service, '_get_result_backend', return_value=None)
        
        response = await service.get_status_response("non-existent-id")
        
        assert response is None
        
    def test_get_job_not_found(self):
        """Test getting non-existent job."""
        service = PDFJobService()
        
        job = service.get_job("non-existent-id")
        
        assert job is None


class TestPDFJobServiceSingleton:
    """Test PDFJobService singleton pattern."""
    
    def test_get_job_service_singleton(self):
        """Test that get_job_service returns a singleton."""
        # Clear the singleton
        import tilellm.modules.pdf_ocr.services.job_service as mod
        mod._job_service = None
        
        service1 = get_job_service()
        service2 = get_job_service()
        
        assert service1 is service2


class TestRedisQueueServiceMinioError:
    """Test RedisQueueService behavior when MinIO is not available."""

    def test_redis_queue_service_requires_minio(self, mocker):
        """Test that RedisQueueService raises error if MinIO is not available."""
        from minio.error import S3Error
        
        # Mock Minio to raise connection error
        mock_minio_class = Mock()
        mock_minio_instance = Mock()
        mock_minio_instance.bucket_exists.side_effect = S3Error(
            code="ConnectionRefusedError",
            message="Connection refused",
            resource="/ocr-pdfs?location=",
            request_id="test",
            host_id="test",
            response=Mock()
        )
        mock_minio_class.return_value = mock_minio_instance
        
        mocker.patch('tilellm.modules.pdf_ocr.services.redis_queue_service.Minio', mock_minio_class)
        mocker.patch('tilellm.modules.pdf_ocr.services.redis_queue_service.S3Error', S3Error)
        
        from tilellm.modules.pdf_ocr.services.redis_queue_service import RedisQueueService
        
        with pytest.raises(RuntimeError, match="Failed to connect to MinIO"):
            RedisQueueService()
            
    def test_redis_queue_service_minio_not_installed(self, mocker):
        """Test RedisQueueService behavior when Minio package is not installed."""
        # Mock Minio as None (not installed)
        mocker.patch('tilellm.modules.pdf_ocr.services.redis_queue_service.Minio', None)
        
        # Need to reload the module to pick up the mock
        import importlib
        import tilellm.modules.pdf_ocr.services.redis_queue_service as mod
        importlib.reload(mod)
        
        # Should not raise on init, but is_minio_available should return False
        # Note: This test may need adjustment based on exact behavior desired
        

class TestIntegration:
    """Integration tests for the PDF OCR services."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_taskiq_url(self, mocker):
        """Test full workflow: create job -> submit URL to Taskiq -> check status."""
        # Mock the task
        mock_task = AsyncMock()
        mock_task.kiq = AsyncMock()
        mocker.patch('tilellm.modules.task_executor.tasks.process_pdf_document_task', mock_task)
        
        # Clear singletons
        import tilellm.modules.pdf_ocr.services.taskiq_service as taskiq_mod
        import tilellm.modules.pdf_ocr.services.job_service as job_mod
        taskiq_mod._taskiq_service = None
        job_mod._job_service = None
        
        # Get services
        job_service = get_job_service()
        taskiq_service = get_taskiq_service()
        taskiq_service.taskiq_available = True
        taskiq_service.enabled = True
        
        # Create request with URL
        request = _create_test_request(
            file_content="https://example.com/test.pdf",
            job_id=None,
            use_docling=True
        )
        
        # doc_id comes from the request (stable vector store identity)
        doc_id = "integration-test-doc-id"
        request.id = doc_id

        # Submit to Taskiq – returns Taskiq task_id for status polling
        result = await taskiq_service.submit(
            doc_id=doc_id,
            request=request
        )

        assert "task_id" in result
        assert result["doc_id"] == doc_id

        task_id = result["task_id"]
        job_service.register_task(task_id=task_id, doc_id=doc_id, file_name="integration-test.pdf")

        # Check status (async now)
        status = await job_service.get_status_response(task_id)
        assert status is not None
        assert status.job_id == task_id

        # Cleanup singletons
        taskiq_mod._taskiq_service = None
        job_mod._job_service = None
        
    @pytest.mark.asyncio
    async def test_taskiq_rejects_base64_in_workflow(self, mocker):
        """Test that Base64 content is rejected in the workflow."""
        # Clear singletons
        import tilellm.modules.pdf_ocr.services.taskiq_service as taskiq_mod
        import tilellm.modules.pdf_ocr.services.job_service as job_mod
        taskiq_mod._taskiq_service = None
        job_mod._job_service = None
        
        # Get services
        job_service = get_job_service()
        taskiq_service = get_taskiq_service()
        taskiq_service.taskiq_available = True
        taskiq_service.enabled = True
        
        # Create request with Base64 (should be rejected)
        request = _create_test_request(
            file_content=base64.b64encode(b"fake pdf").decode('utf-8'),
            job_id=None,
            use_docling=True
        )
        
        doc_id = "integration-test-doc-id"
        request.id = doc_id

        # Submit should raise ValueError for Base64 content
        with pytest.raises(ValueError, match="must be a URL"):
            await taskiq_service.submit(
                doc_id=doc_id,
                request=request
            )
        
        # Cleanup singletons
        taskiq_mod._taskiq_service = None
        job_mod._job_service = None
