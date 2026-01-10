"""
Service to enqueue PDF processing jobs into a Redis queue after uploading the source file to MinIO.
"""

import os
import base64
import json
import redis.asyncio as redis
import requests
import io
try:
    from minio import Minio
except ImportError:
    Minio = None
from minio.error import S3Error

from tilellm.modules.pdf_ocr.models.pdf_scraping import PDFScrapingRequest
from tilellm.shared.utility import get_service_config

class RedisQueueService:
    """
    Handles PDF preparation, uploads to MinIO, and enqueues jobs for the external OCR worker.
    """
    def __init__(self):
        config = get_service_config()
        
        # Redis Config
        redis_conf = config.get("redis", {})
        self.redis_host = redis_conf.get("host", "localhost")
        self.redis_port = int(redis_conf.get("port", 6379))
        self.redis_db = int(redis_conf.get("db", 0))
        self.ocr_queue_name = redis_conf.get("queue_name", "tiledesk_ocr_queue")
        
        # MinIO Config
        minio_conf = config.get("minio", {})
        self.minio_endpoint = minio_conf.get("endpoint", "localhost:9000")
        self.minio_access = minio_conf.get("access_key", "minioadmin")
        self.minio_secret = minio_conf.get("secret_key", "minioadmin")
        self.minio_secure = minio_conf.get("secure", False)
        # Bucket for raw PDFs
        self.minio_bucket = minio_conf.get("bucket_pdfs", "tiledesk-ocr-pdfs")

        # 1. Redis Client (Async)
        # We don't connect immediately in init to avoid async issues in constructor
        self.redis_client = redis.Redis(
            host=self.redis_host, 
            port=self.redis_port, 
            db=self.redis_db, 
            decode_responses=True
        )

        # 2. Connect to MinIO (Sync is fine for now, or could use minio_storage_service)
        try:
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access,
                secret_key=self.minio_secret,
                secure=self.minio_secure
            )
            # Ensure the bucket exists
            found = self.minio_client.bucket_exists(self.minio_bucket)
            if not found:
                self.minio_client.make_bucket(self.minio_bucket)
                print(f"Created MinIO bucket: {self.minio_bucket}")
            else:
                print(f"MinIO bucket '{self.minio_bucket}' already exists.")
        except S3Error as e:
            print(f"FATAL: Could not connect to MinIO at {self.minio_endpoint}. Error: {e}")
            raise RuntimeError("Failed to connect to MinIO") from e

    async def submit(self, job_id: str, request: PDFScrapingRequest):
        """
        Prepares the PDF file, uploads it to MinIO, and submits the job to the Redis queue.

        Args:
            job_id: The unique ID for this job.
            request: The original scraping request.
        """
        try:
            # 1. Get PDF content as bytes
            if request.is_url():
                response = requests.get(request.file_content, timeout=30)
                response.raise_for_status()
                pdf_content = response.content
            else:
                pdf_content = base64.b64decode(request.file_content)

            # 2. Upload the file to MinIO
            object_name = f"{job_id}.pdf"
            self._upload_file_to_minio(object_name, pdf_content)

            # 3. Create the task payload for Redis
            task_payload = {
                "task_id": job_id,
                "object_name": object_name,
                "bucket_name": self.minio_bucket,
                "original_filename": request.file_name,
                "webhook_url": request.webhook_url,
                "callback_token": request.callback_token
            }

            if request.unstructured_config:
                task_payload["unstructured_config"] = request.unstructured_config

            # 4. Push the job to the Redis queue
            await self.redis_client.lpush(self.ocr_queue_name, json.dumps(task_payload))
            
            print(f"Successfully queued job {job_id} for object {object_name} in bucket {self.minio_bucket}.")

        except Exception as e:
            print(f"Error submitting job {job_id}: {e}")
            raise

    def _upload_file_to_minio(self, object_name: str, content: bytes):
        """
        Uploads file content to the configured MinIO bucket.
        """
        try:
            self.minio_client.put_object(
                bucket_name=self.minio_bucket,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type='application/pdf'
            )
            print(f"Successfully uploaded {object_name} to MinIO bucket {self.minio_bucket}.")
        except S3Error as e:
            print(f"MinIO upload failed for {object_name}. Error: {e}")
            raise RuntimeError("Failed to upload file to MinIO") from e


# --- Singleton instance ---
_redis_queue_service: RedisQueueService = None

def get_redis_queue_service() -> RedisQueueService:
    """Get or create a singleton instance of the RedisQueueService."""
    global _redis_queue_service
    if _redis_queue_service is None:
        _redis_queue_service = RedisQueueService()
    return _redis_queue_service
