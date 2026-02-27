"""
MinIO Storage Service for GraphRAG artifacts.
Handles storage and retrieval of Parquet files (community reports, entities, relationships)
from MinIO S3-compatible storage.
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO
from datetime import datetime

try:
    from minio import Minio
except ImportError:
    Minio = None
from minio.error import S3Error

from tilellm.shared.utility import get_service_config

logger = logging.getLogger(__name__)

# Default artifact paths
ARTIFACTS_PREFIX = "artifacts/"
RAW_DOCUMENTS_PREFIX = "raw-documents/"
METADATA_PREFIX = "metadata/"


class MinIOStorageService:
    """
    Service for storing and retrieving GraphRAG artifacts from MinIO.
    
    Organizes files in MinIO with the following structure:
    
    graphrag/
    ├── raw-documents/          # Original PDF/TXT files
    ├── artifacts/              # Pipeline outputs (Parquet files)
    │   ├── {namespace}/
    │   │   ├── {timestamp}/
    │   │   │   ├── community_reports.parquet
    │   │   │   ├── entities.parquet
    │   │   │   ├── relationships.parquet
    │   │   │   ├── communities.parquet
    │   │   │   └── text_units.parquet
    │   │   └── latest -> symlink to latest timestamp
    └── metadata/               # Configurations and logs
    """
    
    def __init__(self):
        """
        Initialize MinIO storage service using environment variables.
        """
        if Minio is None:
            raise ImportError("Minio client not installed. Please install with 'poetry install -E graph'")
        
        service_config = get_service_config()
        minio_config = service_config.get("minio")
        
        if not minio_config:
            raise ValueError("MinIO configuration not found in environment variables")
        
        self.endpoint = minio_config.get("endpoint")
        self.access_key = minio_config.get("access_key")
        self.secret_key = minio_config.get("secret_key")
        self.secure = minio_config.get("secure", False)
        self.bucket_name = minio_config.get("bucket_name", "graphrag") # Default bucket name if not specified
        self.bucket_tables = minio_config.get("bucket_tables", "document-tables")
        self.bucket_images = minio_config.get("bucket_images", "document-images")
        self.bucket_pdfs = minio_config.get("bucket_pdfs", "tiledesk-ocr-pdfs")
        
        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError("endpoint, access_key, and secret_key are required for MinIO configuration")
        
        self._client: Optional[Minio] = None
        # Lazy initialization - don't connect immediately
    
    def _initialize(self):
        """Initialize MinIO client and ensure bucket exists."""
        try:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"MinIO bucket '{self.bucket_name}' already exists.")
                
        except Exception as e:
            logger.warning(f"Failed to connect to MinIO at {self.endpoint}: {e}. MinIO storage will be disabled.")
            self._client = None
    
    @property
    def client(self):
        """Get MinIO client, ensuring it's initialized."""
        if self._client is None:
            self._initialize()
            if self._client is None:
                raise RuntimeError("MinIO client is not available. Check MinIO connection and configuration.")
        return self._client
    
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str = 'application/octet-stream'
    ) -> str:
        """Upload a file to a specific bucket."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type
            )
            logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Failed to upload to {bucket_name}: {e}")
            raise

    def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = 'application/octet-stream'
    ) -> str:
        """Upload raw data to a specific bucket."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            
            data_stream = io.BytesIO(data)
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )
            logger.info(f"Uploaded data to {bucket_name}/{object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Failed to upload data to {bucket_name}: {e}")
            raise

    def get_object(self, bucket_name: str, object_name: str):
        """Get an object from a specific bucket."""
        try:
            return self.client.get_object(bucket_name, object_name)
        except S3Error as e:
            logger.error(f"Failed to get object {object_name} from {bucket_name}: {e}")
            raise

    def upload_parquet_file(
        self,
        namespace: str,
        file_name: str,
        file_path: str,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> str:
        """
        Upload a Parquet file to MinIO.
        
        Args:
            namespace: Namespace/collection name
            file_name: Name of the file (e.g., 'community_reports.parquet')
            file_path: Local path to the Parquet file
            timestamp: Optional timestamp for versioning (default: current UTC)
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Object key/path in MinIO
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Construct object key - require index_name and index_type
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        # New structure: {index_name}/{index_type}/{namespace}/{timestamp}/{file_name}
        object_key = f"{index_name}/{index_type}/{namespace}/{timestamp}/{file_name}"
        
        try:
            # Upload file
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                file_path=file_path,
                content_type='application/octet-stream'
            )
            
            logger.info(f"Uploaded {file_name} to MinIO: {object_key}")
            
            return object_key
            
        except S3Error as e:
            logger.error(f"Failed to upload {file_name} to MinIO: {e}")
            raise
    
    def upload_parquet_data(
        self,
        namespace: str,
        file_name: str,
        data: bytes,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> str:
        """
        Upload Parquet data (bytes) to MinIO.
        
        Args:
            namespace: Namespace/collection name
            file_name: Name of the file
            data: Parquet file data as bytes
            timestamp: Optional timestamp
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Object key/path in MinIO
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Construct object key - require index_name and index_type
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        # New structure: {index_name}/{index_type}/{namespace}/{timestamp}/{file_name}
        object_key = f"{index_name}/{index_type}/{namespace}/{timestamp}/{file_name}"
        
        try:
            data_stream = io.BytesIO(data)
            data_stream.seek(0)
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                data=data_stream,
                length=len(data),
                content_type='application/octet-stream'
            )
            
            logger.info(f"Uploaded {file_name} data to MinIO: {object_key}")
            
            return object_key
            
        except S3Error as e:
            logger.error(f"Failed to upload {file_name} data to MinIO: {e}")
            raise
    
    def download_parquet_file(
        self,
        namespace: str,
        file_name: str,
        timestamp: Optional[str] = None,
        local_path: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> str:
        """
        Download a Parquet file from MinIO to local filesystem.
        
        Args:
            namespace: Namespace/collection name
            file_name: Name of the file to download
            timestamp: Optional timestamp (default: 'latest')
            local_path: Optional local path to save file
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Local path to downloaded file
        """
        if timestamp is None:
            timestamp = "latest"
        
        # Require index_name and index_type for new structure
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        
        # Resolve 'latest' timestamp if needed
        if timestamp == "latest":
            resolved_timestamp = self._get_latest_timestamp_for_index(namespace, index_name, index_type)
            if resolved_timestamp is None:
                raise FileNotFoundError(f"No timestamped artifacts found for namespace '{namespace}' with index {index_name}/{index_type}")
            timestamp = resolved_timestamp
        
        # Construct object key
        object_key = f"{index_name}/{index_type}/{namespace}/{timestamp}/{file_name}"
        
        if local_path is None:
            # Create temporary file path
            import tempfile
            local_path = tempfile.mktemp(suffix=f"_{file_name}")
        
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                file_path=local_path
            )
            logger.info(f"Downloaded {file_name} from MinIO to {local_path} (key: {object_key})")
            return local_path
        except S3Error as e:
            logger.error(f"Failed to download {file_name} from MinIO: {e}")
            raise
    
    def download_parquet_data(
        self,
        namespace: str,
        file_name: str,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> bytes:
        """
        Download Parquet file data as bytes.
        
        Args:
            namespace: Namespace/collection name
            file_name: Name of the file
            timestamp: Optional timestamp (default: 'latest')
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            File data as bytes
        """
        if timestamp is None:
            timestamp = "latest"
        
        # Require index_name and index_type for new structure
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        
        # Resolve 'latest' timestamp if needed
        if timestamp == "latest":
            resolved_timestamp = self._get_latest_timestamp_for_index(namespace, index_name, index_type)
            if resolved_timestamp is None:
                raise FileNotFoundError(f"No timestamped artifacts found for namespace '{namespace}' with index {index_name}/{index_type}")
            timestamp = resolved_timestamp
        
        # Construct object key
        object_key = f"{index_name}/{index_type}/{namespace}/{timestamp}/{file_name}"
        
        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_key
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"Downloaded {file_name} data from MinIO ({len(data)} bytes) (key: {object_key})")
            return data
        except S3Error as e:
            logger.error(f"Failed to download {file_name} data from MinIO: {e}")
            raise
    
    def list_artifacts(
        self,
        namespace: str,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all artifacts for a namespace.
        
        Args:
            namespace: Namespace/collection name
            timestamp: Optional timestamp (default: 'latest')
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            List of artifact metadata dictionaries
        """
        if timestamp is None:
            timestamp = "latest"
        
        # Require index_name and index_type for new structure
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        
        # Resolve 'latest' timestamp if needed
        if timestamp == "latest":
            resolved_timestamp = self._get_latest_timestamp_for_index(namespace, index_name, index_type)
            if resolved_timestamp is None:
                return []  # No artifacts found
            timestamp = resolved_timestamp
        
        prefix = f"{index_name}/{index_type}/{namespace}/{timestamp}/"
        
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            artifacts = []
            for obj in objects:
                artifacts.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                })
            
            return artifacts
            
        except S3Error as e:
            logger.error(f"Failed to list artifacts for {namespace}: {e}")
            raise
    
    def get_latest_timestamp(self, namespace: str, index_name: Optional[str] = None, index_type: Optional[str] = None) -> Optional[str]:
        """
        Get the latest timestamp for a namespace.
        
        Args:
            namespace: Namespace/collection name
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Latest timestamp string or None if no artifacts exist
        """
        # Require index_name and index_type for new structure
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        
        return self._get_latest_timestamp_for_index(namespace, index_name, index_type)
    
    def _get_latest_timestamp_for_index(self, namespace: str, index_name: str, index_type: str) -> Optional[str]:
        """
        Get the latest timestamp for a namespace using the new index structure.
        
        Args:
            namespace: Namespace/collection name
            index_name: Name of the vector index (e.g., 'tilellm')
            index_type: Type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Latest timestamp string or None if no artifacts exist
        """
        prefix = f"{index_name}/{index_type}/{namespace}/"
        
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=False
            )
            
            timestamps = []
            for obj in objects:
                if obj is None:
                    continue
                obj_name = obj.object_name
                if obj_name is None:
                    continue
                # Extract timestamp from object key
                parts = obj_name.split('/')
                if len(parts) >= 4:
                    timestamp = parts[3]  # {index_name}/{index_type}/{namespace}/{timestamp}/
                    if timestamp != "latest" and timestamp not in timestamps:
                        timestamps.append(timestamp)
            
            if not timestamps:
                return None
            
            # Sort timestamps descending (newest first)
            timestamps.sort(reverse=True)
            return timestamps[0]
            
        except S3Error as e:
            logger.error(f"Failed to get latest timestamp for {namespace} (index {index_name}/{index_type}): {e}")
            return None
    

    
    def upload_graphrag_outputs(
        self,
        namespace: str,
        output_dir: str,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Upload all GraphRAG output files from a directory to MinIO.
        
        Args:
            namespace: Namespace/collection name
            output_dir: Local directory containing GraphRAG output files
            timestamp: Optional timestamp
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Dictionary mapping file names to MinIO object keys
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        uploaded_files = {}
        output_path = Path(output_dir)
        
        # List of expected GraphRAG output files
        expected_files = [
            "community_reports.parquet",
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "text_units.parquet"
        ]
        
        for file_name in expected_files:
            file_path = output_path / file_name
            if file_path.exists():
                try:
                    object_key = self.upload_parquet_file(
                        namespace=namespace,
                        file_name=file_name,
                        file_path=str(file_path),
                        timestamp=timestamp,
                        index_name=index_name,
                        index_type=index_type
                    )
                    uploaded_files[file_name] = object_key
                except Exception as e:
                    logger.error(f"Failed to upload {file_name}: {e}")
                    # Continue with other files
        
        logger.info(f"Uploaded {len(uploaded_files)} GraphRAG files to MinIO for namespace '{namespace}'")
        return uploaded_files
    
    def download_graphrag_outputs(
        self,
        namespace: str,
        timestamp: Optional[str] = None,
        local_dir: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Download all GraphRAG output files from MinIO to local directory.
        
        Args:
            namespace: Namespace/collection name
            timestamp: Optional timestamp (default: 'latest')
            local_dir: Optional local directory to save files
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Dictionary mapping file names to local file paths
        """
        if timestamp is None:
            timestamp = "latest"
        
        if local_dir is None:
            import tempfile
            local_dir = tempfile.mkdtemp(prefix=f"graphrag_{namespace}_")
        
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = {}
        
        # List of expected files
        expected_files = [
            "community_reports.parquet",
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "text_units.parquet"
        ]
        
        for file_name in expected_files:
            try:
                local_path = self.download_parquet_file(
                    namespace=namespace,
                    file_name=file_name,
                    timestamp=timestamp,
                    local_path=str(local_dir_path / file_name),
                    index_name=index_name,
                    index_type=index_type
                )
                downloaded_files[file_name] = local_path
            except Exception as e:
                logger.warning(f"Failed to download {file_name}: {e}")
                # Continue with other files
        
        logger.info(f"Downloaded {len(downloaded_files)} GraphRAG files from MinIO for namespace '{namespace}'")
        return downloaded_files
    
    def delete_artifacts(
        self,
        namespace: str,
        timestamp: Optional[str] = None,
        index_name: Optional[str] = None,
        index_type: Optional[str] = None
    ) -> int:
        """
        Delete artifacts for a namespace (or specific timestamp).
        
        Args:
            namespace: Namespace/collection name
            timestamp: Optional timestamp (if None, delete all timestamps)
            index_name: Optional name of the vector index (e.g., 'tilellm')
            index_type: Optional type of index ('serverless', 'pod', 'local', 'cloud')
            
        Returns:
            Number of objects deleted
        """
        # Require index_name and index_type for new structure
        if not index_name or not index_type:
            raise ValueError("index_name and index_type are required for MinIO storage")
        
        if timestamp:
            prefix = f"{index_name}/{index_type}/{namespace}/{timestamp}/"
        else:
            prefix = f"{index_name}/{index_type}/{namespace}/"
        
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            deleted_count = 0
            for obj in objects:
                if obj is None:
                    continue
                obj_name = obj.object_name
                assert obj_name is not None, "object_name should not be None"
                self.client.remove_object(self.bucket_name, obj_name)
                deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} artifacts for namespace '{namespace}'")
            return deleted_count
            
        except S3Error as e:
            logger.error(f"Failed to delete artifacts for {namespace}: {e}")
            raise


# Singleton instance
_minio_storage_service: Optional[MinIOStorageService] = None

def get_minio_storage_service() -> MinIOStorageService:
    """Get or create a singleton instance of MinIOStorageService."""
    global _minio_storage_service
    if _minio_storage_service is None:
        _minio_storage_service = MinIOStorageService()
    return _minio_storage_service