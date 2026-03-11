#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cloud Importer Module for Bespoke Utility
Handles importing data from cloud storage services like AWS S3, Google Cloud Storage, Azure Blob Storage.
"""


import logging
import os
import io
from typing import Dict, Any, Optional, List
import pandas as pd

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

try:
    from google.cloud import storage as gcs
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    gcs = None

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
except ImportError:
    BlobServiceClient = None

logger = logging.getLogger(__name__)


class CloudStorageError(Exception):
    """Custom exception for cloud storage operations."""
    pass


class CloudImporter:
    """
    Class responsible for importing data from cloud storage services.
    Supports AWS S3, Google Cloud Storage, and Azure Blob Storage.
    """

    SUPPORTED_PROVIDERS = ["aws_s3", "gcs", "azure_blob"]
    SUPPORTED_FORMATS = ["csv", "excel", "parquet", "json", "txt"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cloud importer.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.s3_client = None
        self.gcs_client = None
        self.azure_client = None
        logger.info("CloudImporter initialized.")

    def _get_s3_client(self, aws_access_key_id: Optional[str] = None,
                       aws_secret_access_key: Optional[str] = None,
                       region_name: Optional[str] = None):
        """
        Initialize and return an S3 client.
        
        Args:
            aws_access_key_id: AWS Access Key ID
            aws_secret_access_key: AWS Secret Access Key
            region_name: AWS Region name
            
        Returns:
            boto3 S3 client
            
        Raises:
            CloudStorageError: If boto3 is not installed or credentials are invalid
        """
        if boto3 is None:
            raise CloudStorageError("AWS SDK (boto3) not installed. Install with: pip install boto3")
        
        if self.s3_client is None:
            try:
                if aws_access_key_id and aws_secret_access_key:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name or 'us-east-1'
                    )
                else:
                    self.s3_client = boto3.client('s3', region_name=region_name or 'us-east-1')
                
                self.s3_client.list_buckets()
                logger.info("AWS S3 client initialized successfully.")
                
            except NoCredentialsError:
                raise CloudStorageError("AWS credentials not found. Please configure your credentials.")
            except ClientError as e:
                raise CloudStorageError(f"Failed to initialize AWS S3 client: {e}")
            except Exception as e:
                raise CloudStorageError(f"Unexpected error initializing S3 client: {e}")
        
        return self.s3_client

    def _get_gcs_client(self, project: Optional[str] = None, 
                        credentials_path: Optional[str] = None):
        """
        Initialize and return a Google Cloud Storage client.
        
        Args:
            project: GCP Project ID
            credentials_path: Path to service account JSON file
            
        Returns:
            Google Cloud Storage client
            
        Raises:
            CloudStorageError: If google-cloud-storage is not installed or credentials are invalid
        """
        if gcs is None:
            raise CloudStorageError("Google Cloud Storage SDK not installed. Install with: pip install google-cloud-storage")
        
        if self.gcs_client is None:
            try:
                if credentials_path and os.path.exists(credentials_path):
                    self.gcs_client = gcs.Client.from_service_account_json(
                        credentials_path, project=project
                    )
                else:
                    self.gcs_client = gcs.Client(project=project)
                
                list(self.gcs_client.list_buckets(max_results=1))
                logger.info("Google Cloud Storage client initialized successfully.")
                
            except DefaultCredentialsError:
                raise CloudStorageError("GCS credentials not found. Please configure your credentials.")
            except Exception as e:
                raise CloudStorageError(f"Failed to initialize GCS client: {e}")
        
        return self.gcs_client

    def _get_azure_client(self, account_name: str, account_key: Optional[str] = None,
                          connection_string: Optional[str] = None):
        """
        Initialize and return an Azure Blob Storage client.
        
        Args:
            account_name: Azure Storage Account name
            account_key: Azure Storage Account key
            connection_string: Azure Storage connection string
            
        Returns:
            Azure Blob Service client
            
        Raises:
            CloudStorageError: If azure-storage-blob is not installed or credentials are invalid
        """
        if BlobServiceClient is None:
            raise CloudStorageError("Azure Storage SDK not installed. Install with: pip install azure-storage-blob")
        
        if self.azure_client is None:
            try:
                if connection_string:
                    self.azure_client = BlobServiceClient.from_connection_string(connection_string)
                elif account_name and account_key:
                    account_url = f"https://{account_name}.blob.core.windows.net"
                    self.azure_client = BlobServiceClient(
                        account_url=account_url,
                        credential=account_key
                    )
                else:
                    raise CloudStorageError("Either connection_string or account_name + account_key required for Azure")
                
                list(self.azure_client.list_containers(max_results=1))
                logger.info("Azure Blob Storage client initialized successfully.")
                
            except Exception as e:
                raise CloudStorageError(f"Failed to initialize Azure client: {e}")
        
        return self.azure_client

    def list_buckets(self, provider: str, **kwargs) -> List[str]:
        """
        List buckets/containers for the given cloud provider.
        
        Args:
            provider: Cloud provider ('aws_s3', 'gcs', 'azure_blob')
            **kwargs: Provider-specific authentication parameters
            
        Returns:
            List of bucket/container names
            
        Raises:
            CloudStorageError: If operation fails
        """
        provider = provider.lower()
        logger.info(f"Listing buckets for {provider}.")
        
        try:
            if provider == "aws_s3":
                client = self._get_s3_client(**kwargs)
                response = client.list_buckets()
                return [bucket['Name'] for bucket in response.get('Buckets', [])]
            
            elif provider == "gcs":
                client = self._get_gcs_client(**kwargs)
                buckets = client.list_buckets()
                return [bucket.name for bucket in buckets]
            
            elif provider == "azure_blob":
                client = self._get_azure_client(**kwargs)
                containers = client.list_containers()
                return [container.name for container in containers]
            
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error listing buckets for {provider}: {e}", exc_info=True)
            raise CloudStorageError(f"Could not list buckets for {provider}: {e}")

    def list_files(self, provider: str, bucket_name: str, prefix: str = "", 
                   max_files: int = 1000, **kwargs) -> List[Dict[str, Any]]:
        """
        List files in a specific bucket/container with optional prefix.
        
        Args:
            provider: Cloud provider ('aws_s3', 'gcs', 'azure_blob')
            bucket_name: Name of the bucket/container
            prefix: File path prefix to filter by
            max_files: Maximum number of files to return
            **kwargs: Provider-specific authentication parameters
            
        Returns:
            List of file information dictionaries
            
        Raises:
            CloudStorageError: If operation fails
        """
        provider = provider.lower()
        logger.info(f"Listing files in {provider} bucket '{bucket_name}' with prefix '{prefix}'.")
        
        try:
            files = []
            
            if provider == "aws_s3":
                client = self._get_s3_client(**kwargs)
                paginator = client.get_paginator('list_objects_v2')
                
                count = 0
                for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        if count >= max_files:
                            break
                        files.append({
                            'name': obj['Key'],
                            'size': obj['Size'],
                            'modified': obj['LastModified'],
                            'etag': obj.get('ETag', '').strip('\"')
                        })
                        count += 1
                    if count >= max_files:
                        break
            
            elif provider == "gcs":
                client = self._get_gcs_client(**kwargs)
                bucket = client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=prefix, max_results=max_files)
                
                for blob in blobs:
                    files.append({
                        'name': blob.name,
                        'size': blob.size,
                        'modified': blob.updated,
                        'etag': blob.etag
                    })
            
            elif provider == "azure_blob":
                client = self._get_azure_client(**kwargs)
                container_client = client.get_container_client(bucket_name)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                
                count = 0
                for blob in blobs:
                    if count >= max_files:
                        break
                    files.append({
                        'name': blob.name,
                        'size': blob.size,
                        'modified': blob.last_modified,
                        'etag': blob.etag
                    })
                    count += 1
            
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {provider} bucket '{bucket_name}': {e}", exc_info=True)
            raise CloudStorageError(f"Could not list files for {provider} bucket '{bucket_name}': {e}")

    def download_file_to_dataframe(self, provider: str, bucket_name: str, file_key: str,
                                   file_format: str = 'csv', 
                                   pandas_read_options: Optional[Dict] = None,
                                   **kwargs) -> pd.DataFrame:
        """
        Download a file from cloud storage and load it into a pandas DataFrame.
        
        Args:
            provider: Cloud provider ('aws_s3', 'gcs', 'azure_blob')
            bucket_name: Name of the bucket/container
            file_key: The key/path of the file within the bucket
            file_format: Format of the file ('csv', 'excel', 'parquet', 'json', 'txt')
            pandas_read_options: Dictionary of options to pass to pandas read function
            **kwargs: Provider-specific authentication parameters
            
        Returns:
            pandas DataFrame
            
        Raises:
            CloudStorageError: If operation fails
        """
        provider = provider.lower()
        pandas_read_options = pandas_read_options or {}
        
        if file_format not in self.SUPPORTED_FORMATS:
            raise CloudStorageError(f"Unsupported file format: {file_format}. Supported: {self.SUPPORTED_FORMATS}")
        
        logger.info(f"Downloading '{file_key}' from {provider} bucket '{bucket_name}' as {file_format}.")
        
        try:
            file_content_bytes = None
            
            if provider == "aws_s3":
                client = self._get_s3_client(**kwargs)
                response = client.get_object(Bucket=bucket_name, Key=file_key)
                file_content_bytes = response['Body'].read()
            
            elif provider == "gcs":
                client = self._get_gcs_client(**kwargs)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(file_key)
                file_content_bytes = blob.download_as_bytes()
            
            elif provider == "azure_blob":
                client = self._get_azure_client(**kwargs)
                blob_client = client.get_blob_client(container=bucket_name, blob=file_key)
                file_content_bytes = blob_client.download_blob().readall()
            
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            
            if file_content_bytes is None:
                raise CloudStorageError("Failed to download file content")
            
            file_like_object = io.BytesIO(file_content_bytes)
            
            if file_format == 'csv' or file_format == 'txt':
                df = pd.read_csv(file_like_object, **pandas_read_options)
            
            elif file_format == 'excel':
                engine = None
                if file_key.lower().endswith('.xlsx'):
                    engine = 'openpyxl'
                elif file_key.lower().endswith('.xls'):
                    engine = 'xlrd'
                
                if engine:
                    pandas_read_options['engine'] = engine
                
                df = pd.read_excel(file_like_object, **pandas_read_options)
            
            elif file_format == 'parquet':
                df = pd.read_parquet(file_like_object, **pandas_read_options)
            
            elif file_format == 'json':
                df = pd.read_json(file_like_object, **pandas_read_options)
            
            else:
                raise CloudStorageError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded '{file_key}' into DataFrame ({len(df)} rows, {len(df.columns)} columns).")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading/loading file '{file_key}' from {provider} bucket '{bucket_name}': {e}", exc_info=True)
            raise CloudStorageError(f"Could not download/load '{file_key}': {e}")

    def get_file_info(self, provider: str, bucket_name: str, file_key: str, **kwargs) -> Dict[str, Any]:
        """
        Get information about a specific file.
        
        Args:
            provider: Cloud provider ('aws_s3', 'gcs', 'azure_blob')
            bucket_name: Name of the bucket/container
            file_key: The key/path of the file within the bucket
            **kwargs: Provider-specific authentication parameters
            
        Returns:
            Dictionary containing file information
            
        Raises:
            CloudStorageError: If operation fails
        """
        provider = provider.lower()
        logger.info(f"Getting info for '{file_key}' from {provider} bucket '{bucket_name}'.")
        
        try:
            if provider == "aws_s3":
                client = self._get_s3_client(**kwargs)
                response = client.head_object(Bucket=bucket_name, Key=file_key)
                return {
                    'name': file_key,
                    'size': response['ContentLength'],
                    'modified': response['LastModified'],
                    'content_type': response.get('ContentType', ''),
                    'etag': response.get('ETag', '').strip('\"')
                }
            
            elif provider == "gcs":
                client = self._get_gcs_client(**kwargs)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(file_key)
                blob.reload()  # Fetch metadata
                return {
                    'name': blob.name,
                    'size': blob.size,
                    'modified': blob.updated,
                    'content_type': blob.content_type,
                    'etag': blob.etag
                }
            
            elif provider == "azure_blob":
                client = self._get_azure_client(**kwargs)
                blob_client = client.get_blob_client(container=bucket_name, blob=file_key)
                properties = blob_client.get_blob_properties()
                return {
                    'name': file_key,
                    'size': properties.size,
                    'modified': properties.last_modified,
                    'content_type': properties.content_settings.content_type,
                    'etag': properties.etag
                }
            
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error getting info for file '{file_key}' from {provider} bucket '{bucket_name}': {e}", exc_info=True)
            raise CloudStorageError(f"Could not get info for '{file_key}': {e}")

    def test_connection(self, provider: str, **kwargs) -> bool:
        """
        Test connection to the cloud provider.
        
        Args:
            provider: Cloud provider ('aws_s3', 'gcs', 'azure_blob')
            **kwargs: Provider-specific authentication parameters
            
        Returns:
            True if connection successful, False otherwise
        """
        provider = provider.lower()
        logger.info(f"Testing connection to {provider}.")
        
        try:
            if provider == "aws_s3":
                client = self._get_s3_client(**kwargs)
                client.list_buckets()
            
            elif provider == "gcs":
                client = self._get_gcs_client(**kwargs)
                list(client.list_buckets(max_results=1))
            
            elif provider == "azure_blob":
                client = self._get_azure_client(**kwargs)
                list(client.list_containers(max_results=1))
            
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
            
            logger.info(f"Connection to {provider} successful.")
            return True
            
        except Exception as e:
            logger.warning(f"Connection to {provider} failed: {e}")
            return False

    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported cloud providers."""
        return CloudImporter.SUPPORTED_PROVIDERS.copy()

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported file formats."""
        return CloudImporter.SUPPORTED_FORMATS.copy()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    config = {}
    importer = CloudImporter(config)
    
    try:
        if importer.test_connection("aws_s3"):
            buckets = importer.list_buckets("aws_s3")
            print("S3 Buckets:", buckets)
            
            if buckets:
                files = importer.list_files("aws_s3", buckets[0], max_files=5)
                print(f"Files in {buckets[0]}:", [f['name'] for f in files])
                
    except CloudStorageError as e:
        print(f"S3 Error: {e}")
    
    try:
        if importer.test_connection("gcs", project="your-project-id"):
            buckets = importer.list_buckets("gcs", project="your-project-id")
            print("GCS Buckets:", buckets)
            
    except CloudStorageError as e:
        print(f"GCS Error: {e}")
    
    try:
        if importer.test_connection("azure_blob", account_name="your-account", account_key="your-key"):
            containers = importer.list_buckets("azure_blob", account_name="your-account", account_key="your-key")
            print("Azure Containers:", containers)
            
    except CloudStorageError as e:
        print(f"Azure Error: {e}")