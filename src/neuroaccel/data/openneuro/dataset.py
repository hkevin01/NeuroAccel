"""OpenNeuro dataset integration module."""
from typing import Optional, List, Dict, Union
import os
import json
import logging
from pathlib import Path
import tempfile
import hashlib
import asyncio
import aiohttp
import aiofiles
import boto3
import datalad.api as dl
from bids import BIDSLayout

logger = logging.getLogger(__name__)

class OpenNeuroDataset:
    """Interface for downloading and managing OpenNeuro datasets."""

    def __init__(
        self,
        dataset_id: str,
        cache_dir: Optional[str] = None,
        use_aws: bool = True
    ):
        """Initialize OpenNeuro dataset interface.

        Args:
            dataset_id: OpenNeuro dataset ID (e.g., 'ds000001')
            cache_dir: Directory to cache downloaded data
            use_aws: Whether to use AWS S3 for faster downloads
        """
        self.dataset_id = dataset_id
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(),
            'neuroaccel',
            'openneuro'
        )
        self.use_aws = use_aws

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize AWS client if needed
        self.s3_client = boto3.client('s3') if use_aws else None

        # Dataset metadata
        self.metadata = None
        self.layout = None

    async def download_metadata(self) -> Dict:
        """Download dataset metadata from OpenNeuro API.

        Returns:
            Dataset metadata
        """
        async with aiohttp.ClientSession() as session:
            url = f"https://openneuro.org/crn/datasets/{self.dataset_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    self.metadata = await response.json()
                    return self.metadata
                else:
                    raise RuntimeError(
                        f"Failed to fetch metadata for dataset {self.dataset_id}"
                    )

    def get_file_list(self) -> List[str]:
        """Get list of files in the dataset.

        Returns:
            List of file paths
        """
        if self.metadata is None:
            raise RuntimeError("Must download metadata first")

        return [f["filename"] for f in self.metadata["files"]]

    async def download_files(
        self,
        file_patterns: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None
    ) -> List[str]:
        """Download dataset files.

        Args:
            file_patterns: List of file patterns to download
            modalities: List of BIDS modalities to download

        Returns:
            List of downloaded file paths
        """
        if self.use_aws:
            return await self._download_from_aws(file_patterns, modalities)
        else:
            return await self._download_from_openneuro(file_patterns, modalities)

    async def _download_from_aws(
        self,
        file_patterns: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None
    ) -> List[str]:
        """Download files from AWS S3.

        Args:
            file_patterns: List of file patterns to download
            modalities: List of BIDS modalities to download

        Returns:
            List of downloaded file paths
        """
        bucket = "openneuro"
        prefix = f"{self.dataset_id}/"

        # List objects in bucket
        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        files_to_download = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            # Filter by patterns and modalities
            if (file_patterns and not any(pattern in key for pattern in file_patterns)) or \
               (modalities and not any(f"_{mod}." in key for mod in modalities)):
                continue
            files_to_download.append(key)

        # Download files in parallel
        downloaded = []
        async def download_file(key: str):
            local_path = os.path.join(self.cache_dir, key.replace(prefix, ''))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if not os.path.exists(local_path):
                # Use aiohttp to download file asynchronously
                async with aiohttp.ClientSession() as session:
                    url = f"https://s3.amazonaws.com/{bucket}/{key}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            async with aiofiles.open(local_path, 'wb') as f:
                                await f.write(content)
            downloaded.append(local_path)

        # Download files concurrently
        await asyncio.gather(
            *[download_file(key) for key in files_to_download]
        )

        return downloaded

    async def _download_from_openneuro(
        self,
        file_patterns: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None
    ) -> List[str]:
        """Download files directly from OpenNeuro.

        Args:
            file_patterns: List of file patterns to download
            modalities: List of BIDS modalities to download

        Returns:
            List of downloaded file paths
        """
        dataset_path = os.path.join(self.cache_dir, self.dataset_id)

        async with aiohttp.ClientSession() as session:
            # Get repository metadata
            url = f"https://api.github.com/repos/OpenNeuroDatasets/{self.dataset_id}/git/trees/master?recursive=1"
            async with session.get(url) as response:
                if response.status == 200:
                    tree = await response.json()
                else:
                    raise RuntimeError(f"Failed to fetch repository tree: {response.status}")

            # Filter files
            files_to_download = []
            for item in tree['tree']:
                if item['type'] != 'blob':
                    continue

                path = item['path']
                if file_patterns and not any(pattern in path for pattern in file_patterns):
                    continue
                if modalities and not any(f"_{mod}." in path for mod in modalities):
                    continue

                files_to_download.append({
                    'path': path,
                    'url': item['url']
                })

            # Download files
            downloaded = []
            async def download_file(file_info):
                local_path = os.path.join(dataset_path, file_info['path'])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                if not os.path.exists(local_path):
                    async with session.get(file_info['url']) as response:
                        if response.status == 200:
                            content = await response.json()
                            # Decode base64 content
                            import base64
                            file_content = base64.b64decode(content['content'])
                            async with aiofiles.open(local_path, 'wb') as f:
                                await f.write(file_content)

                downloaded.append(local_path)

            await asyncio.gather(
                *[download_file(file_info) for file_info in files_to_download]
            )

        return downloaded

    def create_bids_layout(self) -> BIDSLayout:
        """Create BIDS layout from downloaded data.

        Returns:
            BIDS layout object
        """
        dataset_path = os.path.join(self.cache_dir, self.dataset_id)
        if not os.path.exists(dataset_path):
            raise RuntimeError("Dataset must be downloaded first")

        self.layout = BIDSLayout(dataset_path)
        return self.layout

    def get_task_data(
        self,
        task: str,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        runs: Optional[List[str]] = None
    ) -> List[str]:
        """Get paths to task data files.

        Args:
            task: Task name
            subjects: List of subject IDs
            sessions: List of session IDs
            runs: List of run numbers

        Returns:
            List of file paths
        """
        if self.layout is None:
            self.create_bids_layout()

        return self.layout.get(
            task=task,
            subject=subjects,
            session=sessions,
            run=runs,
            extension=['.nii', '.nii.gz'],
            return_type='filename'
        )

    def validate_dataset(self) -> Dict[str, List[str]]:
        """Validate downloaded dataset against BIDS specification.

        Returns:
            Dictionary of validation issues
        """
        from bids_validator import BIDSValidator

        dataset_path = os.path.join(self.cache_dir, self.dataset_id)
        if not os.path.exists(dataset_path):
            raise RuntimeError("Dataset must be downloaded first")

        validator = BIDSValidator()
        is_valid = validator.is_bids(dataset_path)

        if not is_valid:
            issues = validator.get_issues()
            return {
                'errors': issues.get('errors', []),
                'warnings': issues.get('warnings', [])
            }

        return {'errors': [], 'warnings': []}
