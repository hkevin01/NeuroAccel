"""Data loading utilities for neuroimaging data formats."""
from pathlib import Path
from typing import Union, Optional, List, Dict
import warnings

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pydicom
from bids import BIDSLayout

class NeuroDataset(Dataset):
    """Base dataset class for neuroimaging data."""

    def __init__(
        self,
        data_path: Union[str, Path],
        format_type: str = "nifti",
        transform=None,
        target_transform=None
    ):
        """Initialize dataset.

        Args:
            data_path: Path to data directory
            format_type: Data format ("nifti", "dicom", or "bids")
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
        """
        self.data_path = Path(data_path)
        self.format_type = format_type.lower()
        self.transform = transform
        self.target_transform = target_transform

        self.file_list = self._load_file_list()
        self.metadata = self._load_metadata()

    def _load_file_list(self) -> List[Path]:
        """Load list of data files based on format type."""
        if self.format_type == "nifti":
            return list(self.data_path.glob("**/*.nii*"))
        elif self.format_type == "dicom":
            return list(self.data_path.glob("**/*.dcm"))
        elif self.format_type == "bids":
            layout = BIDSLayout(self.data_path)
            return [Path(f) for f in layout.get(extension=[".nii", ".nii.gz"])]
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}")

    def _load_metadata(self) -> Dict:
        """Load metadata for the dataset."""
        metadata = {}
        if self.format_type == "bids":
            try:
                layout = BIDSLayout(self.data_path)
                metadata["subjects"] = layout.get_subjects()
                metadata["sessions"] = layout.get_sessions()
                metadata["tasks"] = layout.get_tasks()
            except Exception as e:
                warnings.warn(f"Error loading BIDS metadata: {e}")
        return metadata

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to fetch

        Returns:
            dict: Sample data including image and metadata
        """
        file_path = self.file_list[idx]

        if self.format_type == "nifti":
            data = self._load_nifti(file_path)
        elif self.format_type == "dicom":
            data = self._load_dicom(file_path)
        else:  # BIDS
            data = self._load_nifti(file_path)  # BIDS uses NIfTI format

        if self.transform:
            data["image"] = self.transform(data["image"])

        if self.target_transform and "label" in data:
            data["label"] = self.target_transform(data["label"])

        return data

    def _load_nifti(self, file_path: Path) -> dict:
        """Load NIfTI file."""
        img = nib.load(str(file_path))
        data = img.get_fdata()

        return {
            "image": torch.from_numpy(data.astype(np.float32)),
            "affine": torch.from_numpy(img.affine),
            "header": img.header,
            "file_path": str(file_path)
        }

    def _load_dicom(self, file_path: Path) -> dict:
        """Load DICOM file."""
        dcm = pydicom.dcmread(str(file_path))
        data = dcm.pixel_array

        metadata = {
            "patient_id": getattr(dcm, "PatientID", None),
            "study_date": getattr(dcm, "StudyDate", None),
            "series_number": getattr(dcm, "SeriesNumber", None),
            "slice_thickness": getattr(dcm, "SliceThickness", None),
            "pixel_spacing": getattr(dcm, "PixelSpacing", None),
        }

        return {
            "image": torch.from_numpy(data.astype(np.float32)),
            "metadata": metadata,
            "file_path": str(file_path)
        }
