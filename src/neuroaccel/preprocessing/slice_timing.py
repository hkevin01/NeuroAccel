"""GPU-accelerated slice timing correction."""
from typing import Optional, List, Union
import logging

import torch
import torch.nn.functional as F
import numpy as np

from neuroaccel.preprocessing.base import PreprocessingStep
from neuroaccel.core.gpu import DeviceType

logger = logging.getLogger(__name__)

class SliceTimingCorrection(PreprocessingStep):
    """GPU-accelerated slice timing correction for fMRI data."""

    def __init__(
        self,
        device: DeviceType = "cuda",
        slice_times: Optional[Union[List[float], torch.Tensor]] = None,
        tr: float = 2.0,
        interp_method: str = "linear",
        slice_axis: int = 1,
        reference_slice: Optional[int] = None
    ):
        """Initialize slice timing correction.

        Args:
            device: Computing device to use
            slice_times: Acquisition time of each slice relative to start of TR
            tr: Repetition time in seconds
            interp_method: Interpolation method ("linear" or "sinc")
            slice_axis: Axis representing slices (default 1 for [time, slice, row, col])
            reference_slice: Index of reference slice (None for middle slice)
        """
        super().__init__(device)
        self.tr = tr
        self.interp_method = interp_method
        self.slice_axis = slice_axis

        # Set up slice timing information
        if slice_times is None:
            # Assume sequential acquisition
            n_slices = None  # Will be set in forward pass
            self.slice_times = None
        else:
            if isinstance(slice_times, list):
                slice_times = torch.tensor(slice_times)
            self.slice_times = slice_times.to(self.device)
            n_slices = len(slice_times)

        # Set reference slice
        if reference_slice is None and n_slices is not None:
            self.reference_slice = n_slices // 2
        else:
            self.reference_slice = reference_slice

    def _initialize_slice_times(self, n_slices: int) -> None:
        """Initialize slice timing pattern if not provided.

        Args:
            n_slices: Number of slices in volume
        """
        if self.slice_times is None:
            # Create sequential slice timing
            self.slice_times = torch.linspace(0, self.tr, n_slices).to(self.device)
            if self.reference_slice is None:
                self.reference_slice = n_slices // 2

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply slice timing correction to input data.

        Args:
            data: Input 4D data tensor (time, slice, row, col)

        Returns:
            Slice-time corrected data tensor
        """
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D input data, got shape {data.shape}")

        # Get dimensions
        n_volumes = data.shape[0]
        n_slices = data.shape[self.slice_axis]

        # Initialize slice times if needed
        self._initialize_slice_times(n_slices)

        # Move data to device
        data = data.to(self.device)

        # Get reference slice time
        ref_time = self.slice_times[self.reference_slice]

        # Calculate time shifts
        shifts = self.slice_times - ref_time

        # Create time points for interpolation
        timesteps = torch.arange(n_volumes, device=self.device) * self.tr

        # Prepare output tensor
        corrected = torch.zeros_like(data)

        # Process each slice
        for slice_idx in range(n_slices):
            # Skip reference slice
            if slice_idx == self.reference_slice:
                slice_selector = [slice(None)] * 4
                slice_selector[self.slice_axis] = slice_idx
                corrected[tuple(slice_selector)] = data[tuple(slice_selector)]
                continue

            # Get shifted time points for this slice
            slice_times = timesteps + shifts[slice_idx]

            # Select slice data
            slice_selector = [slice(None)] * 4
            slice_selector[self.slice_axis] = slice_idx
            slice_data = data[tuple(slice_selector)]

            # Perform interpolation
            if self.interp_method == "linear":
                corrected_slice = self._linear_interpolate(
                    slice_data,
                    timesteps,
                    slice_times
                )
            else:  # sinc interpolation
                corrected_slice = self._sinc_interpolate(
                    slice_data,
                    timesteps,
                    slice_times
                )

            # Store corrected slice
            corrected[tuple(slice_selector)] = corrected_slice

        return corrected

    def _linear_interpolate(
        self,
        data: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """Perform linear interpolation in time.

        Args:
            data: Input slice data
            orig_times: Original time points
            new_times: Time points to interpolate to

        Returns:
            Interpolated data
        """
        # Normalize time points to [0, 1]
        t_min, t_max = orig_times.min(), orig_times.max()
        x = (orig_times - t_min) / (t_max - t_min)
        xi = (new_times - t_min) / (t_max - t_min)

        # Reshape for grid_sample
        xi = xi.view(-1, 1, 1)  # [time, 1, 1]
        x = x.view(-1, 1, 1)   # [time, 1, 1]

        # Create interpolation grid
        grid = 2.0 * (xi - x.min()) / (x.max() - x.min()) - 1.0

        # Perform interpolation
        return F.grid_sample(
            data.unsqueeze(0),  # Add batch dimension
            grid.expand(-1, 1, 1),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

    def _sinc_interpolate(
        self,
        data: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """Perform sinc interpolation in time.

        Args:
            data: Input slice data
            orig_times: Original time points
            new_times: Time points to interpolate to

        Returns:
            Interpolated data
        """
        # Calculate time differences
        dt = torch.mean(orig_times[1:] - orig_times[:-1])

        # Create sinc kernel
        window_size = 5  # Number of points on each side
        t_diff = new_times.reshape(-1, 1) - orig_times.reshape(1, -1)
        sinc_kernel = torch.sinc(t_diff / dt)

        # Apply Hann window
        window = torch.hann_window(2 * window_size + 1, device=self.device)
        kernel = sinc_kernel * window.reshape(1, -1)

        # Normalize kernel
        kernel = kernel / kernel.sum(dim=1, keepdim=True)

        # Perform convolution
        return torch.matmul(kernel, data.reshape(len(orig_times), -1)).reshape(data.shape)
