"""Core GPU utilities for managing device selection and memory."""
from typing import Literal, Optional, Union
import os

import torch
import cupy as cp

DeviceType = Literal["cuda", "rocm", "cpu"]

class GPUManager:
    """Manages GPU device selection and memory allocation."""

    def __init__(
        self,
        device: DeviceType = "cuda",
        memory_fraction: float = 0.9,
        device_index: int = 0
    ):
        """Initialize GPU manager.

        Args:
            device: Device type to use ("cuda", "rocm", or "cpu")
            memory_fraction: Fraction of GPU memory to use (0.0 to 1.0)
            device_index: GPU device index if multiple devices available
        """
        self.device_type = device
        self.memory_fraction = memory_fraction
        self.device_index = device_index
        self._initialize_device()

    def _initialize_device(self) -> None:
        """Initialize the selected GPU device."""
        if self.device_type == "cpu":
            self.device = torch.device("cpu")
            return

        if self.device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available")
            self.device = torch.device(f"cuda:{self.device_index}")
            torch.cuda.set_device(self.device)
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

        elif self.device_type == "rocm":
            if not hasattr(torch, 'hip') or not torch.hip.is_available():
                raise RuntimeError("ROCm device requested but not available")
            self.device = torch.device(f"hip:{self.device_index}")
            # Set ROCm specific configurations here

        self._setup_memory_pool()

    def _setup_memory_pool(self) -> None:
        """Configure memory pool for efficient GPU memory management."""
        if self.device_type != "cpu":
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device

    def memory_info(self) -> dict:
        """Get current memory usage information."""
        if self.device_type == "cpu":
            return {"device": "cpu", "memory_used": "N/A"}

        memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3  # GB

        return {
            "device": str(self.device),
            "memory_used_gb": f"{memory_used:.2f}",
            "memory_cached_gb": f"{memory_cached:.2f}",
            "memory_fraction": self.memory_fraction
        }

    def synchronize(self) -> None:
        """Synchronize the device."""
        if self.device_type != "cpu":
            torch.cuda.synchronize(self.device)
