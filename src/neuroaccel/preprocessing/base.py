"""Base classes for preprocessing pipelines."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

import torch
import numpy as np

from neuroaccel.core.gpu import GPUManager, DeviceType

logger = logging.getLogger(__name__)

class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    def __init__(self, device: DeviceType = "cuda"):
        """Initialize preprocessing step.

        Args:
            device: Computing device to use
        """
        self.gpu_manager = GPUManager(device=device)
        self.device = self.gpu_manager.get_device()

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing step to input data.

        Args:
            data: Input data tensor

        Returns:
            Processed data tensor
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Call preprocessing step as a function.

        Args:
            data: Input data tensor

        Returns:
            Processed data tensor
        """
        return self.forward(data)

class Pipeline:
    """Preprocessing pipeline that combines multiple steps."""

    def __init__(
        self,
        steps: Optional[List[PreprocessingStep]] = None,
        device: DeviceType = "cuda"
    ):
        """Initialize pipeline.

        Args:
            steps: List of preprocessing steps
            device: Computing device to use
        """
        self.steps = steps or []
        self.gpu_manager = GPUManager(device=device)
        self.device = self.gpu_manager.get_device()
        self.metadata: Dict[str, Any] = {}

    def add_step(self, step: PreprocessingStep) -> None:
        """Add a preprocessing step to the pipeline.

        Args:
            step: Preprocessing step to add
        """
        self.steps.append(step)

    def process(self, data: torch.Tensor) -> torch.Tensor:
        """Apply all preprocessing steps to input data.

        Args:
            data: Input data tensor

        Returns:
            Processed data tensor
        """
        current = data.to(self.device)

        for step in self.steps:
            try:
                current = step(current)
                self.gpu_manager.synchronize()
            except Exception as e:
                logger.error(f"Error in preprocessing step {step.__class__.__name__}: {e}")
                raise

        return current

    def save_metadata(self, output_path: Path) -> None:
        """Save pipeline metadata to file.

        Args:
            output_path: Path to save metadata
        """
        import json

        metadata = {
            "pipeline_steps": [step.__class__.__name__ for step in self.steps],
            "device": str(self.device),
            **self.metadata
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, input_path: Path) -> None:
        """Load pipeline metadata from file.

        Args:
            input_path: Path to load metadata from
        """
        import json

        with open(input_path, "r") as f:
            self.metadata = json.load(f)
