"""GPU-accelerated motion correction implementation."""
from typing import Optional, Tuple, List
import logging

import torch
import torch.nn.functional as F
import numpy as np

from neuroaccel.preprocessing.base import PreprocessingStep
from neuroaccel.core.gpu import DeviceType

logger = logging.getLogger(__name__)

class MotionCorrection(PreprocessingStep):
    """GPU-accelerated motion correction for 3D/4D neuroimaging data."""
    
    def __init__(
        self,
        device: DeviceType = "cuda",
        reference_vol: Optional[int] = None,
        interpolation: str = "trilinear",
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """Initialize motion correction.
        
        Args:
            device: Computing device to use
            reference_vol: Index of reference volume (None for mean)
            interpolation: Interpolation method ("trilinear" or "nearest")
            max_iterations: Maximum number of iterations for optimization
            tolerance: Convergence tolerance
        """
        super().__init__(device)
        self.reference_vol = reference_vol
        self.interpolation = interpolation
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        self.motion_parameters: List[torch.Tensor] = []
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply motion correction to input data.
        
        Args:
            data: Input 4D data tensor (time, depth, height, width)
            
        Returns:
            Motion corrected data tensor
        """
        if len(data.shape) != 4:
            raise ValueError(f"Expected 4D input data, got shape {data.shape}")
        
        data = data.to(self.device)
        
        # Select reference volume
        if self.reference_vol is None:
            reference = torch.mean(data, dim=0, keepdim=True)
        else:
            reference = data[self.reference_vol:self.reference_vol+1]
        
        corrected_data = []
        self.motion_parameters = []
        
        # Process each volume
        for t in range(data.shape[0]):
            volume = data[t:t+1]  # Keep time dimension
            
            # Estimate and apply motion correction
            params = self._estimate_motion(volume, reference)
            corrected = self._apply_transform(volume, params)
            
            corrected_data.append(corrected)
            self.motion_parameters.append(params)
        
        return torch.cat(corrected_data, dim=0)
    
    def _estimate_motion(
        self,
        volume: torch.Tensor,
        reference: torch.Tensor
    ) -> torch.Tensor:
        """Estimate motion parameters between volume and reference.
        
        Args:
            volume: Input volume to align
            reference: Reference volume
            
        Returns:
            Motion parameters (translation and rotation)
        """
        # Initialize parameters (tx, ty, tz, rx, ry, rz)
        params = torch.zeros(6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.01, weight_decay=1e-5)
        
        prev_loss = float('inf')
        
        for _ in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Apply current transformation
            transformed = self._apply_transform(volume, params)
            
            # Compute loss
            loss = F.mse_loss(transformed, reference)
            
            # Check convergence
            if abs(prev_loss - loss.item()) < self.tolerance:
                break
                
            loss.backward()
            optimizer.step()
            prev_loss = loss.item()
        
        return params.detach()
    
    def _apply_transform(
        self,
        volume: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """Apply motion transformation to volume.
        
        Args:
            volume: Input volume
            params: Motion parameters
            
        Returns:
            Transformed volume
        """
        # Extract parameters
        tx, ty, tz, rx, ry, rz = params
        
        # Create affine transformation matrix
        theta = self._create_affine_matrix(tx, ty, tz, rx, ry, rz)
        
        # Create grid
        grid = F.affine_grid(
            theta.unsqueeze(0),
            volume.shape,
            align_corners=False
        )
        
        # Apply transformation
        mode = "bilinear" if self.interpolation == "trilinear" else "nearest"
        transformed = F.grid_sample(
            volume,
            grid,
            mode=mode,
            align_corners=False
        )
        
        return transformed
    
    def _create_affine_matrix(
        self,
        tx: torch.Tensor,
        ty: torch.Tensor,
        tz: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        rz: torch.Tensor
    ) -> torch.Tensor:
        """Create 3D affine transformation matrix.
        
        Args:
            tx, ty, tz: Translation parameters
            rx, ry, rz: Rotation parameters (in radians)
            
        Returns:
            4x4 affine transformation matrix
        """
        # Create rotation matrices
        cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
        cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
        cos_rz, sin_rz = torch.cos(rz), torch.sin(rz)
        
        # Create combined rotation matrix directly
        R = torch.tensor([
            [cos_ry * cos_rz, 
             -cos_rx * sin_rz + sin_rx * sin_ry * cos_rz,
             sin_rx * sin_rz + cos_rx * sin_ry * cos_rz],
            [cos_ry * sin_rz,
             cos_rx * cos_rz + sin_rx * sin_ry * sin_rz,
             -sin_rx * cos_rz + cos_rx * sin_ry * sin_rz],
            [-sin_ry,
             sin_rx * cos_ry,
             cos_rx * cos_ry]
        ], device=self.device)
        
        # Create full transformation matrix
        theta = torch.eye(4, device=self.device)
        theta[:3, :3] = R
        theta[:3, 3] = torch.tensor([tx, ty, tz], device=self.device)
        
        return theta[:3]  # Return 3x4 matrix for grid_sample
    
    def get_motion_parameters(self) -> List[torch.Tensor]:
        """Get estimated motion parameters for each volume.
        
        Returns:
            List of motion parameters
        """
        return self.motion_parameters
