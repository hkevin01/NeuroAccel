"""GPU-accelerated skull stripping using 3D U-Net architecture."""
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from neuroaccel.preprocessing.base import PreprocessingStep
from neuroaccel.core.gpu import DeviceType

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """3D convolutional block with batch normalization and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Initialize conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            padding: Padding size
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, channels, depth, height, width]

        Returns:
            Output tensor
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SkullStrippingModel(nn.Module):
    """3D U-Net model for skull stripping."""

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 16,
        depth: int = 4
    ):
        """Initialize model.

        Args:
            in_channels: Number of input channels
            base_filters: Number of base filters (doubled at each level)
            depth: Number of downsampling/upsampling levels
        """
        super().__init__()

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        # Initial block
        self.encoder_blocks.append(
            ConvBlock(in_channels, base_filters)
        )

        # Remaining encoder blocks
        for i in range(depth-1):
            in_filters = base_filters * (2**i)
            out_filters = base_filters * (2**(i+1))
            self.encoder_blocks.append(
                ConvBlock(in_filters, out_filters)
            )

        # Decoder blocks and upsampling
        self.decoder_blocks = nn.ModuleList()
        self.upconv = nn.ModuleList()

        for i in range(depth-1, 0, -1):
            in_filters = base_filters * (2**i)
            out_filters = base_filters * (2**(i-1))

            self.upconv.append(
                nn.ConvTranspose3d(
                    in_filters,
                    out_filters,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder_blocks.append(
                ConvBlock(in_filters, out_filters)
            )

        # Final 1x1 convolution
        self.final_conv = nn.Conv3d(
            base_filters,
            1,
            kernel_size=1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, channels, depth, height, width]

        Returns:
            Binary brain mask
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Encoder path
        for block in self.encoder_blocks[:-1]:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        # Bottom level
        x = self.encoder_blocks[-1](x)

        # Decoder path
        for i, block in enumerate(self.decoder_blocks):
            # Upsample
            x = self.upconv[i](x)

            # Get corresponding encoder output
            encoder_out = encoder_outputs[-(i+1)]

            # Match sizes for skip connection
            if x.shape != encoder_out.shape:
                x = F.interpolate(
                    x,
                    size=encoder_out.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            # Concatenate and decode
            x = torch.cat([x, encoder_out], dim=1)
            x = block(x)

        # Final 1x1 convolution and sigmoid
        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x

class SkullStripping(PreprocessingStep):
    """GPU-accelerated skull stripping for neuroimaging data."""

    def __init__(
        self,
        device: DeviceType = "cuda",
        model_path: Optional[str] = None,
        use_synthetic_data: bool = True
    ):
        """Initialize skull stripping.

        Args:
            device: Computing device to use
            model_path: Path to pretrained model weights
            use_synthetic_data: Whether to use synthetic data augmentation
        """
        super().__init__(device)

        # Initialize model
        self.model = SkullStrippingModel().to(self.device)

        # Load pretrained weights if provided
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        self.model.eval()

        # Synthetic data options
        self.use_synthetic = use_synthetic_data

    def preprocess(
        self,
        data: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Preprocess input data.

        Args:
            data: Input 3D/4D data tensor
            target_shape: Optional shape to resize to

        Returns:
            Preprocessed data tensor
        """
        # Add channel dimension if needed
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        # Normalize to [0,1]
        data = (data - data.min()) / (data.max() - data.min())

        # Resize if needed
        if target_shape:
            data = F.interpolate(
                data.unsqueeze(0),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        return data

    def postprocess(
        self,
        mask: torch.Tensor,
        orig_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Postprocess output mask.

        Args:
            mask: Output mask tensor
            orig_shape: Original data shape to resize to

        Returns:
            Postprocessed binary mask
        """
        # Threshold to binary mask
        mask = (mask > 0.5).float()

        # Resize to original shape if needed
        if mask.shape[2:] != orig_shape:
            mask = F.interpolate(
                mask,
                size=orig_shape,
                mode='nearest'
            )

        return mask.squeeze()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply skull stripping to input data.

        Args:
            data: Input data tensor

        Returns:
            Brain-extracted data tensor
        """
        # Remember original shape
        orig_shape = data.shape[-3:]

        # Preprocess data
        preprocessed = self.preprocess(data)

        # Get brain mask
        with torch.no_grad():
            mask = self.model(preprocessed.unsqueeze(0))

        # Postprocess mask
        mask = self.postprocess(mask, orig_shape)

        # Apply mask to input
        result = data * mask

        return result
