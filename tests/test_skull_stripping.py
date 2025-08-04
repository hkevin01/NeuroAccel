"""Test suite for skull stripping module."""
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from neuroaccel.preprocessing.skull_stripping import (
    SkullStripping,
    SkullStrippingModel,
    ConvBlock
)

@pytest.fixture
def sample_3d_data():
    """Create sample 3D data for testing."""
    # Create simple 3D volume with brain-like pattern
    size = 32
    data = torch.zeros(size, size, size)

    # Add spherical "brain" region
    for x in range(size):
        for y in range(size):
            for z in range(size):
                r = np.sqrt(
                    (x - size/2)**2 +
                    (y - size/2)**2 +
                    (z - size/2)**2
                )
                if r < size/4:
                    data[x, y, z] = 1.0

    # Add "skull" layer
    kernel = torch.ones(3, 3, 3) / 27
    skull = F.conv3d(
        data.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=1
    ).squeeze()
    skull = (skull > 0.1) & (skull < 0.9)
    data[skull] = 0.5

    return data

@pytest.fixture
def sample_4d_data(sample_3d_data):
    """Create sample 4D data for testing."""
    # Stack multiple 3D volumes
    return torch.stack([sample_3d_data] * 4, dim=0)

@pytest.fixture
def skull_stripping():
    """Create skull stripping instance."""
    return SkullStripping(device="cuda" if torch.cuda.is_available() else "cpu")

def test_conv_block():
    """Test 3D convolutional block."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block = ConvBlock(1, 16).to(device)

    x = torch.randn(1, 1, 16, 16, 16).to(device)
    out = block(x)

    assert out.shape == (1, 16, 16, 16, 16)

def test_model_architecture():
    """Test skull stripping model architecture."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkullStrippingModel().to(device)

    # Test forward pass
    x = torch.randn(1, 1, 32, 32, 32).to(device)
    out = model(x)

    assert out.shape == (1, 1, 32, 32, 32)
    assert torch.all((out >= 0) & (out <= 1))

def test_skull_stripping_3d(sample_3d_data, skull_stripping):
    """Test skull stripping on 3D data."""
    result = skull_stripping(sample_3d_data)

    # Check output shape matches input
    assert result.shape == sample_3d_data.shape

    # Check values are in valid range
    assert torch.all((result >= 0) & (result <= 1))

    # Check some brain tissue is preserved
    assert torch.sum(result > 0) > 0

    # Check some non-brain tissue is removed
    assert torch.sum(result == 0) > 0

def test_skull_stripping_4d(sample_4d_data, skull_stripping):
    """Test skull stripping on 4D data."""
    result = skull_stripping(sample_4d_data)

    # Check output shape matches input
    assert result.shape == sample_4d_data.shape

    # Check consistent results across volumes
    for i in range(len(result)-1):
        assert torch.allclose(
            result[i],
            result[i+1],
            rtol=1e-5
        )

def test_preprocessing(skull_stripping, sample_3d_data):
    """Test data preprocessing."""
    processed = skull_stripping.preprocess(sample_3d_data)

    # Check normalization to [0,1]
    assert torch.min(processed) >= 0
    assert torch.max(processed) <= 1

    # Test shape handling
    target_shape = (16, 16, 16)
    resized = skull_stripping.preprocess(
        sample_3d_data,
        target_shape=target_shape
    )
    assert resized.shape[-3:] == target_shape

def test_postprocessing(skull_stripping):
    """Test mask postprocessing."""
    device = skull_stripping.device

    # Create test mask
    mask = torch.randn(1, 1, 16, 16, 16).to(device)
    orig_shape = (32, 32, 32)

    processed = skull_stripping.postprocess(mask, orig_shape)

    # Check binary output
    assert torch.all(torch.logical_or(processed == 0, processed == 1))

    # Check shape restoration
    assert processed.shape == orig_shape

def test_invalid_input_shape(skull_stripping):
    """Test handling of invalid input shapes."""
    # Test 2D input
    with pytest.raises(ValueError):
        invalid_data = torch.randn(16, 16)
        skull_stripping(invalid_data)

    # Test 5D input
    with pytest.raises(ValueError):
        invalid_data = torch.randn(1, 1, 16, 16, 16, 1)
        skull_stripping(invalid_data)
