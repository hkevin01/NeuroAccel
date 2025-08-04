"""Test suite for motion correction module."""
import pytest
import torch
import numpy as np

from neuroaccel.preprocessing.motion import MotionCorrection
from neuroaccel.core.gpu import GPUManager

@pytest.fixture
def sample_4d_data():
    """Create sample 4D data for testing."""
    # Create a simple 4D volume (time, depth, height, width)
    data = torch.zeros(10, 16, 16, 16)

    # Add a moving sphere pattern
    for t in range(10):
        # Sphere center moves along diagonal
        center = 8 + np.sin(t * np.pi / 5)
        for d in range(16):
            for h in range(16):
                for w in range(16):
                    # Distance from center
                    r = np.sqrt((d-center)**2 + (h-center)**2 + (w-center)**2)
                    # Create sphere with radius 3
                    if r < 3:
                        data[t, d, h, w] = 1.0

    return data

@pytest.fixture
def motion_correction():
    """Create motion correction instance."""
    return MotionCorrection(device="cuda" if torch.cuda.is_available() else "cpu")

def test_motion_correction_initialization():
    """Test motion correction initialization."""
    mc = MotionCorrection()
    assert isinstance(mc.gpu_manager, GPUManager)
    assert mc.reference_vol is None
    assert mc.interpolation == "trilinear"
    assert mc.max_iterations == 100
    assert abs(mc.tolerance - 1e-6) < 1e-10  # Compare with small epsilon

def test_motion_correction_device_selection():
    """Test device selection."""
    mc_cuda = MotionCorrection(device="cuda" if torch.cuda.is_available() else "cpu")
    assert str(mc_cuda.device).startswith("cuda" if torch.cuda.is_available() else "cpu")

    mc_cpu = MotionCorrection(device="cpu")
    assert str(mc_cpu.device) == "cpu"

def test_motion_correction_forward(sample_4d_data, motion_correction):
    """Test forward pass of motion correction."""
    data = sample_4d_data
    corrected = motion_correction.forward(data)

    # Check output shape matches input
    assert corrected.shape == data.shape

    # Check motion parameters were generated
    assert len(motion_correction.motion_parameters) == data.shape[0]

    # Check each motion parameter tensor has correct shape (6 parameters)
    for params in motion_correction.motion_parameters:
        assert params.shape == (6,)

def test_motion_correction_reference_selection(sample_4d_data):
    """Test reference volume selection."""
    # Test with mean reference
    mc_mean = MotionCorrection(reference_vol=None)
    corrected_mean = mc_mean.forward(sample_4d_data)

    # Test with specific reference volume
    mc_specific = MotionCorrection(reference_vol=0)
    corrected_specific = mc_specific.forward(sample_4d_data)

    # Outputs should be different when using different reference volumes
    assert not torch.allclose(corrected_mean, corrected_specific)

def test_motion_correction_interpolation(sample_4d_data):
    """Test different interpolation methods."""
    # Test trilinear interpolation
    mc_trilinear = MotionCorrection(interpolation="trilinear")
    corrected_trilinear = mc_trilinear.forward(sample_4d_data)

    # Test nearest neighbor interpolation
    mc_nearest = MotionCorrection(interpolation="nearest")
    corrected_nearest = mc_nearest.forward(sample_4d_data)

    # Outputs should be different with different interpolation methods
    assert not torch.allclose(corrected_trilinear, corrected_nearest)

def test_invalid_input_shape():
    """Test handling of invalid input shapes."""
    mc = MotionCorrection()

    # Test 3D input (missing time dimension)
    with pytest.raises(ValueError):
        invalid_data = torch.randn(16, 16, 16)
        mc.forward(invalid_data)

    # Test 5D input (extra dimension)
    with pytest.raises(ValueError):
        invalid_data = torch.randn(10, 16, 16, 16, 1)
        mc.forward(invalid_data)

def test_motion_parameter_consistency(sample_4d_data, motion_correction):
    """Test consistency of motion parameters."""
    # Run motion correction twice on same data
    corrected1 = motion_correction.forward(sample_4d_data)
    params1 = motion_correction.get_motion_parameters()

    corrected2 = motion_correction.forward(sample_4d_data)
    params2 = motion_correction.get_motion_parameters()

    # Parameters should be similar for same input
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2, rtol=1e-3)

    # Corrected outputs should be similar
    assert torch.allclose(corrected1, corrected2, rtol=1e-3)
