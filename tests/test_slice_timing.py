"""Test suite for slice timing correction."""
import pytest
import torch
import numpy as np

from neuroaccel.preprocessing.slice_timing import SliceTimingCorrection

@pytest.fixture
def sample_4d_data():
    """Create sample 4D data for testing."""
    # Create simple 4D volume with sinusoidal signal
    time_points = 20
    slices = 10
    size = 16

    data = torch.zeros(time_points, slices, size, size)
    t = torch.linspace(0, 4*np.pi, time_points)

    for i in range(slices):
        # Add phase shift for each slice
        phase_shift = i * np.pi / slices
        signal = torch.sin(t + phase_shift)

        # Create 2D pattern for each slice
        pattern = torch.ones(size, size)
        data[:, i] = signal.reshape(-1, 1, 1) * pattern

    return data

@pytest.fixture
def slice_timing():
    """Create slice timing correction instance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SliceTimingCorrection(device=device)

def test_slice_timing_initialization():
    """Test slice timing correction initialization."""
    # Test default initialization
    stc = SliceTimingCorrection()
    assert abs(stc.tr - 2.0) < 1e-10  # Compare with small epsilon
    assert stc.interp_method == "linear"
    assert stc.slice_axis == 1
    assert stc.reference_slice is None

    # Test custom initialization
    slice_times = [0.0, 0.5, 1.0, 1.5]
    stc = SliceTimingCorrection(
        tr=3.0,
        slice_times=slice_times,
        interp_method="sinc",
        reference_slice=2
    )
    assert abs(stc.tr - 3.0) < 1e-10  # Compare with small epsilon
    assert stc.interp_method == "sinc"
    assert stc.reference_slice == 2
    assert torch.allclose(stc.slice_times, torch.tensor(slice_times))

def test_slice_timing_correction(slice_timing, sample_4d_data):
    """Test slice timing correction on sample data."""
    corrected = slice_timing.forward(sample_4d_data)

    # Check output shape matches input
    assert corrected.shape == sample_4d_data.shape

    # Check reference slice is unchanged
    ref_slice = slice_timing.reference_slice
    assert torch.allclose(
        corrected[:, ref_slice],
        sample_4d_data[:, ref_slice]
    )

    # Check that other slices have been modified
    for i in range(sample_4d_data.shape[1]):
        if i != ref_slice:
            assert not torch.allclose(
                corrected[:, i],
                sample_4d_data[:, i]
            )

def test_slice_timing_interpolation_methods(sample_4d_data):
    """Test different interpolation methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test linear interpolation
    stc_linear = SliceTimingCorrection(
        device=device,
        interp_method="linear"
    )
    corrected_linear = stc_linear.forward(sample_4d_data)

    # Test sinc interpolation
    stc_sinc = SliceTimingCorrection(
        device=device,
        interp_method="sinc"
    )
    corrected_sinc = stc_sinc.forward(sample_4d_data)

    # Results should be different between methods
    assert not torch.allclose(corrected_linear, corrected_sinc)

def test_custom_slice_times(sample_4d_data):
    """Test custom slice acquisition timing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create custom slice timing pattern (interleaved)
    n_slices = sample_4d_data.shape[1]
    slice_times = torch.zeros(n_slices)
    slice_times[::2] = torch.arange(n_slices//2) * 2
    slice_times[1::2] = torch.arange(n_slices//2) * 2 + 1

    stc = SliceTimingCorrection(
        device=device,
        slice_times=slice_times,
        tr=n_slices
    )

    corrected = stc.forward(sample_4d_data)

    # Check output
    assert corrected.shape == sample_4d_data.shape
    assert not torch.allclose(corrected, sample_4d_data)

def test_invalid_input_shape():
    """Test handling of invalid input shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stc = SliceTimingCorrection(device=device)

    # Test 3D input
    with pytest.raises(ValueError):
        invalid_data = torch.randn(10, 16, 16)
        stc.forward(invalid_data)

    # Test 5D input
    with pytest.raises(ValueError):
        invalid_data = torch.randn(10, 16, 16, 16, 1)
        stc.forward(invalid_data)

def test_slice_axis_selection(sample_4d_data):
    """Test correction along different slice axes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test correction along different axes
    for axis in range(1, 4):
        # Rearrange data to have slices along specified axis
        data = sample_4d_data.permute(0, *range(1, 4))

        stc = SliceTimingCorrection(
            device=device,
            slice_axis=axis
        )

        corrected = stc.forward(data)
        assert corrected.shape == data.shape
