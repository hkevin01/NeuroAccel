"""Test suite for quality control visualization."""
import pytest
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from neuroaccel.visualization.quality_control import QualityControlVisualizer

@pytest.fixture
def sample_motion_parameters():
    """Create sample motion parameters for testing."""
    # Create 10 volumes worth of motion parameters
    params = []
    for i in range(10):
        # Create sinusoidal motion pattern
        tx = 0.1 * np.sin(i * np.pi / 5)
        ty = 0.2 * np.cos(i * np.pi / 5)
        tz = 0.15 * np.sin(i * np.pi / 4)
        rx = 0.02 * np.cos(i * np.pi / 6)
        ry = 0.03 * np.sin(i * np.pi / 7)
        rz = 0.01 * np.cos(i * np.pi / 8)
        params.append(torch.tensor([tx, ty, tz, rx, ry, rz]))
    return params

@pytest.fixture
def sample_4d_data():
    """Create sample 4D data for testing."""
    # Create simple 4D volume (time, depth, height, width)
    data = torch.zeros(10, 16, 16, 16)

    # Add some varying patterns over time
    for t in range(10):
        # Create intensity gradient
        intensity = 0.5 + 0.1 * np.sin(t * np.pi / 5)
        noise = 0.1 * torch.randn(16, 16, 16)
        data[t] = intensity + noise

    return data

@pytest.fixture
def visualizer():
    """Create visualizer instance."""
    return QualityControlVisualizer()

def test_motion_parameter_plot(visualizer, sample_motion_parameters):
    """Test motion parameter plotting."""
    fig = visualizer.plot_motion_parameters(sample_motion_parameters)

    # Check that figure was created
    assert isinstance(fig, go.Figure)

    # Check number of traces (3 translation + 3 rotation = 6)
    assert len(fig.data) == 6

    # Check trace names
    expected_names = [
        'Translation x', 'Translation y', 'Translation z',
        'Rotation x', 'Rotation y', 'Rotation z'
    ]
    actual_names = [trace.name for trace in fig.data]
    assert all(exp in actual_names for exp in expected_names)

def test_volume_metrics_plot(visualizer, sample_4d_data):
    """Test volume metrics plotting."""
    fig = visualizer.plot_volume_metrics(sample_4d_data)

    # Check that figure was created
    assert isinstance(fig, go.Figure)

    # Check number of traces (mean, std, snr = 3)
    assert len(fig.data) == 3

    # Check trace names
    expected_names = ['Mean Intensity', 'Std Intensity', 'SNR']
    actual_names = [trace.name for trace in fig.data]
    assert all(exp in actual_names for exp in expected_names)

def test_orthogonal_views(visualizer, sample_4d_data, monkeypatch):
    """Test orthogonal view display."""
    # Mock plt.show to avoid display during testing
    shown = False
    def mock_show():
        nonlocal shown
        shown = True
    monkeypatch.setattr(plt, 'show', mock_show)

    # Test with single volume
    volume = sample_4d_data[0]
    visualizer.display_orthogonal_views(volume)

    # Check that plot was shown
    assert shown

    # Test with custom slice indices
    shown = False
    visualizer.display_orthogonal_views(volume, (8, 8, 8))
    assert shown

    # Test invalid input
    with pytest.raises(ValueError):
        visualizer.display_orthogonal_views(sample_4d_data)  # 4D input

def test_create_report(visualizer, sample_4d_data, sample_motion_parameters, tmp_path):
    """Test QC report creation."""
    # Create report
    report_path = tmp_path / "qc_report.html"
    visualizer.create_report(
        sample_4d_data,
        sample_motion_parameters,
        str(report_path)
    )

    # Check that report was created
    assert report_path.exists()

    # Check report content
    content = report_path.read_text()
    assert "Quality Control Report" in content
    assert "Motion Parameters" in content
    assert "Volume-wise Metrics" in content
    assert "Summary Statistics" in content
