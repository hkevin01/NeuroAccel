"""Real-time visualization tools for quality control."""
from typing import Optional, List, Tuple, Dict
import logging

import to                line={"width": 2}
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                y=snr,
                name='SNR',
                line={"width": 2} numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

logger = logging.getLogger(__name__)

class QualityControlVisualizer:
    """Interactive visualization tools for quality control."""

    def __init__(self):
        """Initialize visualizer."""
        self.figure = None
        self.current_data = None

    def plot_motion_parameters(
        self,
        parameters: List[torch.Tensor],
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot estimated motion parameters over time.

        Args:
            parameters: List of motion parameters (tx, ty, tz, rx, ry, rz)
            output_path: Optional path to save plot

        Returns:
            Plotly figure object
        """
        # Convert parameters to numpy arrays
        params_np = np.array([p.cpu().numpy() for p in parameters])

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Translation', 'Rotation'),
            vertical_spacing=0.15
        )

        # Plot translation parameters
        for i, label in enumerate(['x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(
                    y=params_np[:, i],
                    name=f'Translation {label}',
                    line={"width": 2}
                ),
                row=1, col=1
            )

        # Plot rotation parameters
        for i, label in enumerate(['x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(
                    y=params_np[:, i+3],
                    name=f'Rotation {label}',
                    line={"width": 2}
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title='Motion Parameters Over Time',
            showlegend=True,
            height=800,
            template='plotly_white'
        )

        fig.update_yaxes(title_text='mm', row=1, col=1)
        fig.update_yaxes(title_text='radians', row=2, col=1)
        fig.update_xaxes(title_text='Volume', row=2, col=1)

        if output_path:
            fig.write_html(output_path)

        self.figure = fig
        return fig

    def plot_volume_metrics(
        self,
        data: torch.Tensor,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """Plot quality metrics for each volume.

        Args:
            data: 4D data tensor
            output_path: Optional path to save plot

        Returns:
            Plotly figure object
        """
        # Calculate metrics
        mean_intensity = torch.mean(data, dim=(1,2,3)).cpu().numpy()
        std_intensity = torch.std(data, dim=(1,2,3)).cpu().numpy()
        snr = mean_intensity / std_intensity

        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Mean Intensity',
                'Standard Deviation',
                'Signal-to-Noise Ratio'
            ),
            vertical_spacing=0.1
        )

        # Plot metrics
        fig.add_trace(
            go.Scatter(
                y=mean_intensity,
                name='Mean Intensity',
                line={"width": 2}
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                y=std_intensity,
                name='Std Intensity',
                line={"width": 2}
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                y=snr,
                name='SNR',
                line=dict(width=2)
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title='Volume-wise Quality Metrics',
            showlegend=True,
            height=1000,
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Volume', row=3, col=1)

        if output_path:
            fig.write_html(output_path)

        self.figure = fig
        return fig

    def display_orthogonal_views(
        self,
        volume: torch.Tensor,
        slice_indices: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """Display orthogonal views of a volume.

        Args:
            volume: 3D volume tensor
            slice_indices: Optional tuple of (x,y,z) slice indices
        """
        if len(volume.shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {volume.shape}")

        # Convert to numpy and move to CPU if needed
        volume_np = volume.cpu().numpy()

        if slice_indices is None:
            # Use middle slices by default
            slice_indices = tuple(s // 2 for s in volume_np.shape)

        # Create figure with subplots
        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot sagittal view
        axes[0].imshow(
            volume_np[slice_indices[0], :, :],
            cmap='gray',
            aspect='equal'
        )
        axes[0].set_title('Sagittal')

        # Plot coronal view
        axes[1].imshow(
            volume_np[:, slice_indices[1], :],
            cmap='gray',
            aspect='equal'
        )
        axes[1].set_title('Coronal')

        # Plot axial view
        axes[2].imshow(
            volume_np[:, :, slice_indices[2]],
            cmap='gray',
            aspect='equal'
        )
        axes[2].set_title('Axial')

        plt.tight_layout()
        plt.show()

    def create_report(
        self,
        data: torch.Tensor,
        motion_params: List[torch.Tensor],
        output_path: str
    ) -> None:
        """Create comprehensive QC report.

        Args:
            data: 4D data tensor
            motion_params: List of motion parameters
            output_path: Path to save report
        """
        import plotly.io as pio
        from datetime import datetime

        # Create HTML report
        html_content = [
            "<html>",
            "<head>",
            "<title>NeuroAccel Quality Control Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #34495e; margin-top: 30px; }",
            ".plot { margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Quality Control Report</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "<h2>Motion Parameters</h2>"
        ]

        # Add motion parameter plot
        motion_fig = self.plot_motion_parameters(motion_params)
        html_content.append(
            f'<div class="plot">{pio.to_html(motion_fig)}</div>'
        )

        # Add volume metrics plot
        metrics_fig = self.plot_volume_metrics(data)
        html_content.append("<h2>Volume-wise Metrics</h2>")
        html_content.append(
            f'<div class="plot">{pio.to_html(metrics_fig)}</div>'
        )

        # Add summary statistics
        html_content.extend([
            "<h2>Summary Statistics</h2>",
            "<ul>",
            f"<li>Number of volumes: {data.shape[0]}</li>",
            f"<li>Volume dimensions: {data.shape[1:]}</li>",
            f"<li>Mean intensity: {torch.mean(data):.2f}</li>",
            f"<li>Standard deviation: {torch.std(data, dim=(0,1,2,3)):.2f}</li>",
            "</ul>",
            "</body>",
            "</html>"
        ])

        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_content))

        logger.info(f"QC report generated at: {output_path}")
