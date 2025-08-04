"""Advanced analytics module for GPU-accelerated neuroimaging analysis."""
from typing import Optional, List, Tuple, Dict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats

from neuroaccel.core.gpu import GPUManager, DeviceType

logger = logging.getLogger(__name__)

class ICAModule(nn.Module):
    """GPU-accelerated Independent Component Analysis."""

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        device: DeviceType = "cuda"
    ):
        """Initialize ICA module.

        Args:
            n_components: Number of independent components to extract
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            device: Computing device to use
        """
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.gpu_manager = GPUManager(device=device)
        self.device = self.gpu_manager.get_device()

        # Initialize unmixing matrix
        self.W = nn.Parameter(torch.randn(n_components, n_components))
        self.components_ = None
        self.mixing_ = None

    def _center_data(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Center the data by removing the mean.

        Args:
            X: Input data tensor [n_samples, n_features]

        Returns:
            Centered data and mean
        """
        mean = torch.mean(X, dim=0, keepdim=True)
        return X - mean, mean

    def _whiten_data(
        self,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Whiten the data using PCA.

        Args:
            X: Input data tensor [n_samples, n_features]

        Returns:
            Whitened data, whitening matrix, and dewhitening matrix
        """
        # Compute SVD
        U, S, V = torch.svd(X.t())

        # Select top components
        U = U[:, :self.n_components]
        S = S[:self.n_components]
        V = V[:, :self.n_components]

        # Compute whitening and dewhitening matrices
        whitening = torch.diag(1.0 / torch.sqrt(S + 1e-10)) @ U.t()
        dewhitening = U @ torch.diag(torch.sqrt(S + 1e-10))

        # Apply whitening
        x_white = X @ whitening.t()

        return x_white, whitening, dewhitening

    def _g(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute non-linear function and its derivative.

        Args:
            X: Input data tensor

        Returns:
            Non-linear function output and its derivative
        """
        # Using tanh as non-linearity
        gx = torch.tanh(X)
        g_prime = 1 - gx ** 2
        return gx, g_prime

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data to independent components.

        Args:
            X: Input data tensor [n_samples, n_features]

        Returns:
            Transformed data
        """
        if self.components_ is None:
            raise RuntimeError("Model must be fit before transform")

        X = X.to(self.device)
        x_center, _ = self._center_data(X)
        return x_center @ self.components_.t()

    def fit(self, X: torch.Tensor) -> 'ICAModule':
        """Fit ICA model to data.

        Args:
            X: Input data tensor [n_samples, n_features]

        Returns:
            Self for chaining
        """
        X = X.to(self.device)

        # Center and whiten data
        x_center, _ = self._center_data(X)
        x_white, whitening, dewhitening = self._whiten_data(x_center)

        # Initialize parameters
        W = torch.eye(self.n_components, device=self.device)

        # FastICA algorithm
        for n_iter in range(self.max_iter):
            # Compute ICA update
            gx, g_prime = self._g(x_white @ W.t())
            w = (gx.t() @ x_white) / x_white.shape[0] - \
                    torch.mean(g_prime, dim=0).unsqueeze(1) * W

            # Symmetric orthogonalization
            w = w @ torch.matrix_power(w @ w.t(), -0.5)

            # Check convergence
            if torch.abs(torch.abs((w * W).sum(axis=1)) - 1).max() < self.tolerance:
                break

            # Update weights
            W = w

            if n_iter == self.max_iter - 1:
                logger.warning("FastICA did not converge")

        # Compute mixing and unmixing matrices
        self.components_ = W @ whitening
        self.mixing_ = dewhitening @ W.t()

        return self

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform independent components back to original space.

        Args:
            X: Independent components [n_samples, n_components]

        Returns:
            Data in original space
        """
        if self.mixing_ is None:
            raise RuntimeError("Model must be fit before inverse_transform")

        X = X.to(self.device)
        return X @ self.mixing_.t()

class ConnectivityAnalysis:
    """GPU-accelerated connectivity analysis."""

    def __init__(
        self,
        method: str = "pearson",
        device: DeviceType = "cuda"
    ):
        """Initialize connectivity analysis.

        Args:
            method: Connectivity measure ("pearson", "partial", or "coherence")
            device: Computing device to use
        """
        self.method = method
        self.gpu_manager = GPUManager(device=device)
        self.device = self.gpu_manager.get_device()

    def compute_connectivity(
        self,
        timeseries: torch.Tensor,
        frequency_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """Compute connectivity matrix.

        Args:
            timeseries: Input timeseries [n_samples, n_nodes]
            frequency_range: Optional frequency range for spectral measures

        Returns:
            Connectivity matrix [n_nodes, n_nodes]
        """
        timeseries = timeseries.to(self.device)

        if self.method == "pearson":
            return self._pearson_correlation(timeseries)
        elif self.method == "partial":
            return self._partial_correlation(timeseries)
        elif self.method == "coherence":
            if frequency_range is None:
                raise ValueError("frequency_range required for coherence")
            return self._coherence(timeseries, frequency_range)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _pearson_correlation(self, timeseries: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation matrix.

        Args:
            timeseries: Input timeseries [n_samples, n_nodes]

        Returns:
            Correlation matrix [n_nodes, n_nodes]
        """
        # Center and normalize
        timeseries = timeseries - timeseries.mean(dim=0, keepdim=True)
        timeseries = timeseries / (timeseries.std(dim=0, keepdim=True) + 1e-8)

        # Compute correlation
        return (timeseries.t() @ timeseries) / (timeseries.shape[0] - 1)

    def _partial_correlation(self, timeseries: torch.Tensor) -> torch.Tensor:
        """Compute partial correlation matrix.

        Args:
            timeseries: Input timeseries [n_samples, n_nodes]

        Returns:
            Partial correlation matrix [n_nodes, n_nodes]
        """
        # Compute precision matrix (inverse covariance)
        cov = self._pearson_correlation(timeseries)
        precision = torch.inverse(cov + torch.eye(cov.shape[0], device=self.device) * 1e-6)

        # Convert to partial correlations
        diag = torch.sqrt(torch.diag(precision))
        partial_corr = -precision / (diag.unsqueeze(0) @ diag.unsqueeze(1))
        torch.diagonal(partial_corr)[:] = 1.0

        return partial_corr

    def _coherence(
        self,
        timeseries: torch.Tensor,
        frequency_range: Tuple[float, float]
    ) -> torch.Tensor:
        """Compute coherence matrix.

        Args:
            timeseries: Input timeseries [n_samples, n_nodes]
            frequency_range: Frequency range (min_freq, max_freq)

        Returns:
            Coherence matrix [n_nodes, n_nodes]
        """
        # Compute FFT
        n_samples = timeseries.shape[0]
        fft = torch.fft.rfft(timeseries, dim=0)
        freqs = torch.fft.rfftfreq(n_samples, d=1.0)

        # Select frequency range
        freq_mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
        fft = fft[freq_mask]

        # Compute cross-spectral density
        csd = (fft.unsqueeze(2) @ fft.conj().unsqueeze(1)).mean(dim=0)

        # Compute coherence
        diag = torch.sqrt(torch.diagonal(csd, dim1=0, dim2=1))
        coherence = torch.abs(csd) / (diag.unsqueeze(0) @ diag.unsqueeze(1))

        return coherence.real  # Return magnitude coherence
