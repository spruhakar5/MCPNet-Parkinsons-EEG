"""
MCPNet: Multiscale Convolutional Prototype Network.

Architecture:
1. Multiscale CNN Encoder — parallel conv branches with different kernel sizes
2. Prototype Computation — class centroids in embedding space
3. Prototype Calibration — adapt prototypes to test subject
4. Distance-based Classification — nearest prototype wins
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_DIM, KERNEL_SIZES, CALIBRATION_ALPHA, FREQ_BANDS


class ConvBranch(nn.Module):
    """
    Single convolutional branch with a specific kernel size.
    Processes the input features through conv → batchnorm → relu → pool.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2  # same padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, in_channels, H, W).

        Returns
        -------
        out : torch.Tensor
            Shape (batch, out_channels).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)           # (batch, out_channels, 1, 1)
        x = x.reshape(x.size(0), -1)  # (batch, out_channels)
        return x


class MultiscaleEncoder(nn.Module):
    """
    Multiscale CNN encoder with parallel branches of different kernel sizes.
    Each branch captures features at a different spatial/spectral scale.
    Outputs from all branches are concatenated into one embedding.
    """

    def __init__(self, in_channels: int, branch_dim: int = 64,
                 kernel_sizes: list = None):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (feature maps).
        branch_dim : int
            Output dimension of each branch.
        kernel_sizes : list of int
            Kernel sizes for each parallel branch.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = KERNEL_SIZES

        self.branches = nn.ModuleList([
            ConvBranch(in_channels, branch_dim, ks)
            for ks in kernel_sizes
        ])

        concat_dim = branch_dim * len(kernel_sizes)
        self.fc = nn.Linear(concat_dim, EMBEDDING_DIM)
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, in_channels, H, W).

        Returns
        -------
        embedding : torch.Tensor
            Shape (batch, EMBEDDING_DIM).
        """
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        embedding = F.relu(self.bn(self.fc(concatenated)))
        return embedding


class MCPNet(nn.Module):
    """
    Multiscale Convolutional Prototype Network.

    Combines:
    - Multiscale CNN encoder for feature embedding
    - Prototype computation from support set
    - Prototype calibration for test-subject adaptation
    - Euclidean distance-based classification
    """

    def __init__(self, n_channels: int = 32, n_bands: int = 5,
                 use_plv: bool = True, calibration_alpha: float = None):
        """
        Parameters
        ----------
        n_channels : int
            Number of EEG channels (default 32).
        n_bands : int
            Number of frequency bands (default 5).
        use_plv : bool
            Whether to include PLV features alongside PSD.
        calibration_alpha : float
            Weight for prototype calibration (0-1).
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.use_plv = use_plv
        self.alpha = calibration_alpha or CALIBRATION_ALPHA

        # PSD encoder: input is (batch, 1, n_channels, n_bands) = (batch, 1, 32, 5)
        self.psd_encoder = MultiscaleEncoder(in_channels=1)

        # PLV encoder: input is (batch, n_bands, n_channels, n_channels) = (batch, 5, 32, 32)
        if use_plv:
            self.plv_encoder = MultiscaleEncoder(in_channels=n_bands)
            # Fusion layer to combine PSD and PLV embeddings
            self.fusion = nn.Sequential(
                nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM),
                nn.BatchNorm1d(EMBEDDING_DIM),
                nn.ReLU(),
            )

    def encode(self, psd: torch.Tensor, plv: torch.Tensor = None) -> torch.Tensor:
        """
        Encode input features into embedding space.

        Parameters
        ----------
        psd : torch.Tensor
            Shape (batch, n_channels, n_bands).
        plv : torch.Tensor, optional
            Shape (batch, n_channels, n_channels, n_bands).

        Returns
        -------
        embedding : torch.Tensor
            Shape (batch, EMBEDDING_DIM).
        """
        # PSD path: reshape to (batch, 1, n_channels, n_bands)
        psd_input = psd.unsqueeze(1)
        psd_emb = self.psd_encoder(psd_input)

        if self.use_plv and plv is not None:
            # PLV path: reshape to (batch, n_bands, n_channels, n_channels)
            plv_input = plv.permute(0, 3, 1, 2).contiguous()
            plv_emb = self.plv_encoder(plv_input)

            # Fuse PSD + PLV embeddings
            combined = torch.cat([psd_emb, plv_emb], dim=1)
            embedding = self.fusion(combined)
        else:
            embedding = psd_emb

        return embedding

    def compute_prototypes(self, support_embeddings: torch.Tensor,
                           support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes as mean of support embeddings per class.

        Parameters
        ----------
        support_embeddings : torch.Tensor
            Shape (n_support, EMBEDDING_DIM).
        support_labels : torch.Tensor
            Shape (n_support,). Class labels (0 or 1).

        Returns
        -------
        prototypes : torch.Tensor
            Shape (n_classes, EMBEDDING_DIM).
        """
        classes = torch.unique(support_labels)
        prototypes = []
        for c in sorted(classes.tolist()):
            mask = support_labels == c
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def calibrate_prototypes(self, prototypes: torch.Tensor,
                             calibration_embeddings: torch.Tensor,
                             calibration_labels: torch.Tensor) -> torch.Tensor:
        """
        Calibrate prototypes using a few samples from the test subject.

        calibrated = alpha * original + (1 - alpha) * subject_mean

        Parameters
        ----------
        prototypes : torch.Tensor
            Shape (n_classes, EMBEDDING_DIM). Original prototypes.
        calibration_embeddings : torch.Tensor
            Shape (n_cal, EMBEDDING_DIM). Embeddings of calibration samples.
        calibration_labels : torch.Tensor
            Shape (n_cal,). Labels of calibration samples.

        Returns
        -------
        calibrated : torch.Tensor
            Shape (n_classes, EMBEDDING_DIM).
        """
        calibrated = prototypes.clone()
        classes = torch.unique(calibration_labels)

        for i, c in enumerate(sorted(classes.tolist())):
            mask = calibration_labels == c
            if mask.any():
                cal_mean = calibration_embeddings[mask].mean(dim=0)
                calibrated[i] = (self.alpha * prototypes[i] +
                                 (1 - self.alpha) * cal_mean)

        return calibrated

    def classify(self, query_embeddings: torch.Tensor,
                 prototypes: torch.Tensor) -> tuple:
        """
        Classify queries by Euclidean distance to prototypes.

        Parameters
        ----------
        query_embeddings : torch.Tensor
            Shape (n_query, EMBEDDING_DIM).
        prototypes : torch.Tensor
            Shape (n_classes, EMBEDDING_DIM).

        Returns
        -------
        log_probs : torch.Tensor
            Shape (n_query, n_classes). Log probabilities.
        predictions : torch.Tensor
            Shape (n_query,). Predicted class indices.
        """
        # Euclidean distance: (n_query, n_classes)
        dists = torch.cdist(query_embeddings, prototypes)

        # Convert distances to log probabilities (negative distance → softmax)
        log_probs = F.log_softmax(-dists, dim=1)
        predictions = torch.argmin(dists, dim=1)

        return log_probs, predictions

    def forward(self, support_psd, support_plv, support_labels,
                query_psd, query_plv,
                calibration_psd=None, calibration_plv=None,
                calibration_labels=None):
        """
        Full forward pass: encode → prototype → (calibrate) → classify.

        Parameters
        ----------
        support_psd : (n_support, n_ch, n_bands)
        support_plv : (n_support, n_ch, n_ch, n_bands) or None
        support_labels : (n_support,)
        query_psd : (n_query, n_ch, n_bands)
        query_plv : (n_query, n_ch, n_ch, n_bands) or None
        calibration_* : optional calibration data

        Returns
        -------
        log_probs : (n_query, n_classes)
        predictions : (n_query,)
        """
        # Encode support and query
        support_emb = self.encode(support_psd, support_plv)
        query_emb = self.encode(query_psd, query_plv)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_emb, support_labels)

        # Optional: calibrate prototypes
        if (calibration_psd is not None and calibration_labels is not None):
            cal_emb = self.encode(calibration_psd, calibration_plv)
            prototypes = self.calibrate_prototypes(
                prototypes, cal_emb, calibration_labels
            )

        # Classify
        log_probs, predictions = self.classify(query_emb, prototypes)

        return log_probs, predictions


if __name__ == "__main__":
    # Quick test with random data
    device = 'cpu'
    model = MCPNet(n_channels=32, n_bands=5, use_plv=True).to(device)

    # Simulate a 2-way 5-shot episode
    k_shot = 5
    n_query = 10

    support_psd = torch.randn(2 * k_shot, 32, 5).to(device)
    support_plv = torch.randn(2 * k_shot, 32, 32, 5).to(device)
    support_labels = torch.tensor([0]*k_shot + [1]*k_shot).to(device)

    query_psd = torch.randn(n_query, 32, 5).to(device)
    query_plv = torch.randn(n_query, 32, 32, 5).to(device)

    log_probs, preds = model(support_psd, support_plv, support_labels,
                             query_psd, query_plv)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Support: {support_psd.shape[0]} samples")
    print(f"Query: {query_psd.shape[0]} samples")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Predictions: {preds.tolist()}")
