"""
Coarse-to-Fine Grasp-Aware Attention Module (CF-GAAM).
GAAM extended with a coarse-to-fine prediction stack.

Main ideas:
1. Coarse stage: predict a grasp probability heatmap (spatial distribution of valid grasps).
2. Fine stage: Gaussian attention around heatmap peaks to focus on key regions.
3. Gaussian spatial prior: valid grasps tend to cluster smoothly in space.

Design:
- Valid grasps often form continuous spatial patterns.
- Model that with Gaussians centered on coarse peaks.
- Coarse: global "where might we grasp"; fine: local Gaussian focus.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .grasp_aware_attention import GraspAwareAttentionModule


class GaussianSpatialDistribution(nn.Module):
    """
    Gaussian spatial weighting around peak locations.

    Idea: smooth spatial prior for valid grasp poses.
    Mechanism: Gaussians centered on heatmap peaks.
    """
    def __init__(self, sigma_init: float = 1.5, learnable_sigma: bool = True):
        """
        Args:
            sigma_init: Initial Gaussian standard deviation
            learnable_sigma: If True, sigma is a learned parameter
        """
        super(GaussianSpatialDistribution, self).__init__()
        self.learnable_sigma = learnable_sigma

        if learnable_sigma:
            self.sigma = nn.Parameter(torch.ones(1) * sigma_init)
        else:
            self.register_buffer('sigma', torch.tensor(sigma_init))

    def generate_gaussian_map(self, centers: torch.Tensor, H: int, W: int,
                             sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build a Gaussian weight map from peak centers.

        Args:
            centers: Peak locations [B, N, 2] (x, y) per peak
            H: Map height
            W: Map width
            sigma: Std per batch/peak [B, N] or scalar; None uses self.sigma

        Returns:
            Gaussian map [B, 1, H, W]
        """
        B, N, _ = centers.shape
        device = centers.device

        if sigma is None:
            sigma = self.sigma

        # Normalize sigma shape to [B, N]
        if sigma.dim() == 0:
            # Scalar -> [B, N]
            sigma = sigma.unsqueeze(0).unsqueeze(0).expand(B, N)
        elif sigma.dim() == 1:
            if sigma.shape[0] == 1:
                # [1] -> [B, N]
                sigma = sigma.unsqueeze(0).expand(B, N)
            elif sigma.shape[0] == B:
                # [B] -> [B, N]
                sigma = sigma.unsqueeze(1).expand(B, N)
            elif sigma.shape[0] == N:
                # [N] -> [B, N]
                sigma = sigma.unsqueeze(0).expand(B, -1)
            else:
                raise ValueError(f"Unexpected sigma shape: {sigma.shape}, expected scalar, [1], [B], [N], or [B, N]")
        elif sigma.dim() == 2:
            if sigma.shape == (B, 1):
                sigma = sigma.expand(B, N)
            elif sigma.shape == (1, N):
                sigma = sigma.expand(B, -1)
            elif sigma.shape == (B, N):
                pass
            else:
                raise ValueError(f"Unexpected sigma shape: {sigma.shape}, expected [B, 1], [1, N], or [B, N]")
        else:
            raise ValueError(f"Unexpected sigma dimensions: {sigma.dim()}, expected 0, 1, or 2")

        y_coords = torch.arange(H, dtype=torch.float32, device=device)
        x_coords = torch.arange(W, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)  # [B, 1, H, W]
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)  # [B, 1, H, W]

        gaussian_maps = []
        for i in range(N):
            center = centers[:, i, :]  # [B, 2]
            center_x = center[:, 0].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            center_y = center[:, 1].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            sigma_i = sigma[:, i].view(B, 1, 1, 1)  # [B, 1, 1, 1]

            dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2

            gaussian = torch.exp(-dist_sq / (2 * sigma_i ** 2 + 1e-6))  # [B, 1, H, W]

            if gaussian.dim() == 4 and gaussian.shape == (B, 1, H, W):
                gaussian_maps.append(gaussian)
            else:
                gaussian = gaussian.view(B, 1, H, W)
                gaussian_maps.append(gaussian)

        # Max over peaks to limit overlap boosting
        if len(gaussian_maps) > 0:
            gaussian_map = torch.stack(gaussian_maps, dim=1)  # [B, N, 1, H, W]
            gaussian_map, _ = torch.max(gaussian_map, dim=1)  # [B, 1, H, W]
        else:
            gaussian_map = torch.zeros(B, 1, H, W, device=device)

        return gaussian_map

    def forward(self, centers: torch.Tensor, H: int, W: int,
                sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            centers: Peak locations [B, N, 2]
            H: Map height
            W: Map width
            sigma: Optional std override

        Returns:
            Gaussian map [B, 1, H, W]
        """
        return self.generate_gaussian_map(centers, H, W, sigma)


class CoarseStagePredictor(nn.Module):
    """
    Coarse-stage head: per-pixel probability of a valid grasp.

    Output: heatmap [B, 1, H, W].
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        """
        Args:
            in_channels: Input feature channels
            hidden_channels: Hidden width
        """
        super(CoarseStagePredictor, self).__init__()

        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.probability_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]

        Returns:
            Grasp probability map [B, 1, H, W]
        """
        feat = self.feature_extract(features)
        prob_map = self.probability_head(feat)
        return prob_map


class FineStageGaussianAttention(nn.Module):
    """
    Fine stage: Gaussian attention from coarse peaks, reweight features.

    Mechanism: emphasize regions around detected peaks.
    """
    def __init__(self, in_channels: int, num_peaks: int = 5,
                 sigma_init: float = 1.5, learnable_sigma: bool = True):
        """
        Args:
            in_channels: Feature channels
            num_peaks: Number of peaks to keep
            sigma_init: Initial Gaussian std
            learnable_sigma: Learn sigma or fix it
        """
        super(FineStageGaussianAttention, self).__init__()
        self.num_peaks = num_peaks

        self.gaussian_dist = GaussianSpatialDistribution(
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma
        )

        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.attention_adapter = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def detect_peaks(self, prob_map: torch.Tensor, min_distance: int = 20,
                     threshold: float = 0.2) -> torch.Tensor:
        """
        Peak detection on GPU (no CPU round-trip).

        Args:
            prob_map: [B, 1, H, W]
            min_distance: Minimum spacing between peaks
            threshold: Response threshold

        Returns:
            Peak coordinates [B, N, 2] (x, y)
        """
        B, _, H, W = prob_map.shape
        device = prob_map.device

        all_peaks = []
        for b in range(B):
            prob = prob_map[b, 0]  # [H, W]

            prob_unsqueezed = prob.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            kernel_size = min_distance * 2 + 1
            padding = min_distance
            local_max = F.max_pool2d(prob_unsqueezed, kernel_size=kernel_size,
                                    stride=1, padding=padding)
            local_max = F.interpolate(local_max, size=(H, W), mode='bilinear', align_corners=False)
            local_max = local_max.squeeze()  # [H, W]

            mask = (prob == local_max) & (prob > threshold)

            y_coords, x_coords = torch.where(mask)

            if len(y_coords) > 0:
                peak_probs = prob[y_coords, x_coords]
                _, top_indices = torch.topk(peak_probs, min(self.num_peaks, len(y_coords)))
                y_coords = y_coords[top_indices]
                x_coords = x_coords[top_indices]

                peaks = torch.stack([x_coords.float(), y_coords.float()], dim=1)  # [N, 2]
            else:
                peaks = torch.tensor([[W//2, H//2]], dtype=torch.float32, device=device)

            if len(peaks) == 0:
                peaks = torch.tensor([[W//2, H//2]], dtype=torch.float32, device=device)

            if len(peaks) < self.num_peaks:
                padding = peaks[0:1].repeat(self.num_peaks - len(peaks), 1)
                peaks = torch.cat([peaks, padding], dim=0)
            elif len(peaks) > self.num_peaks:
                peaks = peaks[:self.num_peaks]

            all_peaks.append(peaks)

        peaks_tensor = torch.stack(all_peaks, dim=0)  # [B, N, 2]
        return peaks_tensor

    def forward(self, features: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            prob_map: Coarse probability [B, 1, H, W]

        Returns:
            Gaussian-refined features [B, C, H, W]
        """
        B, C, H, W = features.shape

        peaks = self.detect_peaks(prob_map)  # [B, N, 2]

        gaussian_attention = self.gaussian_dist(peaks, H, W)  # [B, 1, H, W]

        enhanced_features = self.feature_enhance(features)

        feat_with_attn = torch.cat([features, gaussian_attention], dim=1)  # [B, C+1, H, W]
        adaptive_attention = self.attention_adapter(feat_with_attn)  # [B, 1, H, W]

        final_attention = (gaussian_attention + adaptive_attention) / 2
        output = features + final_attention * enhanced_features

        return output, peaks, gaussian_attention


class CoarseToFineGAAM(nn.Module):
    """
    Coarse-to-Fine GAAM.

    Pipeline:
    Input -> coarse heatmap -> fine Gaussian attention -> base GAAM -> fusion + residual

    Contributions:
    1. Couples coarse-to-fine structure with grasp-aware attention.
    2. Gaussian maps encode smooth spatial grasp distributions.
    3. Global coarse cue plus local fine focus.
    """
    def __init__(self, channels: int,
                 use_coarse_fine: bool = True,
                 num_peaks: int = 5,
                 sigma_init: float = 1.5,
                 use_gaam: bool = True,
                 use_edge: bool = True,
                 use_center: bool = True,
                 use_width: bool = True,
                 use_angle: bool = True):
        """
        Args:
            channels: Input feature channels
            use_coarse_fine: Enable coarse-to-fine branch
            num_peaks: Peaks for Gaussian attention
            sigma_init: Initial Gaussian std
            use_gaam: Enable base GAAM
            use_edge: Edge attention in GAAM
            use_center: Center attention in GAAM
            use_width: Width attention in GAAM
            use_angle: Angle attention in GAAM
        """
        super(CoarseToFineGAAM, self).__init__()
        self.channels = channels
        self.use_coarse_fine = use_coarse_fine
        self.use_gaam = use_gaam

        if use_coarse_fine:
            self.coarse_predictor = CoarseStagePredictor(channels)
            self.fine_attention = FineStageGaussianAttention(
                channels,
                num_peaks=num_peaks,
                sigma_init=sigma_init,
                learnable_sigma=True
            )

        if use_gaam:
            self.gaam = GraspAwareAttentionModule(
                channels=channels,
                use_edge=use_edge,
                use_center=use_center,
                use_width=use_width,
                use_angle=use_angle
            )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            return_aux: If True and coarse-fine is on, return (prob_map, peaks, gaussian_attn)

        Returns:
            Enhanced [B, C, H, W], optionally with auxiliary tensors.
        """
        enhanced = x
        prob_map = None
        peaks = None
        gaussian_attn = None

        if self.use_coarse_fine:
            prob_map = self.coarse_predictor(enhanced)  # [B, 1, H, W]
            enhanced, peaks, gaussian_attn = self.fine_attention(enhanced, prob_map)

        if self.use_gaam:
            enhanced = self.gaam(enhanced)

        output = self.final_fusion(enhanced)

        output = self.residual_weight * x + output

        if return_aux and self.use_coarse_fine:
            return output, prob_map, peaks, gaussian_attn
        return output


class CFGAAMLoss(nn.Module):
    """
    Loss for coarse-to-fine GAAM.

    1. Coarse: heatmap vs. ground-truth grasp locations.
    2. Fine: encourage high grasp quality inside high Gaussian-attention regions.
    """
    def __init__(self, coarse_weight: float = 1.0, fine_weight: float = 1.0):
        """
        Args:
            coarse_weight: Weight for coarse heatmap loss
            fine_weight: Weight for fine Gaussian alignment loss
        """
        super(CFGAAMLoss, self).__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, prob_map: torch.Tensor, gt_pos: torch.Tensor,
                peaks: torch.Tensor, gaussian_attn: torch.Tensor) -> dict:
        """
        Args:
            prob_map: Predicted heatmap [B, 1, H, W]
            gt_pos: Ground-truth grasp map [B, 1, H, W]
            peaks: Detected peaks [B, N, 2]
            gaussian_attn: Gaussian attention [B, 1, H, W]

        Returns:
            Dict of coarse_loss, fine_loss, total_loss
        """
        coarse_loss = self.mse_loss(prob_map, gt_pos)

        # Encourage overlap between high attention and high-quality regions
        fine_loss = -torch.mean(gaussian_attn * gt_pos)

        total_loss = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss

        return {
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss,
            'total_loss': total_loss
        }
