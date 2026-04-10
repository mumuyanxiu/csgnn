"""
Grasp-Aware Attention Module (GAAM).
Core module: attention tailored for grasping.

Design:
1. Edge-aware attention: grasps often lie on object edges; strengthen edge features.
2. Center-stability attention: grasps need central support; emphasize stability.
3. Width-adaptive attention: adjust attention span with object scale.
4. Angle-consistency attention: grasp angle should align with edge normals.
5. Multi-scale grasp fusion: different scales emphasize different grasp cues.

Contributions:
- Encode grasp physics (edges, center, width, angle) as explicit attention.
- Multiple attention paths work together for more accurate, robust grasps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math


class EdgeAwareAttention(nn.Module):
    """
    Edge-aware attention.

    Idea: grasp points often sit on edges; this block boosts edge features.
    Mechanism: Sobel edge magnitude, then higher weights on edge regions.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Args:
            channels: Number of input feature channels
            kernel_size: Edge-detection kernel size
        """
        super(EdgeAwareAttention, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Sobel kernels for edge detection
        # Horizontal edges
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # Vertical edges
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(channels, 1, 1, 1))

        # Edge boost: 1-channel edge_map in, channel attention weights out
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=1),  # single-channel edge_map
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()  # attention in [0, 1]
        )

        # Edge feature extraction
        self.edge_extract = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Edge-enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Edge strength per channel
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=C)  # [B, C, H, W]
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=C)  # [B, C, H, W]
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # [B, C, H, W]

        # 2. Channel-averaged edge map
        edge_map = edge_magnitude.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 3. Edge attention weights
        edge_attention = self.edge_enhance(edge_map)  # [B, C, H, W]

        # 4. Edge features
        edge_features = self.edge_extract(x)

        # 5. Apply edge attention
        enhanced = x + edge_attention * edge_features

        return enhanced


class CenterStabilityAttention(nn.Module):
    """
    Center-stability attention.

    Idea: grasps need support from the object center; emphasize stable regions.
    Mechanism: spatial center bias plus global channel gating.
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input feature channels
        """
        super(CenterStabilityAttention, self).__init__()
        self.channels = channels

        # Global center cue
        self.center_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial center bias from coordinate map
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Feature refinement
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Center-stability-enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Channel attention from global pooling
        channel_attention = self.center_extract(x)  # [B, C, 1, 1]

        # 2. Spatial center bias: closer to image center -> higher weight
        y_coord = torch.arange(H, dtype=torch.float32, device=x.device)
        x_coord = torch.arange(W, dtype=torch.float32, device=x.device)
        y_coord = (y_coord - H / 2) / (H / 2)  # normalize to [-1, 1]
        x_coord = (x_coord - W / 2) / (W / 2)  # normalize to [-1, 1]

        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing='ij')
        coord_map = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        spatial_attention = self.spatial_attention(coord_map)  # [B, 1, H, W]

        # 3. Local feature refinement
        enhanced_features = self.feature_enhance(x)

        # 4. Channel + spatial gating
        enhanced = x + channel_attention * spatial_attention * enhanced_features

        return enhanced


class WidthAdaptiveAttention(nn.Module):
    """
    Width-adaptive (multi-scale) attention.

    Idea: larger objects need a wider effective receptive field.
    Mechanism: estimate scale and mix multi-kernel features.
    """
    def __init__(self, channels: int, num_scales: int = 3):
        """
        Args:
            channels: Number of input feature channels
            num_scales: Number of scales
        """
        super(WidthAdaptiveAttention, self).__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Multi-scale depthwise convs (varying RF)
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=2*i+1, padding=i, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(1, num_scales + 1)  # kernel_size: 3, 5, 7
        ])

        # Scale selection from global context
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_scales, kernel_size=1),
            nn.Softmax(dim=1)  # [B, num_scales, 1, 1]
        )

        # Fuse mixed scale features
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Width-adaptive-enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Multi-scale features
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            feat = conv(x)  # [B, C, H, W]
            multi_scale_features.append(feat)

        # 2. Scale weights
        scale_weights = self.scale_selector(x)  # [B, num_scales, 1, 1]

        # 3. Weighted sum
        adaptive_feat = torch.zeros_like(x)
        for i, feat in enumerate(multi_scale_features):
            weight = scale_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            adaptive_feat += weight * feat

        # 4. Fuse and residual
        enhanced = x + self.fusion(adaptive_feat)

        return enhanced


class AngleConsistencyAttention(nn.Module):
    """
    Angle-consistency attention.

    Idea: grasp orientation should agree with local edge direction.
    Mechanism: predict direction (cos, sin), build consistency map.
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input feature channels
        """
        super(AngleConsistencyAttention, self).__init__()
        self.channels = channels

        # Direction head (cos, sin)
        self.direction_extract = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=1)
        )

        # Consistency attention from features + direction
        self.consistency_attention = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Feature refinement
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Angle-consistency-enhanced features [B, C, H, W]
        """
        # 1. Direction field
        direction = self.direction_extract(x)  # [B, 2, H, W] (cos, sin)

        # 2. Concatenate features and direction
        x_with_dir = torch.cat([x, direction], dim=1)  # [B, C+2, H, W]

        # 3. Consistency attention
        attention = self.consistency_attention(x_with_dir)  # [B, C, H, W]

        # 4. Local refinement
        enhanced_features = self.feature_enhance(x)

        # 5. Gated residual
        enhanced = x + attention * enhanced_features

        return enhanced, direction  # direction for downstream angle prediction


class MultiScaleGraspFusion(nn.Module):
    """
    Multi-scale grasp fusion.

    Idea: fine scales favor precise angle; coarse scales favor quality; mid for width.
    """
    def __init__(self, channels_list: list, out_channels: int = 192):
        """
        Args:
            channels_list: Per-scale channel counts, e.g. [48, 96, 192]
            out_channels: Output channels
        """
        super(MultiScaleGraspFusion, self).__init__()
        self.num_scales = len(channels_list)

        # Project each scale to out_channels
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in channels_list
        ])

        # Learn importance per scale
        self.scale_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sum(channels_list), out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Final conv
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list: list) -> torch.Tensor:
        """
        Args:
            features_list: List of multi-scale features, e.g. [f1, f2, f3]

        Returns:
            Fused features [B, out_channels, H, W]
        """
        B = features_list[0].shape[0]

        # 1. Align channels and spatial size
        adapted_features = []
        for i, feat in enumerate(features_list):
            adapted = self.scale_adapters[i](feat)  # [B, out_channels, H, W]
            if adapted.shape[2:] != features_list[-1].shape[2:]:
                adapted = F.interpolate(
                    adapted, size=features_list[-1].shape[2:],
                    mode='bilinear', align_corners=True
                )
            adapted_features.append(adapted)

        # 2. Scale weights from pooled raw features
        concat_feat = torch.cat([
            F.adaptive_avg_pool2d(feat, 1) for feat in features_list
        ], dim=1)  # [B, sum(channels), 1, 1]

        scale_weights = self.scale_weights(concat_feat)  # [B, num_scales, 1, 1]

        # 3. Weighted fusion
        fused = torch.zeros_like(adapted_features[0])
        for i, feat in enumerate(adapted_features):
            weight = scale_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            fused += weight * feat

        # 4. Final fusion
        output = self.fusion(fused)

        return output


class GraspAwareAttentionModule(nn.Module):
    """
    Grasp-Aware Attention Module (GAAM).

    Stacks edge, center, width, and angle attention, then a final conv and residual.

    Flow:
    Input -> edge -> center -> width -> angle -> conv -> output
           (+ learnable residual from input)

    Contributions:
    1. Encode grasp physics as attention.
    2. Multiple complementary attention paths.
    3. Drop-in block for grasp detectors.
    """
    def __init__(self, channels: int, use_edge: bool = True,
                 use_center: bool = True, use_width: bool = True,
                 use_angle: bool = True):
        """
        Args:
            channels: Input feature channels
            use_edge: Enable edge-aware attention
            use_center: Enable center-stability attention
            use_width: Enable width-adaptive attention
            use_angle: Enable angle-consistency attention
        """
        super(GraspAwareAttentionModule, self).__init__()
        self.channels = channels

        self.edge_attention = EdgeAwareAttention(channels) if use_edge else None
        self.center_attention = CenterStabilityAttention(channels) if use_center else None
        self.width_attention = WidthAdaptiveAttention(channels) if use_width else None
        self.angle_attention = AngleConsistencyAttention(channels) if use_angle else None

        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, return_direction: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            return_direction: If True, also return direction [B, 2, H, W]

        Returns:
            Enhanced features [B, C, H, W], optionally with direction if requested.
        """
        enhanced = x

        if self.edge_attention is not None:
            enhanced = self.edge_attention(enhanced)

        if self.center_attention is not None:
            enhanced = self.center_attention(enhanced)

        if self.width_attention is not None:
            enhanced = self.width_attention(enhanced)

        direction = None
        if self.angle_attention is not None:
            enhanced, direction = self.angle_attention(enhanced)

        output = self.final_fusion(enhanced)

        output = self.residual_weight * x + output

        if return_direction and direction is not None:
            return output, direction
        return output


class GAAMDecoder(nn.Module):
    """
    Decoder stub with GAAM.

    Inserts GAAM after channel reduction to improve grasp maps.
    """
    def __init__(self, input_channels: int = 768, gaam_channels: int = 192,
                 use_gaam: bool = True):
        """
        Args:
            input_channels: Encoder output channels (e.g. Swin)
            gaam_channels: Channels inside GAAM
            use_gaam: Whether to run GAAM
        """
        super(GAAMDecoder, self).__init__()
        self.use_gaam = use_gaam

        self.reduce = nn.Sequential(
            nn.Conv2d(input_channels, gaam_channels, kernel_size=1),
            nn.BatchNorm2d(gaam_channels),
            nn.ReLU(inplace=True)
        )

        if use_gaam:
            self.gaam = GraspAwareAttentionModule(
                channels=gaam_channels,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )

        # Further upsampling layers can follow the original decoder design.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, input_channels, H, W]

        Returns:
            Features [B, gaam_channels, H, W]
        """
        x = self.reduce(x)

        if self.use_gaam:
            x = self.gaam(x)

        return x
