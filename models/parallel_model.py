"""
Parallel hybrid grasp network (Parallel Hybrid Grasp Network).
CNN backbone and Swin Transformer run in parallel, then fuse.
Integrates coarse-to-fine grasp-aware attention (CF-GAAM).
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Import classic Swin Transformer from swin.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swin import (
    PatchEmbed, BasicLayer, PatchMerging,
    to_2tuple, trunc_normal_
)

# Grasp-aware attention modules
from .grasp_aware_attention import GraspAwareAttentionModule
from .coarse_to_fine_gaam import CoarseToFineGAAM


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.

    Paper: Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018).
    Idea: L = sum_i exp(-s_i) * L_i + s_i, where s_i = log(sigma_i^2) is a
    learnable log-variance.

    Benefits:
    - Learns task weights automatically; no manual tuning.
    - Grounded in Bayesian maximum likelihood.
    - Weights adapt to per-task uncertainty.
    """
    def __init__(self, num_tasks: int = 4):
        """
        Args:
            num_tasks: Number of tasks (quality, cos, sin, width = 4).
        """
        super(UncertaintyWeightedLoss, self).__init__()
        
        # Learnable log-variance (init 0 => initial weight 1).
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List [p_loss, cos_loss, sin_loss, width_loss].

        Returns:
            total_loss: Total loss after automatic weighting.
        """
        total_loss = 0
        
        for i, loss in enumerate(losses):
            # Precision (weight): precision = exp(-log_var) = 1/sigma^2.
            # Clamp log_vars for numerical stability; [-3, 3] keeps loss well-behaved.
            log_var = torch.clamp(self.log_vars[i], min=-3.0, max=3.0)
            precision = torch.exp(-log_var)
            
            # Standard form: L = (1/sigma^2) * L_i + log(sigma^2), log(sigma^2) = log_var.
            # log(1 + exp(log_var)) as regularizer stays positive.
            weighted_loss = precision * loss
            regularization = torch.log(1 + torch.exp(log_var))  # positive regularizer
            
            total_loss += weighted_loss + regularization
        
        # Keep total loss positive (numerical safety).
        total_loss = torch.clamp(total_loss, min=1e-6)
        
        return total_loss
    
    def get_weights(self) -> torch.Tensor:
        """
        Current learned weights (for visualization / analysis).

        Returns:
            weights: [w1, w2, w3, w4] i.e. [1/sigma1^2, ...].
        """
        return torch.exp(-self.log_vars).detach()
    
    def get_uncertainties(self) -> torch.Tensor:
        """
        Current learned uncertainties (standard deviations).

        Returns:
            sigmas: [sigma1, sigma2, sigma3, sigma4].
        """
        return torch.exp(0.5 * self.log_vars).detach()


class ResidualBlock(nn.Module):
    """
    Basic residual block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            stride: Convolution stride.
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C_in, H, W].

        Returns:
            Output features [B, C_out, H', W'].
        """
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class CNNBackbone(nn.Module):
    """
    CNN backbone: multi-scale features.
    Input: [B, 4, H, W] (RGB-D).
    Output: F1 [B, 48, H/2, W/2], F2 [B, 96, H/4, W/4], F3 [B, 192, H/8, W/8].
    """
    def __init__(self, in_chans: int = 4):
        """
        Args:
            in_chans: Input channels, default 4 (RGB-D).
        """
        super(CNNBackbone, self).__init__()
        
        # Stage 1: H -> H/2, 48 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_chans, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48, 48, stride=1)
        )
        
        # Stage 2: H/2 -> H/4, 96 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 96, stride=1)
        )
        
        # Stage 3: H/4 -> H/8, 192 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(96, 192, stride=2),
            ResidualBlock(192, 192, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input image [B, 4, H, W].

        Returns:
            (F1, F2, F3): Feature maps at three scales.
        """
        f1 = self.stage1(x)     # [B, 48, H/2, W/2]
        f2 = self.stage2(f1)    # [B, 96, H/4, W/4]
        f3 = self.stage3(f2)    # [B, 192, H/8, W/8]
        
        return f1, f2, f3


# No PatchifyForSwin: Swin takes 4-channel input directly.
# PatchEmbed supports arbitrary channel count; no extra reduction needed.


class SwinTransformerBranch(nn.Module):
    """
    Swin Transformer branch (parallel path) with fixed classic config.
    Input: [B, 4, H, W] (RGB-D; H, W variable, e.g. 224 or 300).
    Output: [B, 192, H/16, W/16].

    Uses 4-channel input directly (depth preserved).
    Four encoder stages (aligned with serial_model).
    Fixed custom config; no model_size parameter.
    """
    def __init__(self, in_chans: int = 4, pretrained: bool = False):
        """
        Args:
            in_chans: Input channels, default 4 (RGB-D).
            pretrained: Unused (kept for API compatibility).
        """
        super(SwinTransformerBranch, self).__init__()
        
        # Fixed custom config (not tied to model_size)
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        
        print(f"[Swin branch] Classic Swin Transformer with fixed config (custom-fixed, pretrained=False)")
        print(f"  - Input channels: {in_chans} (RGB-D, depth preserved)")
        print(f"  - embed_dim: {embed_dim}")
        print(f"  - depths: {depths}")
        print(f"  - num_heads: {num_heads}")
        
        # Internal image size 224x224 for the backbone
        self.img_size = 224
        self.patch_size = 4
        self.in_chans = in_chans  # e.g. 4 for RGB-D
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = 7
        self.mlp_ratio = 4.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.num_layers = len(self.depths)
        
        # Resize inputs to 224x224
        self.resize_to_224 = nn.AdaptiveAvgPool2d((224, 224))
        
        # Patch Embedding
        patches_resolution = [self.img_size // self.patch_size, self.img_size // self.patch_size]
        self.patches_resolution = patches_resolution
        
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm
        )
        num_patches = self.patch_embed.num_patches
        
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        # Encoder stages (4 layers down to H/16): H -> H/2 -> H/4 -> H/8 -> H/16
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(4):  # all 4 stages
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < 3) else None,  # downsample first 3 stages
                use_checkpoint=False
            )
            self.layers.append(layer)
        
        # Stage-4 output channels
        # Stage0: 96 -> 192 (downsample)
        # Stage1: 192 -> 384 (downsample)
        # Stage2: 384 -> 768 (downsample)
        # Stage3: 768 -> 768 (no downsample)
        stage4_channels = int(self.embed_dim * 2 ** 3)  # 96 * 8 = 768 (tiny)
        
        # Final norm
        self.norm = nn.LayerNorm(stage4_channels)
        
        # Channel adapter: Swin -> 192 (match F3)
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(stage4_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Weight init (same as swin.py)."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, 4, H, W] (RGB-D; arbitrary H, W).

        Returns:
            Features [B, 192, H/16, W/16].
        """
        x = self.resize_to_224(x)  # [B, 4, 224, 224]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 56*56, 96] (H/4 * W/4, embed_dim)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)  # [B, L, C] per stage
        
        x = self.norm(x)  # [B, 7*7, 768] (H/16 * W/16, 768)
        
        # [B, 7*7, 768] -> [B, 768, 7, 7]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 7, 7]
        
        x = self.channel_adapt(x)  # [B, 192, 7, 7]
        
        return x


class CrossAttentionModule(nn.Module):
    """
    Cross-attention: two streams query each other.
    CNN queries Swin; Swin queries CNN.
    """
    def __init__(self, channels: int = 192, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            channels: Input/output channels.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super(CrossAttentionModule, self).__init__()
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # CNN -> Swin cross-attention
        self.q_cnn = nn.Conv2d(channels, channels, 1)
        self.k_swin = nn.Conv2d(channels, channels, 1)
        self.v_swin = nn.Conv2d(channels, channels, 1)
        
        # Swin -> CNN cross-attention
        self.q_swin = nn.Conv2d(channels, channels, 1)
        self.k_cnn = nn.Conv2d(channels, channels, 1)
        self.v_cnn = nn.Conv2d(channels, channels, 1)
        
        # Output projection
        self.proj_cnn = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout)
        )
        self.proj_swin = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout)
        )
        
        # Layer Norm
        self.norm_cnn = nn.LayerNorm(channels)
        self.norm_swin = nn.LayerNorm(channels)
        
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, C, H, W].

        Returns:
            Attention output [B, C, H, W].
        """
        B, C, H, W = q.shape
        
        # Reshape: [B, C, H, W] -> [B, num_heads, H*W, head_dim]
        q = q.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Attention scores: [B, num_heads, H*W, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum: [B, num_heads, H*W, head_dim]
        out = attn @ v
        
        # Reshape back: [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        return out
    
    def forward(self, f_cnn: torch.Tensor, f_swin: torch.Tensor) -> tuple:
        """
        Args:
            f_cnn: CNN features [B, C, H, W].
            f_swin: Swin features [B, C, H, W].

        Returns:
            (enhanced CNN features, enhanced Swin features).
        """
        B, C, H, W = f_cnn.shape
        
        # 1. CNN queries Swin (global context from Swin)
        q_c = self.q_cnn(f_cnn)
        k_s = self.k_swin(f_swin)
        v_s = self.v_swin(f_swin)
        cnn_enhanced = self._attention(q_c, k_s, v_s)
        cnn_enhanced = self.proj_cnn(cnn_enhanced)
        
        # Residual + LayerNorm
        f_cnn_out = f_cnn + cnn_enhanced
        f_cnn_out = self.norm_cnn(f_cnn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 2. Swin queries CNN (local detail from CNN)
        q_s = self.q_swin(f_swin)
        k_c = self.k_cnn(f_cnn)
        v_c = self.v_cnn(f_cnn)
        swin_enhanced = self._attention(q_s, k_c, v_c)
        swin_enhanced = self.proj_swin(swin_enhanced)
        
        # Residual + LayerNorm
        f_swin_out = f_swin + swin_enhanced
        f_swin_out = self.norm_swin(f_swin_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return f_cnn_out, f_swin_out


class FusionBlock(nn.Module):
    """
    Fusion block: cross-attention between CNN and Swin features.
    Bidirectional queries for deep interaction.
    """
    def __init__(self, cnn_channels: int = 192, swin_channels: int = 192, 
                 out_channels: int = 192, num_heads: int = 8):
        """
        Args:
            cnn_channels: CNN feature channels (F3).
            swin_channels: Swin feature channels.
            out_channels: Output channels.
            num_heads: Cross-attention heads.
        """
        super(FusionBlock, self).__init__()
        
        print(f"[FusionBlock] Cross-Attention mode (num_heads={num_heads})")
        self.cross_attention = CrossAttentionModule(
            channels=cnn_channels,  # expect cnn_channels == swin_channels
            num_heads=num_heads
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(cnn_channels + swin_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, f_cnn: torch.Tensor, f_swin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_cnn: CNN features [B, 192, H/8, W/8].
            f_swin: Swin features [B, 192, H/16, W/16] (upsampled to H/8 if needed).

        Returns:
            Fused features [B, 192, H/8, W/8].
        """
        # Match spatial size
        if f_cnn.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_cnn.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        f_cnn_enh, f_swin_enh = self.cross_attention(f_cnn, f_swin)
        # Concatenate enhanced features
        x = torch.cat([f_cnn_enh, f_swin_enh], dim=1)  # [B, 384, H/8, W/8]
        x = self.fusion(x)  # [B, 192, H/8, W/8]
        
        return x


class GGCNNDecoder(nn.Module):
    """
    GGCNN-style decoder with optional GAAM / CF-GAAM.
    Two upsampling stages + skip connections (F2, F1) + grasp-aware attention.
    Outputs: Q (quality), angle (sin, cos), W (width).

    GAAM/CF-GAAM blocks are placed at key depths to improve grasp maps.
    """
    def __init__(self, fusion_channels: int = 192, use_gaam: bool = False,
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        Args:
            fusion_channels: Fused feature channels.
            use_gaam: Use grasp-aware attention (GAAM).
            use_cf_gaam: Use coarse-to-fine GAAM (CF-GAAM).
            num_peaks: Number of peaks for CF-GAAM.
        """
        super(GGCNNDecoder, self).__init__()
        self.use_gaam = use_gaam and not use_cf_gaam  # CF-GAAM replaces plain GAAM
        self.use_cf_gaam = use_cf_gaam
        
        # GAAM/CF-GAAM 1: after fused features
        if use_cf_gaam:
            self.cf_gaam1 = CoarseToFineGAAM(
                channels=fusion_channels,
                use_coarse_fine=True,
                num_peaks=num_peaks,
                use_gaam=True,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        elif use_gaam:
            self.gaam1 = GraspAwareAttentionModule(
                channels=fusion_channels,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Upsample 1: H/8 -> H/4
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(fusion_channels, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Conv after concat with F2
        self.conv_skip2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 2: after F2 skip (H/4)
        if use_cf_gaam:
            self.cf_gaam2 = CoarseToFineGAAM(
                channels=96,
                use_coarse_fine=True,
                num_peaks=num_peaks,
                use_gaam=True,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        elif use_gaam:
            self.gaam2 = GraspAwareAttentionModule(
                channels=96,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Upsample 2: H/4 -> H/2
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Conv after concat with F1
        self.conv_skip1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 3: after F1 skip (H/2)
        if use_cf_gaam:
            self.cf_gaam3 = CoarseToFineGAAM(
                channels=48,
                use_coarse_fine=True,
                num_peaks=num_peaks,
                use_gaam=True,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        elif use_gaam:
            self.gaam3 = GraspAwareAttentionModule(
                channels=48,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Final features + upsample toward full resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # x2 upsample toward input resolution
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 4: before heads (full-res refinement)
        if use_cf_gaam:
            self.cf_gaam4 = CoarseToFineGAAM(
                channels=32,
                use_coarse_fine=True,
                num_peaks=num_peaks,
                use_gaam=True,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        elif use_gaam:
            self.gaam4 = GraspAwareAttentionModule(
                channels=32,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Separate output heads (see swin.py)
        # Pos: quality / placement [0, 1]
        self.pos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Cos: angle cosine [-1, 1]
        self.cos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Sin: angle sine [-1, 1]
        self.sin_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Width: grasp width [0, inf)
        self.width_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Fused features [B, 192, H/8, W/8].
            f1: CNN F1 [B, 48, H/2, W/2].
            f2: CNN F2 [B, 96, H/4, W/4].

        Returns:
            Dict with 'pos', 'cos', 'sin', 'width'.
        """
        # GAAM/CF-GAAM 1
        if self.use_cf_gaam:
            x = self.cf_gaam1(x)  # [B, 192, H/8, W/8]
        elif self.use_gaam:
            x = self.gaam1(x)  # [B, 192, H/8, W/8]
        
        x = self.upsample1(x)  # [B, 96, H/4, W/4]
        
        # Concat F2 (match sizes)
        if x.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f2], dim=1)  # [B, 192, H/4, W/4]
        x = self.conv_skip2(x)  # [B, 96, H/4, W/4]
        
        # GAAM/CF-GAAM 2 after F2 skip
        if self.use_cf_gaam:
            x = self.cf_gaam2(x)  # [B, 96, H/4, W/4]
        elif self.use_gaam:
            x = self.gaam2(x)  # [B, 96, H/4, W/4]
        
        x = self.upsample2(x)  # [B, 48, H/2, W/2]
        
        # Concat F1 (match sizes)
        if x.shape[2:] != f1.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f1], dim=1)  # [B, 96, H/2, W/2]
        x = self.conv_skip1(x)  # [B, 48, H/2, W/2]
        
        # GAAM/CF-GAAM 3 after F1 skip
        if self.use_cf_gaam:
            x = self.cf_gaam3(x)  # [B, 48, H/2, W/2]
        elif self.use_gaam:
            x = self.gaam3(x)  # [B, 48, H/2, W/2]
        
        x = self.final_conv(x)  # [B, 32, H, W] near full res
        
        # GAAM/CF-GAAM 4 before heads
        if self.use_cf_gaam:
            x = self.cf_gaam4(x)  # [B, 32, H, W]
        elif self.use_gaam:
            x = self.gaam4(x)  # [B, 32, H, W]
        
        pos_output = self.pos_output(x)   # [B, 1, H, W] quality
        cos_output = self.cos_output(x)   # [B, 1, H, W] cos
        sin_output = self.sin_output(x)   # [B, 1, H, W] sin
        width_output = self.width_output(x)  # [B, 1, H, W] width
        
        return {"pos": pos_output, "cos": cos_output, "sin": sin_output, "width": width_output}


class ParallelHybridGraspNet(nn.Module):
    """
    Parallel hybrid grasp network.

    Layout:
    Input RGB-D [B, 4, H, W] (variable size, e.g. 224 or 300)
        -> CNN backbone
           -> F1 [B, 48, H/2, W/2] for skip connection
           -> F2 [B, 96, H/4, W/4] for skip connection
           -> F3 [B, 192, H/8, W/8] for fusion
        -> Swin Transformer with 4 encoder stages
           -> S_out [B, 192, H/16, W/16], then upsampled for fusion
        -> FusionBlock(F3, S_out)
        -> Decoder
           -> Skip from F2
           -> Skip from F1
        -> Output: Pos, Cos, Sin, Width
    """
    
    def __init__(self, in_chans: int = 4, input_channels: int = None,
                 use_pretrained: bool = True, num_heads: int = 8, 
                 use_uncertainty_loss: bool = True, use_gaam: bool = False, 
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        Args:
            in_chans: Input channels, default 4 (RGB-D).
            input_channels: Alias for in_chans (compatibility).
            use_pretrained: Unused Swin pretrained flag (API compatibility).
            num_heads: Cross-attention heads in fusion.
            use_uncertainty_loss: Enable uncertainty-weighted multi-task loss.
            use_gaam: Enable grasp-aware attention (GAAM).
            use_cf_gaam: Enable coarse-to-fine GAAM (CF-GAAM).
            num_peaks: Peak count for CF-GAAM.
        """
        super(ParallelHybridGraspNet, self).__init__()
        
        if input_channels is not None:
            in_chans = input_channels
        
        print(f"[Model] Initializing ParallelHybridGraspNet (parallel version)")
        print(f"  - input channels: {in_chans} (RGB-D)")
        print(f"  - Swin branch: fixed configuration (custom-fixed)")
        print(f"  - fusion strategy: Cross-Attention (bidirectional querying)")
        print(f"  - number of attention heads: {num_heads}")
        print(f"  - uncertainty loss: {'enabled' if use_uncertainty_loss else 'disabled'}")
        if use_cf_gaam:
            print(f"  - Coarse-to-Fine Grasp-Aware Attention (CF-GAAM): enabled")
            print(f"    - number of peaks: {num_peaks}")
        else:
            print(f"  - Grasp-Aware Attention (GAAM): {'enabled' if use_gaam else 'disabled'}")
        
        self.cnn_backbone = CNNBackbone(in_chans=in_chans)
        
        self.swin_branch = SwinTransformerBranch(
            in_chans=in_chans,
            pretrained=use_pretrained
        )
        
        self.fusion = FusionBlock(
            cnn_channels=192,
            swin_channels=192,
            out_channels=192,
            num_heads=num_heads
        )
        
        self.decoder = GGCNNDecoder(
            fusion_channels=192,
            use_gaam=use_gaam,
            use_cf_gaam=use_cf_gaam,
            num_peaks=num_peaks
        )
        
        self.use_uncertainty_loss = use_uncertainty_loss
        if use_uncertainty_loss:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=4)
            print(f"  - automatic loss weighting: enabled (the model will learn task weights)")
        
        print(f"[Model] Initialization complete.")
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input [B, 4, H, W] (variable size).
            verbose: Print intermediate tensor shapes.

        Returns:
            Dict with 'pos', 'cos', 'sin', 'width' maps [B, 1, H, W].
        """
        if verbose:
            print(f"Input: {x.shape}")
        
        f1, f2, f3 = self.cnn_backbone(x)
        if verbose:
            print(f"[CNN branch] F1: {f1.shape}, F2: {f2.shape}, F3: {f3.shape}")
        
        s_out = self.swin_branch(x)  # [B, 192, H/16, W/16] (7x7 for 224x224 input)
        if verbose:
            print(f"[Swin branch] Output: {s_out.shape}")
        
        fused = self.fusion(f3, s_out)  # [B, 192, H/8, W/8]
        if verbose:
            print(f"[Fusion] Output: {fused.shape}")
        
        outputs = self.decoder(fused, f1, f2)
        if verbose:
            print(f"[Output] Pos: {outputs['pos'].shape}, Cos: {outputs['cos'].shape}, Sin: {outputs['sin'].shape}, Width: {outputs['width'].shape}")
        
        return outputs
    
    def compute_loss(self, xc: torch.Tensor, yc: list, 
                     loss_weights: Dict[str, float] = None) -> Dict[str, any]:
        """
        Loss for GGCNN-style training.

        Modes:
        1. Uncertainty weighting (use_uncertainty_loss=True): learned task weights.
        2. Manual weights (use_uncertainty_loss=False): loss_weights dict.

        Args:
            xc: Input [B, C, H, W].
            yc: Targets [pos_img, cos_img, sin_img, width_img].
            loss_weights: Manual weights when uncertainty loss is off.

        Returns:
            Dict with 'loss', 'losses' (per-term + optional learned stats), 'pred'.
        """
        outputs = self.forward(xc)
        
        pos_pred = outputs['pos']       # [B, 1, H', W']
        cos_pred = outputs['cos']       # [B, 1, H', W']
        sin_pred = outputs['sin']       # [B, 1, H', W']
        width_pred = outputs['width']   # [B, 1, H', W']
        
        pos_gt, cos_gt, sin_gt, width_gt = yc
        
        if pos_pred.shape[2:] != pos_gt.shape[2:]:
            pos_pred = F.interpolate(pos_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            sin_pred = F.interpolate(sin_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            cos_pred = F.interpolate(cos_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            width_pred = F.interpolate(width_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
        
        p_loss_raw = F.mse_loss(pos_pred, pos_gt)
        cos_loss_raw = F.mse_loss(cos_pred, cos_gt)
        sin_loss_raw = F.mse_loss(sin_pred, sin_gt)
        width_loss_raw = F.mse_loss(width_pred, width_gt)
        
        if self.use_uncertainty_loss:
            losses_list = [p_loss_raw, cos_loss_raw, sin_loss_raw, width_loss_raw]
            total_loss = self.uncertainty_loss(losses_list)
            
            learned_weights = self.uncertainty_loss.get_weights()
            learned_sigmas = self.uncertainty_loss.get_uncertainties()
            
            p_loss = p_loss_raw * learned_weights[0]
            cos_loss = cos_loss_raw * learned_weights[1]
            sin_loss = sin_loss_raw * learned_weights[2]
            width_loss = width_loss_raw * learned_weights[3]
            
        else:
            if loss_weights is None:
                loss_weights = {
                    'p': 1.5,
                    'cos': 1.0,
                    'sin': 1.0,
                    'width': 0.8
                }
            
            p_loss = p_loss_raw * loss_weights['p']
            cos_loss = cos_loss_raw * loss_weights['cos']
            sin_loss = sin_loss_raw * loss_weights['sin']
            width_loss = width_loss_raw * loss_weights['width']
            
            total_loss = p_loss + cos_loss + sin_loss + width_loss
        
        loss_dict = {
            'p_loss': p_loss,
            'cos_loss': cos_loss,
            'sin_loss': sin_loss,
            'width_loss': width_loss
        }
        
        if self.use_uncertainty_loss:
            loss_dict.update({
                'learned_weight_p': learned_weights[0],
                'learned_weight_cos': learned_weights[1],
                'learned_weight_sin': learned_weights[2],
                'learned_weight_width': learned_weights[3],
                'uncertainty_p': learned_sigmas[0],
                'uncertainty_cos': learned_sigmas[1],
                'uncertainty_sin': learned_sigmas[2],
                'uncertainty_width': learned_sigmas[3],
            })
        
        return {
            'loss': total_loss,
            'losses': loss_dict,
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
