"""
Hybrid Grasp Network.
Combines CNN backbone + Swin Transformer + GGCNN-style decoder,
with the Grasp-Aware Attention Module (GAAM).
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Classic Swin Transformer from swin.py
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
    Uncertainty-weighted multi-task loss (shared with parallel_model).

    Paper: Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)
    Objective: L = sum_i exp(-s_i) * L_i + s_i
    """
    def __init__(self, num_tasks: int = 4):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss
    
    def get_weights(self) -> torch.Tensor:
        return torch.exp(-self.log_vars).detach()
    
    def get_uncertainties(self) -> torch.Tensor:
        return torch.exp(0.5 * self.log_vars).detach()


class ResidualBlock(nn.Module):
    """
    Basic residual block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
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
            x: Input features [B, C_in, H, W]
        
        Returns:
            Output features [B, C_out, H', W']
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
    Input: [B, 4, H, W] (RGB-D)
    Output: F1 [B, 48, H/2, W/2], F2 [B, 96, H/4, W/4], F3 [B, 192, H/8, W/8]
    """
    def __init__(self, in_chans: int = 4):
        """
        Args:
            in_chans: Input channels, default 4 (RGB-D)
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
            x: Input image [B, 4, H, W]
        
        Returns:
            (F1, F2, F3): Feature maps at three scales
        """
        f1 = self.stage1(x)     # [B, 48, H/2, W/2]
        f2 = self.stage2(f1)    # [B, 96, H/4, W/4]
        f3 = self.stage3(f2)    # [B, 192, H/8, W/8]
        
        return f1, f2, f3


class ChannelAdapter(nn.Module):
    """
    Channel adapter: map CNN feature channels to Swin input requirements.
    """
    def __init__(self, in_channels: int = 96, out_channels: int = 3):
        """
        Args:
            in_channels: Input channels (from CNN)
            out_channels: Output channels (Swin expects 3-channel RGB)
        """
        super(ChannelAdapter, self).__init__()
        
        self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, in_channels, H, W]
        
        Returns:
            Adapted features [B, out_channels, H, W]
        """
        return self.adapt(x)


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block using the classic implementation (no pretrained weights).
    Input: [B, 3, 56, 56]
    Output: [B, 768, 7, 7] for Swin-Tiny (stage-4 output)
    """
    def __init__(self, img_size: int = 56, pretrained: bool = False, model_size: str = 'tiny'):
        """
        Args:
            img_size: Input spatial size (56 for F2 features)
            pretrained: Unused; kept for API compatibility
            model_size: Model variant ('tiny', 'small', 'base')
        """
        super(SwinTransformerBlock, self).__init__()
        
        # Config (classic swin.py layout)
        configs = {
            'tiny': {
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'stage4_channels': 768  # Stage-4 output channels
            },
            'small': {
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
                'stage4_channels': 768
            },
            'base': {
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'stage4_channels': 1024
            }
        }
        
        config = configs.get(model_size, configs['tiny'])
        self.out_channels = config['stage4_channels']
        
        print(f"[Swin] Classic Swin Transformer (model_size={model_size}, pretrained=False)")
        print(f"  - embed_dim: {config['embed_dim']}")
        print(f"  - depths: {config['depths']}")
        print(f"  - num_heads: {config['num_heads']}")
        print(f"  - output channels: {self.out_channels}")
        
        # Input size (upsampled to 224x224 for Swin)
        self.input_img_size = img_size  # 56
        self.swin_img_size = 224  # Standard Swin input size
        self.patch_size = 4
        self.in_chans = 3
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = 7
        self.mlp_ratio = 4.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.1
        
        # Upsample 56 -> 224
        self.upsample_for_swin = nn.Upsample(size=self.swin_img_size, mode='bilinear', align_corners=True)
        
        # Patch Embedding
        patches_resolution = [self.swin_img_size // self.patch_size, self.swin_img_size // self.patch_size]
        self.patches_resolution = patches_resolution
        
        self.patch_embed = PatchEmbed(
            img_size=self.swin_img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm
        )
        
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        # Encoder: all 4 stages; output is last stage
        # Four stages: H -> H/2 -> H/4 -> H/8 -> H/16
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(4):  # Build all 4 stages
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
                downsample=PatchMerging if (i_layer < 3) else None,  # Downsample first 3 stages
                use_checkpoint=False
            )
            self.layers.append(layer)
        
        # Norm after final stage
        self.norm = nn.LayerNorm(int(self.embed_dim * 2 ** 3))  # Stage-4 channel count
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights (from swin.py)."""
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
            x: Input features [B, 3, 56, 56]
        
        Returns:
            Output features [B, 768, 7, 7] (for tiny)
        """
        # Upsample to Swin input: 56 -> 224
        x = self.upsample_for_swin(x)  # [B, 3, 224, 224]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, 56*56, 96] (H/4 * W/4, embed_dim)
        x = self.pos_drop(x)
        
        # Encoder stages
        for layer in self.layers:
            x = layer(x)  # Per stage: [B, L, C]
        
        # Norm
        x = self.norm(x)  # [B, 7*7, 768] (H/16 * W/16, 768)
        
        # Reshape to [B, 768, 7, 7]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 7, 7]
        
        return x


class GGCNNDecoder(nn.Module):
    """
    GGCNN-style decoder with GAAM.
    Three upsampling stages + skip connections (F3, F2, F1) + grasp-aware attention.
    Outputs: Pos (quality/location), Cos (angle cosine), Sin (angle sine), Width.

    GAAM is inserted at key depths to improve grasp prediction quality.
    """
    def __init__(self, swin_channels: int = 768, use_gaam: bool = True, 
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        Initialize decoder.

        Args:
            swin_channels: Swin output channels (tiny/small=768, base=1024)
            use_gaam: Use grasp-aware attention (GAAM)
            use_cf_gaam: Use coarse-to-fine GAAM (CF-GAAM), stronger variant
            num_peaks: Number of peaks for CF-GAAM
        """
        super(GGCNNDecoder, self).__init__()
        self.use_gaam = use_gaam and not use_cf_gaam  # CF-GAAM replaces standalone GAAM
        self.use_cf_gaam = use_cf_gaam
        
        # Reduce Swin channels
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(swin_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM #1: after channel reduction
        if use_cf_gaam:
            self.cf_gaam1 = CoarseToFineGAAM(
                channels=256,
                use_coarse_fine=True,
                num_peaks=num_peaks,
                use_gaam=True,  # CF-GAAM uses base GAAM inside
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        elif use_gaam:
            self.gaam1 = GraspAwareAttentionModule(
                channels=256,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Upsample layer 1: 7x7 -> 14x14
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Upsample layer 2: 14x14 -> 28x28
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Conv after concatenating F3
        # F3: [B, 192, 28, 28], current: [B, 192, 28, 28], shapes align
        self.conv_skip3 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM #2: after F3 skip (28x28)
        if use_cf_gaam:
            self.cf_gaam2 = CoarseToFineGAAM(
                channels=192,
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
                channels=192,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Upsample layer 3: 28x28 -> 56x56
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Conv after concatenating F2
        # F2: [B, 96, 56, 56], current: [B, 96, 56, 56], shapes align
        self.conv_skip2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM #3: after F2 skip (56x56)
        if use_cf_gaam:
            self.cf_gaam3 = CoarseToFineGAAM(
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
            self.gaam3 = GraspAwareAttentionModule(
                channels=96,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # Upsample layer 4: 56x56 -> 112x112
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Conv after concatenating F1
        # F1: [B, 48, 112, 112], current: [B, 48, 112, 112], shapes align
        self.conv_skip1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final feature processing + upsample toward full resolution
        # 112x112 -> 224x224 (x2)
        self.upsample_final1 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.upsample_final2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM #4: before heads (224x224), final refinement
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
        
        # Separate output heads (see swin.py / parallel_model.py)
        # Pos head: quality / location in [0, 1]
        self.pos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Cos head: angle cosine in [-1, 1]
        self.cos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Sin head: angle sine in [-1, 1]
        self.sin_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Width head: width in [0, +inf)
        self.width_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Swin output [B, 768, 7, 7]
            f1: CNN F1 [B, 48, 112, 112]
            f2: CNN F2 [B, 96, 56, 56]
            f3: CNN F3 [B, 192, 28, 28]
        
        Returns:
            Dict with keys 'pos', 'cos', 'sin', 'width'
        """
        # Reduce: 768 -> 256
        x = self.reduce_channels(x)  # [B, 256, 7, 7]
        
        # GAAM/CF-GAAM #1: start of decoder
        if self.use_cf_gaam:
            x = self.cf_gaam1(x)  # [B, 256, 7, 7]
        elif self.use_gaam:
            x = self.gaam1(x)  # [B, 256, 7, 7]
        
        # Upsample: 7x7 -> 14x14
        x = self.upsample1(x)  # [B, 192, 14, 14]
        
        # Upsample: 14x14 -> 28x28
        x = self.upsample2(x)  # [B, 192, 28, 28]
        
        # Concat F3 (resize if needed)
        if x.shape[2:] != f3.shape[2:]:
            f3 = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f3], dim=1)  # [B, 384, 28, 28]
        x = self.conv_skip3(x)  # [B, 192, 28, 28]
        
        # GAAM/CF-GAAM #2: after F3 skip
        if self.use_cf_gaam:
            x = self.cf_gaam2(x)  # [B, 192, 28, 28]
        elif self.use_gaam:
            x = self.gaam2(x)  # [B, 192, 28, 28]
        
        # Upsample: 28x28 -> 56x56
        x = self.upsample3(x)  # [B, 96, 56, 56]
        
        # Concat F2 (resize if needed)
        if x.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f2], dim=1)  # [B, 192, 56, 56]
        x = self.conv_skip2(x)  # [B, 96, 56, 56]
        
        # GAAM/CF-GAAM #3: after F2 skip
        if self.use_cf_gaam:
            x = self.cf_gaam3(x)  # [B, 96, 56, 56]
        elif self.use_gaam:
            x = self.gaam3(x)  # [B, 96, 56, 56]
        
        # Upsample: 56x56 -> 112x112
        x = self.upsample4(x)  # [B, 48, 112, 112]
        
        # Concat F1 (resize if needed)
        if x.shape[2:] != f1.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f1], dim=1)  # [B, 96, 112, 112]
        x = self.conv_skip1(x)  # [B, 48, 112, 112]
        
        # Final feature path
        x = self.upsample_final1(x)  # [B, 32, 112, 112]
        x = self.upsample_final2(x)  # [B, 32, 224, 224]
        
        # GAAM/CF-GAAM #4: before output heads
        if self.use_cf_gaam:
            x = self.cf_gaam4(x)  # [B, 32, 224, 224]
        elif self.use_gaam:
            x = self.gaam4(x)  # [B, 32, 224, 224]
        
        # Output heads
        pos_output = self.pos_output(x)    # [B, 1, 224, 224] quality / location
        cos_output = self.cos_output(x)    # [B, 1, 224, 224] angle cosine
        sin_output = self.sin_output(x)    # [B, 1, 224, 224] angle sine
        width_output = self.width_output(x)  # [B, 1, 224, 224] width
        
        return {"pos": pos_output, "cos": cos_output, "sin": sin_output, "width": width_output}


class HybridGraspNet(nn.Module):
    """
    Hybrid Grasp Network (serial path; Swin may use ImageNet-pretrained weights when enabled).

    Flow:
    Input RGB-D [B, 4, 224, 224]
        |
    CNN backbone
        +-- F1 [B, 48, 112, 112]  (skip 1)
        +-- F2 [B, 96, 56, 56]    (Swin input + skip 2)
        +-- F3 [B, 192, 28, 28]   (skip 3)

    F2 [B, 96, 56, 56]
        | Channel adapter (96 -> 3)
    [B, 3, 56, 56]
        | Swin Transformer (ImageNet pretrained if use_pretrained)
    [B, 768, 7, 7]
        | Decoder
        + skip F3 [192, 28, 28]
        + skip F2 [96, 56, 56]
        + skip F1 [48, 112, 112]
    Output: Pos(1), Cos(1), Sin(1), Width(1) at [B, *, 224, 224]
    """
    
    def __init__(self, in_chans: int = 4, input_channels: int = None, 
                 use_pretrained: bool = True, swin_size: str = 'tiny',
                 use_uncertainty_loss: bool = True, use_gaam: bool = True,
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        Args:
            in_chans: Input channels, default 4 (RGB-D)
            input_channels: Alias for in_chans
            use_pretrained: Use ImageNet-pretrained Swin when supported
            swin_size: Swin variant ('tiny', 'small', 'base')
            use_uncertainty_loss: Use uncertainty-weighted loss (recommended)
            use_gaam: Use grasp-aware attention (GAAM)
            use_cf_gaam: Use coarse-to-fine GAAM (CF-GAAM), enhanced variant
            num_peaks: Peak count for CF-GAAM
        """
        super(HybridGraspNet, self).__init__()
        
        if input_channels is not None:
            in_chans = input_channels
        
        print(f"[Model] Initializing HybridGraspNet (serial)")
        print(f"  - input channels: {in_chans} (RGB-D)")
        print(f"  - Swin size: {swin_size}")
        print(f"  - pretrained: {use_pretrained}")
        print(f"  - uncertainty loss: {'on' if use_uncertainty_loss else 'off'}")
        if use_cf_gaam:
            print(f"  - coarse-to-fine GAAM (CF-GAAM): on [enhanced]")
            print(f"    - num_peaks: {num_peaks}")
        else:
            print(f"  - grasp-aware attention (GAAM): {'on' if use_gaam else 'off'} [core module]")
        
        # CNN backbone: multi-scale features
        self.backbone = CNNBackbone(in_chans=in_chans)
        
        # Channel adapter: 96 -> 3 (F2 -> Swin RGB)
        self.channel_adapter = ChannelAdapter(in_channels=96, out_channels=3)
        
        # Swin Transformer
        self.swin = SwinTransformerBlock(
            img_size=56,  # F2 spatial size
            pretrained=use_pretrained,
            model_size=swin_size
        )
        
        swin_out_ch = self.swin.out_channels
        
        # Decoder with GAAM or CF-GAAM
        self.decoder = GGCNNDecoder(
            swin_channels=swin_out_ch, 
            use_gaam=use_gaam,
            use_cf_gaam=use_cf_gaam,
            num_peaks=num_peaks
        )
        
        self.use_uncertainty_loss = use_uncertainty_loss
        if use_uncertainty_loss:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=4)
            print(f"  - loss auto-weighting: enabled (model learns task weights)")
        
        print(f"[Model] Initialization done.")
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, 4, 224, 224]
            verbose: Print intermediate tensor shapes

        Returns:
            Dict with:
                - 'pos': quality map [B, 1, 224, 224]
                - 'cos': angle cosine map [B, 1, 224, 224]
                - 'sin': angle sine map [B, 1, 224, 224]
                - 'width': width map [B, 1, 224, 224]
        """
        if verbose:
            print(f"input: {x.shape}")
        
        # Step 1: CNN backbone
        f1, f2, f3 = self.backbone(x)
        if verbose:
            print(f"CNN F1: {f1.shape}")  # [B, 48, 112, 112]
            print(f"CNN F2: {f2.shape}")  # [B, 96, 56, 56]
            print(f"CNN F3: {f3.shape}")  # [B, 192, 28, 28]
        
        # Step 2: adapt F2 (96 -> 3)
        f2_adapted = self.channel_adapter(f2)  # [B, 3, 56, 56]
        if verbose:
            print(f"F2 after adapter: {f2_adapted.shape}")
        
        # Step 3: Swin
        swin_out = self.swin(f2_adapted)  # [B, 768, 7, 7]
        if verbose:
            print(f"Swin out: {swin_out.shape}")
        
        # Step 4: decoder with F3, F2, F1 skips
        outputs = self.decoder(swin_out, f1, f2, f3)
        if verbose:
            print(f"outputs:")
            print(f"  Pos (quality): {outputs['pos'].shape}")
            print(f"  Cos (angle cosine): {outputs['cos'].shape}")
            print(f"  Sin (angle sine): {outputs['sin'].shape}")
            print(f"  Width: {outputs['width'].shape}")
        
        return outputs
    
    def compute_loss(self, xc: torch.Tensor, yc: list, 
                     loss_weights: Dict[str, float] = None) -> Dict[str, any]:
        """
        Compute loss (compatible with GGCNN training).

        Args:
            xc: Input image [B, C, H, W]
            yc: Targets [pos_img, cos_img, sin_img, width_img]
            loss_weights: Manual weights {'p': 1.5, 'cos': 1.0, 'sin': 1.0, 'width': 0.8}

        Returns:
            Dict with 'loss', 'losses', 'pred'
        """
        if loss_weights is None:
            loss_weights = {
                'p': 1.5,      # quality (primary)
                'cos': 1.0,
                'sin': 1.0,
                'width': 0.8
            }
        
        outputs = self.forward(xc)
        
        pos_pred = outputs['pos']
        cos_pred = outputs['cos']
        sin_pred = outputs['sin']
        width_pred = outputs['width']
        
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
            # Uncertainty weighting (learned task weights)
            losses_list = [p_loss_raw, cos_loss_raw, sin_loss_raw, width_loss_raw]
            total_loss = self.uncertainty_loss(losses_list)
            
            learned_weights = self.uncertainty_loss.get_weights()
            learned_sigmas = self.uncertainty_loss.get_uncertainties()
            
            p_loss = p_loss_raw * learned_weights[0]
            cos_loss = cos_loss_raw * learned_weights[1]
            sin_loss = sin_loss_raw * learned_weights[2]
            width_loss = width_loss_raw * learned_weights[3]
            
        else:
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
