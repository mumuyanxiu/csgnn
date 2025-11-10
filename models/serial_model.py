"""
混合抓取网络 (Hybrid Grasp Network)
结合 CNN Backbone + Swin Transformer + GGCNN-style Decoder
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 导入 swin.py 中的经典 Swin Transformer 实现
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swin import (
    PatchEmbed, BasicLayer, PatchMerging,
    to_2tuple, trunc_normal_
)


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权多任务损失（与 parallel_model 共享）
    
    论文: Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)
    原理: L = Σ exp(-s_i) * L_i + s_i
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
    基本残差块: Conv3x3 → BN → ReLU → Conv3x3 → BN + Skip
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长
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
            x: 输入特征 [B, C_in, H, W]
        
        Returns:
            输出特征 [B, C_out, H', W']
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
    CNN Backbone: 提取多尺度特征
    输入: [B, 4, H, W] (RGB-D)
    输出: F1 [B, 48, H/2, W/2], F2 [B, 96, H/4, W/4], F3 [B, 192, H/8, W/8]
    """
    def __init__(self, in_chans: int = 4):
        """
        Args:
            in_chans: 输入通道数，默认4 (RGB-D)
        """
        super(CNNBackbone, self).__init__()
        
        # Stage 1: H → H/2, 输出48通道
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_chans, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48, 48, stride=1)
        )
        
        # Stage 2: H/2 → H/4, 输出96通道
        self.stage2 = nn.Sequential(
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 96, stride=1)
        )
        
        # Stage 3: H/4 → H/8, 输出192通道
        self.stage3 = nn.Sequential(
            ResidualBlock(96, 192, stride=2),
            ResidualBlock(192, 192, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: 输入图像 [B, 4, H, W]
        
        Returns:
            (F1, F2, F3): 三个尺度的特征图
        """
        f1 = self.stage1(x)     # [B, 48, H/2, W/2]
        f2 = self.stage2(f1)    # [B, 96, H/4, W/4]
        f3 = self.stage3(f2)    # [B, 192, H/8, W/8]
        
        return f1, f2, f3


class ChannelAdapter(nn.Module):
    """
    通道适配层：将 CNN 特征通道数适配到 Swin 输入要求
    """
    def __init__(self, in_channels: int = 96, out_channels: int = 3):
        """
        Args:
            in_channels: 输入通道数 (来自 CNN)
            out_channels: 输出通道数 (Swin 需要 3 通道 RGB)
        """
        super(ChannelAdapter, self).__init__()
        
        self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, in_channels, H, W]
        
        Returns:
            适配后特征 [B, out_channels, H, W]
        """
        return self.adapt(x)


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer 模块 - 使用经典实现，不使用预训练
    输入: [B, 3, 56, 56] 
    输出: [B, 768, 7, 7] (对于 Swin-Tiny，即第4层输出)
    """
    def __init__(self, img_size: int = 56, pretrained: bool = False, model_size: str = 'tiny'):
        """
        Args:
            img_size: 输入图像尺寸 (56 for F2 feature)
            pretrained: 不使用预训练（保留参数以兼容接口）
            model_size: 模型大小 ('tiny', 'small', 'base')
        """
        super(SwinTransformerBlock, self).__init__()
        
        # 模型配置（基于 swin.py 中的经典配置）
        configs = {
            'tiny': {
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'stage4_channels': 768  # 第4层输出通道数
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
        
        print(f"[Swin] 使用经典 Swin Transformer 实现 (model_size={model_size}, pretrained=False)")
        print(f"  - embed_dim: {config['embed_dim']}")
        print(f"  - depths: {config['depths']}")
        print(f"  - num_heads: {config['num_heads']}")
        print(f"  - 输出通道: {self.out_channels}")
        
        # 输入图像尺寸（需要上采样到 224x224）
        self.input_img_size = img_size  # 56
        self.swin_img_size = 224  # Swin 标准输入尺寸
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
        
        # 上采样层：56 → 224
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
        
        # 构建编码器层（全部4层，取最后一层）
        # 我们需要4层：H → H/2 → H/4 → H/8 → H/16
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(4):  # 构建全部4层
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
                downsample=PatchMerging if (i_layer < 3) else None,  # 前3层需要下采样
                use_checkpoint=False
            )
            self.layers.append(layer)
        
        # 归一化层（最后一层）
        self.norm = nn.LayerNorm(int(self.embed_dim * 2 ** 3))  # 第4层通道数
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重（从 swin.py 复制）"""
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
            x: 输入特征 [B, 3, 56, 56]
        
        Returns:
            输出特征 [B, 768, 7, 7] (for tiny)
        """
        # 上采样到 Swin 所需尺寸: 56 → 224
        x = self.upsample_for_swin(x)  # [B, 3, 224, 224]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, 56*56, 96] (H/4 * W/4, embed_dim)
        x = self.pos_drop(x)
        
        # 通过编码器层（全部4层）
        for layer in self.layers:
            x = layer(x)  # 每层输出: [B, L, C]
        
        # 归一化
        x = self.norm(x)  # [B, 7*7, 768] (H/16 * W/16, 768)
        
        # 转换为 [B, 768, 7, 7]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 7, 7]
        
        return x


class GGCNNDecoder(nn.Module):
    """
    GGCNN 风格的解码器
    三级上采样 + Skip Connections (F3, F2, F1)
    输出: Pos (质量/位置), Cos (角度余弦), Sin (角度正弦), Width (宽度)
    """
    def __init__(self, swin_channels: int = 768):
        """
        初始化解码器
        
        Args:
            swin_channels: Swin 输出通道数 (tiny/small=768, base=1024)
        """
        super(GGCNNDecoder, self).__init__()
        
        # 降维层：Swin 输出 → 较小通道数
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(swin_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 layer 1: 7x7 → 14x14
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 layer 2: 14x14 → 28x28
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # 拼接 F3 后的卷积
        # F3: [B, 192, 28, 28]，当前: [B, 192, 28, 28]，尺寸匹配
        self.conv_skip3 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 layer 3: 28x28 → 56x56
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 拼接 F2 后的卷积
        # F2: [B, 96, 56, 56]，当前: [B, 96, 56, 56]，尺寸匹配
        self.conv_skip2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 上采样 layer 4: 56x56 → 112x112
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 拼接 F1 后的卷积
        # F1: [B, 48, 112, 112]，当前: [B, 48, 112, 112]，尺寸匹配
        self.conv_skip1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 最后的特征处理 + 上采样到接近全分辨率
        # 112x112 → 224x224 (×2)
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
        
        # 独立输出头（每个头都有独立的卷积层，参考 swin.py 和 parallel_model.py）
        # Pos head: 位置/质量评估 [0, 1]
        self.pos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Cos head: 角度余弦 [-1, 1]
        self.cos_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Sin head: 角度正弦 [-1, 1]
        self.sin_output = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        
        # Width head: 宽度 [0, +∞)
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
            x: Swin 输出 [B, 768, 7, 7]
            f1: CNN F1 特征 [B, 48, 112, 112]
            f2: CNN F2 特征 [B, 96, 56, 56]
            f3: CNN F3 特征 [B, 192, 28, 28]
        
        Returns:
            字典包含 'pos', 'cos', 'sin', 'width'
        """
        # 降维: 768 → 256
        x = self.reduce_channels(x)  # [B, 256, 7, 7]
        
        # 上采样: 7x7 → 14x14
        x = self.upsample1(x)  # [B, 192, 14, 14]
        
        # 上采样: 14x14 → 28x28
        x = self.upsample2(x)  # [B, 192, 28, 28]
        
        # 拼接 F3 (确保尺寸匹配)
        if x.shape[2:] != f3.shape[2:]:
            f3 = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f3], dim=1)  # [B, 384, 28, 28]
        x = self.conv_skip3(x)  # [B, 192, 28, 28]
        
        # 上采样: 28x28 → 56x56
        x = self.upsample3(x)  # [B, 96, 56, 56]
        
        # 拼接 F2 (确保尺寸匹配)
        if x.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f2], dim=1)  # [B, 192, 56, 56]
        x = self.conv_skip2(x)  # [B, 96, 56, 56]
        
        # 上采样: 56x56 → 112x112
        x = self.upsample4(x)  # [B, 48, 112, 112]
        
        # 拼接 F1 (确保尺寸匹配)
        if x.shape[2:] != f1.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f1], dim=1)  # [B, 96, 112, 112]
        x = self.conv_skip1(x)  # [B, 48, 112, 112]
        
        # 最后的特征处理
        x = self.upsample_final1(x)  # [B, 32, 112, 112]
        x = self.upsample_final2(x)  # [B, 32, 224, 224]
        
        # 独立输出头（每个头独立处理共享特征）
        pos_output = self.pos_output(x)    # [B, 1, 224, 224] (质量/位置)
        cos_output = self.cos_output(x)    # [B, 1, 224, 224] (角度余弦)
        sin_output = self.sin_output(x)    # [B, 1, 224, 224] (角度正弦)
        width_output = self.width_output(x)  # [B, 1, 224, 224] (宽度)
        
        return {"pos": pos_output, "cos": cos_output, "sin": sin_output, "width": width_output}


class HybridGraspNet(nn.Module):
    """
    混合抓取网络 (Hybrid Grasp Network with Pretrained Swin Transformer)
    
    结构:
    Input RGB-D [B, 4, 224, 224]
        ↓
    CNN Backbone
        ├─→ F1 [B, 48, 112, 112]  (skip connection 1)
        ├─→ F2 [B, 96, 56, 56]    (输入到 Swin + skip 2)
        └─→ F3 [B, 192, 28, 28]   (skip connection 3)
        
    F2 [B, 96, 56, 56]
        ↓ Channel Adapter (96 → 3)
    [B, 3, 56, 56]
        ↓ Swin Transformer (ImageNet Pretrained)
    [B, 768, 7, 7]
        ↓ Decoder
        ├─ Skip from F3 [192, 28, 28]
        ├─ Skip from F2 [96, 56, 56]
        └─ Skip from F1 [48, 112, 112]
    Output: Pos(1), Cos(1), Sin(1), Width(1) [B, *, 224, 224]
    """
    
    def __init__(self, in_chans: int = 4, input_channels: int = None, 
                 use_pretrained: bool = True, swin_size: str = 'tiny',
                 use_uncertainty_loss: bool = True):
        """
        Args:
            in_chans: 输入通道数，默认4 (RGB-D)
            input_channels: 兼容参数（与 in_chans 相同）
            use_pretrained: 是否使用 ImageNet 预训练的 Swin
            swin_size: Swin 模型大小 ('tiny', 'small', 'base')
            use_uncertainty_loss: 是否使用不确定性加权损失（推荐开启）
        """
        super(HybridGraspNet, self).__init__()
        
        # 兼容性处理
        if input_channels is not None:
            in_chans = input_channels
        
        print(f"[Model] 初始化 HybridGraspNet (串行版本)")
        print(f"  - 输入通道: {in_chans} (RGB-D)")
        print(f"  - Swin 模型: {swin_size}")
        print(f"  - 使用预训练: {use_pretrained}")
        print(f"  - 不确定性损失: {'开启' if use_uncertainty_loss else '关闭'}")
        
        # CNN Backbone：提取多尺度特征
        self.backbone = CNNBackbone(in_chans=in_chans)
        
        # 通道适配：96 → 3 (F2 通道数 → Swin 输入通道数)
        self.channel_adapter = ChannelAdapter(in_channels=96, out_channels=3)
        
        # Swin Transformer (预训练)
        self.swin = SwinTransformerBlock(
            img_size=56,  # F2 的空间尺寸
            pretrained=use_pretrained,
            model_size=swin_size
        )
        
        # Swin 输出通道数
        swin_out_ch = self.swin.out_channels
        
        # Decoder
        self.decoder = GGCNNDecoder(swin_channels=swin_out_ch)
        
        # 不确定性加权损失模块
        self.use_uncertainty_loss = use_uncertainty_loss
        if use_uncertainty_loss:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=4)
            print(f"  - 损失自动加权: 启用（模型将学习最优权重）")
        
        print(f"[Model] 初始化完成！")
    
    def forward(self, x: torch.Tensor, verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 4, 224, 224]
            verbose: 是否打印中间张量尺寸
        
        Returns:
            字典包含:
                - 'pos': 位置/质量图 [B, 1, 224, 224]
                - 'cos': 角度余弦图 [B, 1, 224, 224]
                - 'sin': 角度正弦图 [B, 1, 224, 224]
                - 'width': 宽度图 [B, 1, 224, 224]
        """
        if verbose:
            print(f"输入: {x.shape}")
        
        # Step 1: CNN Backbone 提取多尺度特征
        f1, f2, f3 = self.backbone(x)
        if verbose:
            print(f"CNN F1: {f1.shape}")  # [B, 48, 112, 112]
            print(f"CNN F2: {f2.shape}")  # [B, 96, 56, 56]
            print(f"CNN F3: {f3.shape}")  # [B, 192, 28, 28] (不使用)
        
        # Step 2: F2 通道适配 (96 → 3)
        f2_adapted = self.channel_adapter(f2)  # [B, 3, 56, 56]
        if verbose:
            print(f"F2 适配后: {f2_adapted.shape}")
        
        # Step 3: Swin Transformer (预训练)
        swin_out = self.swin(f2_adapted)  # [B, 768, 7, 7]
        if verbose:
            print(f"Swin 输出: {swin_out.shape}")
        
        # Step 4: Decoder (with F3, F2, F1 skip connections)
        outputs = self.decoder(swin_out, f1, f2, f3)
        if verbose:
            print(f"最终输出:")
            print(f"  Pos (质量): {outputs['pos'].shape}")
            print(f"  Cos (角度余弦): {outputs['cos'].shape}")
            print(f"  Sin (角度正弦): {outputs['sin'].shape}")
            print(f"  Width (宽度): {outputs['width'].shape}")
        
        return outputs
    
    def compute_loss(self, xc: torch.Tensor, yc: list, 
                     loss_weights: Dict[str, float] = None) -> Dict[str, any]:
        """
        计算损失函数（兼容 GGCNN 训练框架）
        
        Args:
            xc: 输入图像 [B, C, H, W]
            yc: 标签列表 [pos_img, cos_img, sin_img, width_img]
            loss_weights: 损失权重字典 {'p': 1.5, 'cos': 1.0, 'sin': 1.0, 'width': 0.8}
        
        Returns:
            字典包含:
                - 'loss': 总损失
                - 'losses': 各分项损失字典
                - 'pred': 预测输出字典
        """
        # 默认权重：质量最重要 > 角度 > 宽度
        if loss_weights is None:
            loss_weights = {
                'p': 1.5,      # 质量损失权重（最关键）
                'cos': 1.0,    # cos 角度损失
                'sin': 1.0,    # sin 角度损失
                'width': 0.8   # 宽度损失（相对次要）
            }
        
        # 前向传播
        outputs = self.forward(xc)
        
        # 提取预测（已经是独立输出头）
        pos_pred = outputs['pos']       # [B, 1, H', W']
        cos_pred = outputs['cos']       # [B, 1, H', W']
        sin_pred = outputs['sin']       # [B, 1, H', W']
        width_pred = outputs['width']   # [B, 1, H', W']
        
        # 标签
        pos_gt, cos_gt, sin_gt, width_gt = yc
        
        # 确保尺寸匹配：将预测上采样到标签尺寸（保持标签的稀疏性）
        if pos_pred.shape[2:] != pos_gt.shape[2:]:
            pos_pred = F.interpolate(pos_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            sin_pred = F.interpolate(sin_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            cos_pred = F.interpolate(cos_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
            width_pred = F.interpolate(width_pred, size=pos_gt.shape[2:], mode='bilinear', align_corners=True)
        
        # 计算各项原始损失（不加权）
        p_loss_raw = F.mse_loss(pos_pred, pos_gt)
        cos_loss_raw = F.mse_loss(cos_pred, cos_gt)
        sin_loss_raw = F.mse_loss(sin_pred, sin_gt)
        width_loss_raw = F.mse_loss(width_pred, width_gt)
        
        # 根据模式计算总损失
        if self.use_uncertainty_loss:
            # 🔥 不确定性加权（自动学习权重）
            losses_list = [p_loss_raw, cos_loss_raw, sin_loss_raw, width_loss_raw]
            total_loss = self.uncertainty_loss(losses_list)
            
            # 获取学到的权重
            learned_weights = self.uncertainty_loss.get_weights()
            learned_sigmas = self.uncertainty_loss.get_uncertainties()
            
            p_loss = p_loss_raw * learned_weights[0]
            cos_loss = cos_loss_raw * learned_weights[1]
            sin_loss = sin_loss_raw * learned_weights[2]
            width_loss = width_loss_raw * learned_weights[3]
            
        else:
            # 手动加权（传统方法）
            p_loss = p_loss_raw * loss_weights['p']
            cos_loss = cos_loss_raw * loss_weights['cos']
            sin_loss = sin_loss_raw * loss_weights['sin']
            width_loss = width_loss_raw * loss_weights['width']
            total_loss = p_loss + cos_loss + sin_loss + width_loss
        
        # 构建返回字典
        loss_dict = {
            'p_loss': p_loss,
            'cos_loss': cos_loss,
            'sin_loss': sin_loss,
            'width_loss': width_loss
        }
        
        # 如果使用不确定性损失，添加学到的权重
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
        
        # 返回结果（兼容训练框架）
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
