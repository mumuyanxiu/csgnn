"""
并行混合抓取网络 (Parallel Hybrid Grasp Network)
CNN Backbone 和 Swin Transformer 并行处理，然后融合
集成粗到精抓取感知注意力模块 (CF-GAAM)
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

# 导入抓取感知注意力模块
from .grasp_aware_attention import GraspAwareAttentionModule
from .coarse_to_fine_gaam import CoarseToFineGAAM


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权多任务损失
    
    论文: Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)
    原理: L = Σ exp(-s_i) * L_i + s_i
    其中 s_i = log(σ_i²) 是可学习的对数方差
    
    优势:
    - 自动学习任务权重，无需手动调参
    - 基于贝叶斯最大似然估计，有理论保证
    - 根据任务不确定性动态调整权重
    """
    def __init__(self, num_tasks: int = 4):
        """
        Args:
            num_tasks: 任务数量（质量、cos、sin、宽度 = 4）
        """
        super(UncertaintyWeightedLoss, self).__init__()
        
        # 可学习的对数方差参数（初始化为0，即初始权重=1）
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: 损失列表 [p_loss, cos_loss, sin_loss, width_loss]
        
        Returns:
            total_loss: 自动加权后的总损失
        """
        total_loss = 0
        
        for i, loss in enumerate(losses):
            # 计算精度（权重）: precision = exp(-log_var) = 1/σ²
            precision = torch.exp(-self.log_vars[i])
            
            # 加权损失 + 正则项（防止方差无限增大）
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss
    
    def get_weights(self) -> torch.Tensor:
        """
        获取当前学到的权重（用于可视化和分析）
        
        Returns:
            weights: [w1, w2, w3, w4]，即 [1/σ1², 1/σ2², 1/σ3², 1/σ4²]
        """
        return torch.exp(-self.log_vars).detach()
    
    def get_uncertainties(self) -> torch.Tensor:
        """
        获取当前学到的不确定性（标准差）
        
        Returns:
            sigmas: [σ1, σ2, σ3, σ4]
        """
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


# 删除 PatchifyForSwin，直接让 Swin 接受 4 通道输入
# Swin 的 PatchEmbed 本身支持任意通道数，不需要降维


class SwinTransformerBranch(nn.Module):
    """
    Swin Transformer 分支 (并行路径) - 使用经典实现，不使用预训练
    输入: [B, 4, H, W] (RGB-D，H, W 可变，例如 224 或 300)
    输出: [B, 192, H/16, W/16]
    
    注意：直接接受 4 通道输入，不会丢失深度信息
    使用全部 4 层编码器（与 serial_model 一致）
    """
    def __init__(self, in_chans: int = 4, pretrained: bool = False, model_size: str = 'tiny'):
        """
        Args:
            in_chans: 输入通道数，默认 4 (RGB-D)
            pretrained: 不使用预训练（保留参数以兼容接口）
            model_size: 模型大小 ('tiny', 'small', 'base')
        """
        super(SwinTransformerBranch, self).__init__()
        
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
        
        print(f"[Swin分支] 使用经典 Swin Transformer 实现 (model_size={model_size}, pretrained=False)")
        print(f"  - 输入通道: {in_chans} (直接接受 RGB-D，保留深度信息)")
        print(f"  - embed_dim: {config['embed_dim']}")
        print(f"  - depths: {config['depths']}")
        print(f"  - num_heads: {config['num_heads']}")
        
        # 输入图像尺寸（固定为 224x224，便于处理）
        self.img_size = 224
        self.patch_size = 4
        self.in_chans = in_chans  # 支持任意通道数（如 4 通道 RGB-D）
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = 7
        self.mlp_ratio = 4.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.num_layers = len(self.depths)
        
        # 自适应调整到 224x224
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
        
        # 构建编码器层（全部4层，对应 H/16）
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
        
        # 第4层输出通道数
        # 第0层: 96 → 192 (有downsample)
        # 第1层: 192 → 384 (有downsample)  
        # 第2层: 384 → 768 (有downsample)
        # 第3层: 768 → 768 (无downsample，输出768通道)
        stage4_channels = int(self.embed_dim * 2 ** 3)  # 96 * 8 = 768 (tiny)
        
        # 归一化层（最后一层）
        self.norm = nn.LayerNorm(stage4_channels)
        
        # 通道适配：Swin输出 → 192 (与 F3 匹配)
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(stage4_channels, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
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
            x: 输入特征 [B, 4, H, W] (RGB-D，H, W 可以是任意尺寸)
        
        Returns:
            输出特征 [B, 192, H/16, W/16]
        """
        # 调整到 224x224
        x = self.resize_to_224(x)  # [B, 4, 224, 224]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, 56*56, 96] (H/4 * W/4, embed_dim)
        x = self.pos_drop(x)
        
        # 通过编码器层（全部4层）
        for layer in self.layers:
            x = layer(x)  # 每层输出: [B, L, C]，L 和 C 会逐渐减小/增大
        
        # 归一化
        x = self.norm(x)  # [B, 7*7, 768] (H/16 * W/16, 768)
        
        # 第4层输出: [B, 7*7, 768] (H/16 * W/16, 768)
        # 转换为 [B, 768, 7, 7]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 7, 7]
        
        # 通道适配
        x = self.channel_adapt(x)  # [B, 192, 7, 7]
        
        return x


class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块：让两个特征流互相查询
    CNN 特征作为 Query 查询 Swin，Swin 特征作为 Query 查询 CNN
    """
    def __init__(self, channels: int = 192, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            channels: 输入输出通道数
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
        super(CrossAttentionModule, self).__init__()
        
        assert channels % num_heads == 0, f"channels {channels} 必须能被 num_heads {num_heads} 整除"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # CNN → Swin 交叉注意力
        self.q_cnn = nn.Conv2d(channels, channels, 1)
        self.k_swin = nn.Conv2d(channels, channels, 1)
        self.v_swin = nn.Conv2d(channels, channels, 1)
        
        # Swin → CNN 交叉注意力
        self.q_swin = nn.Conv2d(channels, channels, 1)
        self.k_cnn = nn.Conv2d(channels, channels, 1)
        self.v_cnn = nn.Conv2d(channels, channels, 1)
        
        # 输出投影
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
        计算注意力
        
        Args:
            q, k, v: [B, C, H, W]
        
        Returns:
            注意力输出 [B, C, H, W]
        """
        B, C, H, W = q.shape
        
        # Reshape: [B, C, H, W] → [B, num_heads, H*W, head_dim]
        q = q.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.flatten(2).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # 计算注意力分数: [B, num_heads, H*W, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 加权求和: [B, num_heads, H*W, head_dim]
        out = attn @ v
        
        # Reshape 回去: [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        return out
    
    def forward(self, f_cnn: torch.Tensor, f_swin: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            f_cnn: CNN 特征 [B, C, H, W]
            f_swin: Swin 特征 [B, C, H, W]
        
        Returns:
            (增强的 CNN 特征, 增强的 Swin 特征)
        """
        B, C, H, W = f_cnn.shape
        
        # 1. CNN 查询 Swin（CNN 想知道：Swin 看到了什么全局信息）
        q_c = self.q_cnn(f_cnn)
        k_s = self.k_swin(f_swin)
        v_s = self.v_swin(f_swin)
        cnn_enhanced = self._attention(q_c, k_s, v_s)
        cnn_enhanced = self.proj_cnn(cnn_enhanced)
        
        # 残差连接 + LayerNorm
        f_cnn_out = f_cnn + cnn_enhanced
        f_cnn_out = self.norm_cnn(f_cnn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 2. Swin 查询 CNN（Swin 想知道：CNN 提取了什么局部细节）
        q_s = self.q_swin(f_swin)
        k_c = self.k_cnn(f_cnn)
        v_c = self.v_cnn(f_cnn)
        swin_enhanced = self._attention(q_s, k_c, v_c)
        swin_enhanced = self.proj_swin(swin_enhanced)
        
        # 残差连接 + LayerNorm
        f_swin_out = f_swin + swin_enhanced
        f_swin_out = self.norm_swin(f_swin_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return f_cnn_out, f_swin_out


class FusionBlock(nn.Module):
    """
    融合块：融合 CNN 和 Swin 特征
    支持三种模式：
    1. Simple: 简单拼接 + Conv
    2. CrossAttention: 交叉注意力融合（互相查询）
    3. Attention: 单向注意力融合（仅 CNN 查询 Swin）
    """
    def __init__(self, cnn_channels: int = 192, swin_channels: int = 192, 
                 out_channels: int = 192, fusion_type: str = 'simple',
                 num_heads: int = 8):
        """
        Args:
            cnn_channels: CNN 特征通道数 (F3)
            swin_channels: Swin 特征通道数
            out_channels: 输出通道数
            fusion_type: 融合类型 ('simple', 'cross_attention', 'attention')
            num_heads: 交叉注意力头数（仅 cross_attention 模式）
        """
        super(FusionBlock, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'simple':
            # 简单融合：拼接 + Conv
            self.fusion = nn.Sequential(
                nn.Conv2d(cnn_channels + swin_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif fusion_type == 'cross_attention':
            # Cross-Attention 融合（双向查询）
            print(f"[FusionBlock] 使用 Cross-Attention 模式 (num_heads={num_heads})")
            self.cross_attention = CrossAttentionModule(
                channels=cnn_channels,  # 假设 cnn_channels == swin_channels
                num_heads=num_heads
            )
            # 最后的融合层
            self.fusion = nn.Sequential(
                nn.Conv2d(cnn_channels + swin_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif fusion_type == 'attention':
            # 单向注意力（仅 CNN 查询 Swin，轻量级）
            print(f"[FusionBlock] 使用单向 Attention 模式")
            self.q_proj = nn.Conv2d(cnn_channels, cnn_channels, 1)
            self.k_proj = nn.Conv2d(swin_channels, cnn_channels, 1)
            self.v_proj = nn.Conv2d(swin_channels, cnn_channels, 1)
            self.fusion = nn.Sequential(
                nn.Conv2d(cnn_channels + swin_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, f_cnn: torch.Tensor, f_swin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_cnn: CNN 特征 [B, 192, H/8, W/8]
            f_swin: Swin 特征 [B, 192, H/16, W/16] (会自动上采样到 H/8)
        
        Returns:
            融合特征 [B, 192, H/8, W/8]
        """
        # 确保空间尺寸匹配
        if f_cnn.shape[2:] != f_swin.shape[2:]:
            f_swin = F.interpolate(f_swin, size=f_cnn.shape[2:], 
                                   mode='bilinear', align_corners=True)
        
        if self.fusion_type == 'cross_attention':
            # 交叉注意力增强
            f_cnn_enh, f_swin_enh = self.cross_attention(f_cnn, f_swin)
            # 拼接增强后的特征
            x = torch.cat([f_cnn_enh, f_swin_enh], dim=1)  # [B, 384, H/8, W/8]
            x = self.fusion(x)  # [B, 192, H/8, W/8]
        
        elif self.fusion_type == 'attention':
            # 单向注意力（CNN 查询 Swin）
            B, C, H, W = f_cnn.shape
            q = self.q_proj(f_cnn).flatten(2)  # [B, C, H*W]
            k = self.k_proj(f_swin).flatten(2)  # [B, C, H*W]
            v = self.v_proj(f_swin).flatten(2)  # [B, C, H*W]
            
            # 注意力分数
            attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)  # [B, H*W, H*W]
            attn = F.softmax(attn, dim=-1)
            
            # 加权求和
            f_cnn_enh = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
            
            # 拼接
            x = torch.cat([f_cnn_enh, f_swin], dim=1)
            x = self.fusion(x)
        
        else:
            # Simple 模式
            x = torch.cat([f_cnn, f_swin], dim=1)  # [B, 384, H/8, W/8]
            x = self.fusion(x)  # [B, 192, H/8, W/8]
        
        return x


class GGCNNDecoder(nn.Module):
    """
    GGCNN 风格的解码器（集成 GAAM/CF-GAAM 模块）
    两级上采样 + Skip Connections (F2, F1) + 抓取感知注意力
    输出: Q (质量), A (角度: sin, cos), W (宽度)
    
    创新点：在关键位置插入 GAAM/CF-GAAM 模块，提升抓取预测质量
    """
    def __init__(self, fusion_channels: int = 192, use_gaam: bool = False,
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        初始化解码器
        
        Args:
            fusion_channels: 融合特征通道数
            use_gaam: 是否使用抓取感知注意力模块（GAAM）
            use_cf_gaam: 是否使用粗到精GAAM模块（CF-GAAM）- 增强版
            num_peaks: CF-GAAM中检测的峰值数量
        """
        super(GGCNNDecoder, self).__init__()
        self.use_gaam = use_gaam and not use_cf_gaam  # 如果使用CF-GAAM，则不使用原始GAAM
        self.use_cf_gaam = use_cf_gaam
        
        # GAAM/CF-GAAM 模块1：在解码器开始处（融合特征后）
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
        
        # 上采样 layer 1: H/8 → H/4
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(fusion_channels, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 拼接 F2 后的卷积
        self.conv_skip2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 模块2：在 F2 skip connection 之后（H/4 尺度）
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
        
        # 上采样 layer 2: H/4 → H/2
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 拼接 F1 后的卷积
        self.conv_skip1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 模块3：在 F1 skip connection 之后（H/2 尺度）
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
        
        # 最后的特征处理 + 上采样到接近全分辨率
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 上采样 ×2 以接近输入分辨率
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # GAAM/CF-GAAM 模块4：在最终输出之前（全分辨率，用于最终精化）
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
        
        # 独立输出头（每个头都有独立的卷积层，参考 swin.py）
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
    
    def forward(self, x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 融合特征 [B, 192, H/8, W/8]
            f1: CNN F1 特征 [B, 48, H/2, W/2]
            f2: CNN F2 特征 [B, 96, H/4, W/4]
        
        Returns:
            字典包含 'pos', 'cos', 'sin', 'width'
        """
        # GAAM/CF-GAAM 模块1：在解码器开始处
        if self.use_cf_gaam:
            x = self.cf_gaam1(x)  # [B, 192, H/8, W/8]
        elif self.use_gaam:
            x = self.gaam1(x)  # [B, 192, H/8, W/8]
        
        # 上采样: H/8 → H/4
        x = self.upsample1(x)  # [B, 96, H/4, W/4]
        
        # 拼接 F2 (确保尺寸匹配)
        if x.shape[2:] != f2.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f2], dim=1)  # [B, 192, H/4, W/4]
        x = self.conv_skip2(x)  # [B, 96, H/4, W/4]
        
        # GAAM/CF-GAAM 模块2：在 F2 skip connection 之后
        if self.use_cf_gaam:
            x = self.cf_gaam2(x)  # [B, 96, H/4, W/4]
        elif self.use_gaam:
            x = self.gaam2(x)  # [B, 96, H/4, W/4]
        
        # 上采样: H/4 → H/2
        x = self.upsample2(x)  # [B, 48, H/2, W/2]
        
        # 拼接 F1 (确保尺寸匹配)
        if x.shape[2:] != f1.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, f1], dim=1)  # [B, 96, H/2, W/2]
        x = self.conv_skip1(x)  # [B, 48, H/2, W/2]
        
        # GAAM/CF-GAAM 模块3：在 F1 skip connection 之后
        if self.use_cf_gaam:
            x = self.cf_gaam3(x)  # [B, 48, H/2, W/2]
        elif self.use_gaam:
            x = self.gaam3(x)  # [B, 48, H/2, W/2]
        
        # 最后的特征处理
        x = self.final_conv(x)  # [B, 32, H, W] (全分辨率)
        
        # GAAM/CF-GAAM 模块4：在最终输出之前（最终精化）
        if self.use_cf_gaam:
            x = self.cf_gaam4(x)  # [B, 32, H, W]
        elif self.use_gaam:
            x = self.gaam4(x)  # [B, 32, H, W]
        
        # 独立输出头（每个头独立处理共享特征）
        pos_output = self.pos_output(x)   # [B, 1, H, W] (质量/位置)
        cos_output = self.cos_output(x)   # [B, 1, H, W] (角度余弦)
        sin_output = self.sin_output(x)   # [B, 1, H, W] (角度正弦)
        width_output = self.width_output(x)  # [B, 1, H, W] (宽度)
        
        return {"pos": pos_output, "cos": cos_output, "sin": sin_output, "width": width_output}


class ParallelHybridGraspNet(nn.Module):
    """
    并行混合抓取网络 (Parallel Hybrid Grasp Network)
    
    结构:
    Input RGB-D [B, 4, H, W] (支持任意尺寸，例如 224x224 或 300x300)
        ├── CNN Backbone
        │   ├─→ F1 [B, 48, H/2, W/2]  (skip)
        │   ├─→ F2 [B, 96, H/4, W/4]  (skip)
        │   └─→ F3 [B, 192, H/8, W/8] (融合)
        │
        └── Swin Transformer (4层编码器)
                                    ↓
                              S_out [B, 192, H/16, W/16] (融合，会上采样到 H/8)
                       
    FusionBlock(F3, S_out)
        ↓ [B, 192, H/8, W/8]
    Decoder
        ├─ Skip from F2 [96, H/4, W/4]
        └─ Skip from F1 [48, H/2, W/2]
        ↓
    Output: Pos(1), Cos(1), Sin(1), Width(1) [B, *, H, W]
    """
    
    def __init__(self, in_chans: int = 4, input_channels: int = None,
                 use_pretrained: bool = True, swin_size: str = 'tiny',
                 fusion_type: str = 'simple', num_heads: int = 8,
                 use_uncertainty_loss: bool = True, use_gaam: bool = False,
                 use_cf_gaam: bool = False, num_peaks: int = 5):
        """
        Args:
            in_chans: 输入通道数，默认4 (RGB-D)
            input_channels: 兼容参数（与 in_chans 相同）
            use_pretrained: 是否使用 ImageNet 预训练的 Swin
            swin_size: Swin 模型大小 ('tiny', 'small', 'base')
            fusion_type: 融合类型 ('simple', 'cross_attention', 'attention')
            num_heads: 交叉注意力头数（仅 cross_attention 模式有效）
            use_uncertainty_loss: 是否使用不确定性加权损失（推荐开启）
            use_gaam: 是否使用抓取感知注意力模块（GAAM）- 核心创新模块
            use_cf_gaam: 是否使用粗到精GAAM模块（CF-GAAM）- 增强版，包含粗到精预测框架
            num_peaks: CF-GAAM中检测的峰值数量
        """
        super(ParallelHybridGraspNet, self).__init__()
        
        # 兼容性处理
        if input_channels is not None:
            in_chans = input_channels
        
        print(f"[Model] 初始化 ParallelHybridGraspNet (并行版本)")
        print(f"  - 输入通道: {in_chans} (RGB-D)")
        print(f"  - Swin 模型: {swin_size}")
        print(f"  - 使用预训练: {use_pretrained}")
        print(f"  - 融合方式: {fusion_type}")
        if fusion_type == 'cross_attention':
            print(f"  - 注意力头数: {num_heads}")
        print(f"  - 不确定性损失: {'开启' if use_uncertainty_loss else '关闭'}")
        if use_cf_gaam:
            print(f"  - 粗到精抓取感知注意力 (CF-GAAM): 开启 ⭐⭐ 增强版创新模块")
            print(f"    - 峰值数量: {num_peaks}")
        else:
            print(f"  - 抓取感知注意力 (GAAM): {'开启' if use_gaam else '关闭'} ⭐ 核心创新模块")
        
        # CNN Backbone 分支
        self.cnn_backbone = CNNBackbone(in_chans=in_chans)
        
        # Swin Transformer 分支（直接接受 4 通道输入，不丢失深度信息）
        self.swin_branch = SwinTransformerBranch(
            in_chans=in_chans,  # 直接传入 4 通道
            pretrained=use_pretrained,
            model_size=swin_size
        )
        
        # 融合模块
        self.fusion = FusionBlock(
            cnn_channels=192,
            swin_channels=192,
            out_channels=192,
            fusion_type=fusion_type,
            num_heads=num_heads
        )
        
        # 解码器（集成 GAAM 或 CF-GAAM）
        self.decoder = GGCNNDecoder(
            fusion_channels=192,
            use_gaam=use_gaam,
            use_cf_gaam=use_cf_gaam,
            num_peaks=num_peaks
        )
        
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
            x: 输入图像 [B, 4, H, W] (支持任意尺寸，例如 224x224 或 300x300)
            verbose: 是否打印中间张量尺寸
        
        Returns:
            字典包含:
                - 'pos': 位置/质量图 [B, 1, H, W]
                - 'cos': 角度余弦图 [B, 1, H, W]
                - 'sin': 角度正弦图 [B, 1, H, W]
                - 'width': 宽度图 [B, 1, H, W]
        """
        if verbose:
            print(f"输入: {x.shape}")
        
        # ============ 并行分支 ============
        
        # 分支1: CNN Backbone
        f1, f2, f3 = self.cnn_backbone(x)
        if verbose:
            print(f"[CNN分支] F1: {f1.shape}, F2: {f2.shape}, F3: {f3.shape}")
        
        # 分支2: Swin Transformer（直接接受 4 通道 RGB-D）
        s_out = self.swin_branch(x)  # [B, 192, H/16, W/16] (7x7 for 224x224 input)
        if verbose:
            print(f"[Swin分支] 输出: {s_out.shape}")
        
        # ============ 融合 ============
        # 注意：s_out 是 H/16，f3 是 H/8，FusionBlock 会自动处理尺寸匹配
        fused = self.fusion(f3, s_out)  # [B, 192, H/8, W/8]
        if verbose:
            print(f"[融合] 输出: {fused.shape}")
        
        # ============ 解码器 ============
        outputs = self.decoder(fused, f1, f2)
        if verbose:
            print(f"[输出] Pos: {outputs['pos'].shape}, Cos: {outputs['cos'].shape}, Sin: {outputs['sin'].shape}, Width: {outputs['width'].shape}")
        
        return outputs
    
    def compute_loss(self, xc: torch.Tensor, yc: list, 
                     loss_weights: Dict[str, float] = None) -> Dict[str, any]:
        """
        计算损失函数（兼容 GGCNN 训练框架）
        
        支持两种模式:
        1. 不确定性加权（use_uncertainty_loss=True）: 模型自动学习最优权重
        2. 手动加权（use_uncertainty_loss=False）: 使用 loss_weights 参数
        
        Args:
            xc: 输入图像 [B, C, H, W]
            yc: 标签列表 [pos_img, cos_img, sin_img, width_img]
            loss_weights: 手动权重字典（仅当 use_uncertainty_loss=False 时使用）
        
        Returns:
            字典包含:
                - 'loss': 总损失
                - 'losses': 各分项损失字典（包含学到的权重信息）
                - 'pred': 预测输出字典
        """
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
            
            # 获取学到的权重（用于可视化）
            learned_weights = self.uncertainty_loss.get_weights()
            learned_sigmas = self.uncertainty_loss.get_uncertainties()
            
            # 用于返回的加权损失（便于监控各项损失）
            p_loss = p_loss_raw * learned_weights[0]
            cos_loss = cos_loss_raw * learned_weights[1]
            sin_loss = sin_loss_raw * learned_weights[2]
            width_loss = width_loss_raw * learned_weights[3]
            
        else:
            # 手动加权（传统方法）
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
        
        # 构建返回字典
        loss_dict = {
            'p_loss': p_loss,
            'cos_loss': cos_loss,
            'sin_loss': sin_loss,
            'width_loss': width_loss
        }
        
        # 如果使用不确定性损失，添加学到的权重和不确定性
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
