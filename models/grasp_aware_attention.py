"""
抓取感知注意力模块 (Grasp-Aware Attention Module, GAAM)
核心创新模块：专门为抓取任务设计的注意力机制

设计理念：
1. 边缘感知注意力：抓取点通常在物体边缘，需要强化边缘特征
2. 中心稳定性注意力：抓取需要物体中心支撑，关注稳定性
3. 宽度自适应注意力：根据物体尺寸动态调整注意力范围
4. 角度一致性注意力：抓取角度应该垂直于边缘方向
5. 多尺度抓取融合：不同尺度特征关注不同抓取属性

论文创新点：
- 首次将抓取任务的物理约束（边缘、中心、宽度、角度）显式建模为注意力机制
- 多维度注意力协同工作，提升抓取预测的准确性和鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EdgeAwareAttention(nn.Module):
    """
    边缘感知注意力模块
    
    创新点：抓取点通常位于物体边缘，此模块专门强化边缘特征
    原理：使用 Sobel 算子检测边缘，然后对边缘区域赋予更高注意力权重
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Args:
            channels: 输入特征通道数
            kernel_size: 边缘检测核大小
        """
        super(EdgeAwareAttention, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 边缘检测卷积核（Sobel算子）
        # 水平边缘检测
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 垂直边缘检测
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(channels, 1, 1, 1))
        
        # 边缘特征增强网络
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()  # 生成注意力权重 [0, 1]
        )
        
        # 边缘特征提取
        self.edge_extract = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            边缘增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 边缘检测：计算每个通道的边缘强度
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=C)  # [B, C, H, W]
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=C)  # [B, C, H, W]
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # [B, C, H, W]
        
        # 2. 通道级边缘强度（平均所有通道）
        edge_map = edge_magnitude.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 3. 生成边缘注意力权重
        edge_attention = self.edge_enhance(edge_map)  # [B, C, H, W]
        
        # 4. 提取边缘特征
        edge_features = self.edge_extract(x)
        
        # 5. 应用边缘注意力：边缘区域得到强化
        enhanced = x + edge_attention * edge_features
        
        return enhanced


class CenterStabilityAttention(nn.Module):
    """
    中心稳定性注意力模块
    
    创新点：抓取需要物体中心支撑，此模块关注物体的重心和稳定性区域
    原理：计算特征的空间中心，对中心区域赋予更高权重
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: 输入特征通道数
        """
        super(CenterStabilityAttention, self).__init__()
        self.channels = channels
        
        # 中心特征提取
        self.center_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化得到中心特征
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间中心注意力生成
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3),  # 输入是坐标图
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            中心稳定性增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 通道级中心注意力（全局特征）
        channel_attention = self.center_extract(x)  # [B, C, 1, 1]
        
        # 2. 空间中心注意力（生成中心偏向的权重图）
        # 创建坐标图：距离中心越近，权重越高
        y_coord = torch.arange(H, dtype=torch.float32, device=x.device)
        x_coord = torch.arange(W, dtype=torch.float32, device=x.device)
        y_coord = (y_coord - H / 2) / (H / 2)  # 归一化到 [-1, 1]
        x_coord = (x_coord - W / 2) / (W / 2)  # 归一化到 [-1, 1]
        
        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing='ij')
        coord_map = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        spatial_attention = self.spatial_attention(coord_map)  # [B, 1, H, W]
        
        # 3. 特征增强
        enhanced_features = self.feature_enhance(x)
        
        # 4. 应用双重注意力：通道注意力 + 空间注意力
        # 中心区域和重要通道都得到强化
        enhanced = x + channel_attention * spatial_attention * enhanced_features
        
        return enhanced


class WidthAdaptiveAttention(nn.Module):
    """
    宽度自适应注意力模块
    
    创新点：根据物体尺寸动态调整注意力范围，大物体需要更大的注意力窗口
    原理：估计物体的尺度，自适应调整卷积核大小和注意力范围
    """
    def __init__(self, channels: int, num_scales: int = 3):
        """
        Args:
            channels: 输入特征通道数
            num_scales: 多尺度数量
        """
        super(WidthAdaptiveAttention, self).__init__()
        self.channels = channels
        self.num_scales = num_scales
        
        # 多尺度卷积（不同大小的感受野）
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=2*i+1, padding=i, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(1, num_scales + 1)  # kernel_size: 3, 5, 7
        ])
        
        # 尺度选择网络（根据特征动态选择合适尺度）
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_scales, kernel_size=1),
            nn.Softmax(dim=1)  # [B, num_scales, 1, 1]
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            宽度自适应增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            feat = conv(x)  # [B, C, H, W]
            multi_scale_features.append(feat)
        
        # 2. 尺度选择（根据特征内容动态选择）
        scale_weights = self.scale_selector(x)  # [B, num_scales, 1, 1]
        
        # 3. 加权融合多尺度特征
        adaptive_feat = torch.zeros_like(x)
        for i, feat in enumerate(multi_scale_features):
            weight = scale_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            adaptive_feat += weight * feat
        
        # 4. 特征融合
        enhanced = x + self.fusion(adaptive_feat)
        
        return enhanced


class AngleConsistencyAttention(nn.Module):
    """
    角度一致性注意力模块
    
    创新点：抓取角度应该垂直于边缘方向，此模块确保角度预测与边缘方向一致
    原理：从特征中提取方向信息，生成角度一致性注意力图
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: 输入特征通道数
        """
        super(AngleConsistencyAttention, self).__init__()
        self.channels = channels
        
        # 方向特征提取（提取边缘方向信息）
        self.direction_extract = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=1)  # 输出方向向量 (cos, sin)
        )
        
        # 角度一致性注意力生成
        self.consistency_attention = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            角度一致性增强后的特征 [B, C, H, W]
        """
        # 1. 提取方向特征（边缘方向）
        direction = self.direction_extract(x)  # [B, 2, H, W] (cos, sin)
        
        # 2. 拼接特征和方向信息
        x_with_dir = torch.cat([x, direction], dim=1)  # [B, C+2, H, W]
        
        # 3. 生成角度一致性注意力
        attention = self.consistency_attention(x_with_dir)  # [B, C, H, W]
        
        # 4. 特征增强
        enhanced_features = self.feature_enhance(x)
        
        # 5. 应用注意力：方向一致的区域得到强化
        enhanced = x + attention * enhanced_features
        
        return enhanced, direction  # 同时返回方向信息，可用于后续角度预测


class MultiScaleGraspFusion(nn.Module):
    """
    多尺度抓取融合模块
    
    创新点：不同尺度特征关注不同抓取属性
    - 细粒度特征 → 精确角度预测
    - 粗粒度特征 → 抓取质量评估
    - 中等特征 → 抓取宽度估计
    """
    def __init__(self, channels_list: list, out_channels: int = 192):
        """
        Args:
            channels_list: 不同尺度特征的通道数列表，如 [48, 96, 192]
            out_channels: 输出通道数
        """
        super(MultiScaleGraspFusion, self).__init__()
        self.num_scales = len(channels_list)
        
        # 为每个尺度创建特征适配层
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in channels_list
        ])
        
        # 尺度权重学习（学习不同尺度的重要性）
        self.scale_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sum(channels_list), out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list: list) -> torch.Tensor:
        """
        Args:
            features_list: 不同尺度的特征列表，如 [f1, f2, f3]
        
        Returns:
            融合后的特征 [B, out_channels, H, W]
        """
        B = features_list[0].shape[0]
        
        # 1. 适配所有尺度到统一通道数
        adapted_features = []
        for i, feat in enumerate(features_list):
            adapted = self.scale_adapters[i](feat)  # [B, out_channels, H, W]
            # 上采样到最大尺寸
            if adapted.shape[2:] != features_list[-1].shape[2:]:
                adapted = F.interpolate(
                    adapted, size=features_list[-1].shape[2:],
                    mode='bilinear', align_corners=True
                )
            adapted_features.append(adapted)
        
        # 2. 计算尺度权重
        # 拼接所有原始特征用于权重计算
        concat_feat = torch.cat([
            F.adaptive_avg_pool2d(feat, 1) for feat in features_list
        ], dim=1)  # [B, sum(channels), 1, 1]
        
        scale_weights = self.scale_weights(concat_feat)  # [B, num_scales, 1, 1]
        
        # 3. 加权融合
        fused = torch.zeros_like(adapted_features[0])
        for i, feat in enumerate(adapted_features):
            weight = scale_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            fused += weight * feat
        
        # 4. 最终融合
        output = self.fusion(fused)
        
        return output


class GraspAwareAttentionModule(nn.Module):
    """
    抓取感知注意力模块 (Grasp-Aware Attention Module, GAAM)
    
    核心创新模块：整合所有抓取感知注意力子模块
    
    架构流程：
    Input Features
        ↓
    [边缘感知注意力] → 强化边缘特征（抓取点位置）
        ↓
    [中心稳定性注意力] → 强化中心特征（抓取稳定性）
        ↓
    [宽度自适应注意力] → 自适应尺度（抓取宽度）
        ↓
    [角度一致性注意力] → 方向一致性（抓取角度）
        ↓
    Output Enhanced Features
    
    论文贡献：
    1. 首次将抓取任务的物理约束显式建模为注意力机制
    2. 多维度注意力协同工作，提升抓取预测准确性
    3. 可插入到任何抓取检测网络中，即插即用
    """
    def __init__(self, channels: int, use_edge: bool = True, 
                 use_center: bool = True, use_width: bool = True,
                 use_angle: bool = True):
        """
        Args:
            channels: 输入特征通道数
            use_edge: 是否使用边缘感知注意力
            use_center: 是否使用中心稳定性注意力
            use_width: 是否使用宽度自适应注意力
            use_angle: 是否使用角度一致性注意力
        """
        super(GraspAwareAttentionModule, self).__init__()
        self.channels = channels
        
        # 子模块
        self.edge_attention = EdgeAwareAttention(channels) if use_edge else None
        self.center_attention = CenterStabilityAttention(channels) if use_center else None
        self.width_attention = WidthAdaptiveAttention(channels) if use_width else None
        self.angle_attention = AngleConsistencyAttention(channels) if use_angle else None
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        # 残差连接的权重（可学习）
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, return_direction: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            return_direction: 是否返回方向信息（用于角度预测）
        
        Returns:
            增强后的特征 [B, C, H, W]
            如果 return_direction=True，还返回方向信息 [B, 2, H, W]
        """
        enhanced = x
        
        # 1. 边缘感知注意力
        if self.edge_attention is not None:
            enhanced = self.edge_attention(enhanced)
        
        # 2. 中心稳定性注意力
        if self.center_attention is not None:
            enhanced = self.center_attention(enhanced)
        
        # 3. 宽度自适应注意力
        if self.width_attention is not None:
            enhanced = self.width_attention(enhanced)
        
        # 4. 角度一致性注意力
        direction = None
        if self.angle_attention is not None:
            enhanced, direction = self.angle_attention(enhanced)
        
        # 5. 最终融合
        output = self.final_fusion(enhanced)
        
        # 6. 残差连接（可学习权重）
        output = self.residual_weight * x + output
        
        if return_direction and direction is not None:
            return output, direction
        return output


class GAAMDecoder(nn.Module):
    """
    集成 GAAM 的解码器
    
    在解码器的关键位置插入 GAAM 模块，提升抓取预测质量
    """
    def __init__(self, input_channels: int = 768, gaam_channels: int = 192,
                 use_gaam: bool = True):
        """
        Args:
            input_channels: 输入通道数（Swin输出）
            gaam_channels: GAAM模块的通道数
            use_gaam: 是否使用GAAM
        """
        super(GAAMDecoder, self).__init__()
        self.use_gaam = use_gaam
        
        # 初始降维
        self.reduce = nn.Sequential(
            nn.Conv2d(input_channels, gaam_channels, kernel_size=1),
            nn.BatchNorm2d(gaam_channels),
            nn.ReLU(inplace=True)
        )
        
        # GAAM模块（在解码器开始处）
        if use_gaam:
            self.gaam = GraspAwareAttentionModule(
                channels=gaam_channels,
                use_edge=True,
                use_center=True,
                use_width=True,
                use_angle=True
            )
        
        # 后续解码层（可以继续使用原有的解码器结构）
        # 这里简化，实际可以接原有的上采样层
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, input_channels, H, W]
        
        Returns:
            增强后的特征 [B, gaam_channels, H, W]
        """
        x = self.reduce(x)
        
        if self.use_gaam:
            x = self.gaam(x)
        
        return x

