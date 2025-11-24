"""
粗到精抓取感知注意力模块 (Coarse-to-Fine Grasp-Aware Attention Module, CF-GAAM)
集成粗到精预测框架的增强版GAAM

核心创新：
1. 粗略阶段：预测抓取概率热图，建模有效抓取姿态的空间连续分布
2. 精细阶段：以热图峰值为中心构建高斯注意力，引导模型聚焦关键区域
3. 高斯空间分布建模：有效抓取姿态在空间上呈连续分布趋势

设计理念：
- 观察到有效抓取姿态在空间上呈连续分布趋势
- 使用高斯分布建模该趋势
- 粗阶段：全局概率热图（哪些区域可能有抓取）
- 精阶段：局部高斯注意力（聚焦关键抓取区域）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .grasp_aware_attention import GraspAwareAttentionModule


class GaussianSpatialDistribution(nn.Module):
    """
    高斯空间分布模块
    
    创新点：建模有效抓取姿态在空间上的连续分布趋势
    原理：以热图峰值为中心，生成高斯分布权重图
    """
    def __init__(self, sigma_init: float = 1.5, learnable_sigma: bool = True):
        """
        Args:
            sigma_init: 初始高斯标准差
            learnable_sigma: 是否让sigma可学习
        """
        super(GaussianSpatialDistribution, self).__init__()
        self.learnable_sigma = learnable_sigma
        
        if learnable_sigma:
            # 可学习的高斯标准差（每个峰值可以有不同的sigma）
            self.sigma = nn.Parameter(torch.ones(1) * sigma_init)
        else:
            self.register_buffer('sigma', torch.tensor(sigma_init))
    
    def generate_gaussian_map(self, centers: torch.Tensor, H: int, W: int, 
                             sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成高斯分布权重图
        
        Args:
            centers: 峰值中心坐标 [B, N, 2] (N个峰值，每个峰值有(x,y)坐标)
            H: 图像高度
            W: 图像宽度
            sigma: 高斯标准差 [B, N] 或标量，如果为None则使用self.sigma
        
        Returns:
            高斯权重图 [B, 1, H, W]
        """
        B, N, _ = centers.shape
        device = centers.device
        
        # 如果没有提供sigma，使用默认值
        if sigma is None:
            sigma = self.sigma
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).unsqueeze(0).expand(B, N)  # [B, N]
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(0).expand(B, -1)  # [B, N]
        
        # 创建坐标网格
        y_coords = torch.arange(H, dtype=torch.float32, device=device)
        x_coords = torch.arange(W, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # [H, W] -> [1, 1, H, W]
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 对每个峰值生成高斯分布
        gaussian_maps = []
        for i in range(N):
            center = centers[:, i:i+1, :]  # [B, 1, 2]
            center_x = center[:, :, 0:1].unsqueeze(-1)  # [B, 1, 1, 1]
            center_y = center[:, :, 1:2].unsqueeze(-1)  # [B, 1, 1, 1]
            sigma_i = sigma[:, i:i+1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            
            # 计算距离
            dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            
            # 高斯分布: exp(-dist^2 / (2 * sigma^2))
            gaussian = torch.exp(-dist_sq / (2 * sigma_i ** 2 + 1e-6))
            gaussian_maps.append(gaussian)
        
        # 合并所有峰值的高斯分布（取最大值，避免重叠区域过度增强）
        gaussian_map = torch.stack(gaussian_maps, dim=1)  # [B, N, 1, H, W]
        gaussian_map, _ = torch.max(gaussian_map, dim=1)  # [B, 1, H, W]
        
        return gaussian_map
    
    def forward(self, centers: torch.Tensor, H: int, W: int, 
                sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            centers: 峰值中心坐标 [B, N, 2]
            H: 图像高度
            W: 图像宽度
            sigma: 可选的高斯标准差
        
        Returns:
            高斯权重图 [B, 1, H, W]
        """
        return self.generate_gaussian_map(centers, H, W, sigma)


class CoarseStagePredictor(nn.Module):
    """
    粗略阶段预测器
    
    功能：预测每个像素存在有效抓取姿态的概率热图
    输出：抓取概率热图 [B, 1, H, W]
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        """
        Args:
            in_channels: 输入特征通道数
            hidden_channels: 隐藏层通道数
        """
        super(CoarseStagePredictor, self).__init__()
        
        # 特征提取
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 概率热图预测
        self.probability_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, C, H, W]
        
        Returns:
            抓取概率热图 [B, 1, H, W]
        """
        feat = self.feature_extract(features)
        prob_map = self.probability_head(feat)
        return prob_map


class FineStageGaussianAttention(nn.Module):
    """
    精细阶段高斯注意力
    
    功能：以热图峰值为中心构建高斯注意力得分，重新加权特征
    原理：引导模型聚焦于关键抓取区域
    """
    def __init__(self, in_channels: int, num_peaks: int = 5, 
                 sigma_init: float = 1.5, learnable_sigma: bool = True):
        """
        Args:
            in_channels: 输入特征通道数
            num_peaks: 检测的峰值数量
            sigma_init: 初始高斯标准差
            learnable_sigma: 是否让sigma可学习
        """
        super(FineStageGaussianAttention, self).__init__()
        self.num_peaks = num_peaks
        
        # 高斯空间分布生成器
        self.gaussian_dist = GaussianSpatialDistribution(
            sigma_init=sigma_init,
            learnable_sigma=learnable_sigma
        )
        
        # 特征增强网络
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力权重生成（可选：根据特征内容调整高斯权重）
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
        检测概率热图中的峰值
        
        Args:
            prob_map: 概率热图 [B, 1, H, W]
            min_distance: 峰值之间的最小距离
            threshold: 峰值阈值
        
        Returns:
            峰值中心坐标 [B, N, 2] (N个峰值，每个峰值有(x,y)坐标)
        """
        B, _, H, W = prob_map.shape
        device = prob_map.device
        
        # 使用PyTorch实现峰值检测（不依赖scipy）
        all_peaks = []
        for b in range(B):
            prob = prob_map[b, 0]  # [H, W]
            
            # 使用最大池化检测局部最大值
            prob_unsqueezed = prob.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            kernel_size = min_distance * 2 + 1
            padding = min_distance
            local_max = F.max_pool2d(prob_unsqueezed, kernel_size=kernel_size, 
                                    stride=1, padding=padding)
            local_max = F.interpolate(local_max, size=(H, W), mode='bilinear', align_corners=False)
            local_max = local_max.squeeze()  # [H, W]
            
            # 找到局部最大值且大于阈值的点
            mask = (prob == local_max) & (prob > threshold)
            
            # 找到峰值坐标
            y_coords, x_coords = torch.where(mask)
            
            # 按概率值排序，取前N个
            if len(y_coords) > 0:
                peak_probs = prob[y_coords, x_coords]
                _, top_indices = torch.topk(peak_probs, min(self.num_peaks, len(y_coords)))
                y_coords = y_coords[top_indices]
                x_coords = x_coords[top_indices]
                
                # 转换为numpy格式 [N, 2] (x, y)
                peaks = torch.stack([x_coords.float(), y_coords.float()], dim=1).cpu().numpy()
            else:
                # 如果没有检测到峰值，使用图像中心
                peaks = np.array([[W//2, H//2]])
            
            # 确保至少有1个峰值
            if len(peaks) == 0:
                peaks = np.array([[W//2, H//2]])
            
            # 填充到固定数量
            if len(peaks) < self.num_peaks:
                padding = np.tile(peaks[0:1], (self.num_peaks - len(peaks), 1))
                peaks = np.vstack([peaks, padding])
            elif len(peaks) > self.num_peaks:
                peaks = peaks[:self.num_peaks]
            
            all_peaks.append(peaks)
        
        # 转换为tensor [B, N, 2]
        peaks_tensor = torch.tensor(np.array(all_peaks), dtype=torch.float32, device=device)
        return peaks_tensor
    
    def forward(self, features: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, C, H, W]
            prob_map: 粗略阶段的概率热图 [B, 1, H, W]
        
        Returns:
            高斯注意力增强后的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # 1. 检测峰值
        peaks = self.detect_peaks(prob_map)  # [B, N, 2]
        
        # 2. 生成高斯注意力图
        gaussian_attention = self.gaussian_dist(peaks, H, W)  # [B, 1, H, W]
        
        # 3. 特征增强
        enhanced_features = self.feature_enhance(features)
        
        # 4. 可选的注意力适配（根据特征内容调整高斯权重）
        feat_with_attn = torch.cat([features, gaussian_attention], dim=1)  # [B, C+1, H, W]
        adaptive_attention = self.attention_adapter(feat_with_attn)  # [B, 1, H, W]
        
        # 5. 应用高斯注意力：关键区域得到强化
        # 结合原始高斯注意力和自适应注意力
        final_attention = (gaussian_attention + adaptive_attention) / 2
        output = features + final_attention * enhanced_features
        
        return output, peaks, gaussian_attention


class CoarseToFineGAAM(nn.Module):
    """
    粗到精抓取感知注意力模块 (Coarse-to-Fine GAAM)
    
    集成粗到精预测框架的增强版GAAM
    
    架构流程：
    Input Features
        ↓
    [粗略阶段] → 预测抓取概率热图
        ↓
    [精细阶段] → 以峰值为中心生成高斯注意力
        ↓
    [原始GAAM] → 边缘/中心/宽度/角度注意力
        ↓
    Output Enhanced Features
    
    论文贡献：
    1. 首次将粗到精预测框架与抓取感知注意力结合
    2. 高斯空间分布建模有效抓取姿态的连续分布趋势
    3. 粗略阶段全局感知 + 精细阶段局部聚焦
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
            channels: 输入特征通道数
            use_coarse_fine: 是否使用粗到精框架
            num_peaks: 检测的峰值数量
            sigma_init: 初始高斯标准差
            use_gaam: 是否使用原始GAAM模块
            use_edge: 是否使用边缘感知注意力
            use_center: 是否使用中心稳定性注意力
            use_width: 是否使用宽度自适应注意力
            use_angle: 是否使用角度一致性注意力
        """
        super(CoarseToFineGAAM, self).__init__()
        self.channels = channels
        self.use_coarse_fine = use_coarse_fine
        self.use_gaam = use_gaam
        
        # 粗略阶段：概率热图预测
        if use_coarse_fine:
            self.coarse_predictor = CoarseStagePredictor(channels)
            self.fine_attention = FineStageGaussianAttention(
                channels, 
                num_peaks=num_peaks,
                sigma_init=sigma_init,
                learnable_sigma=True
            )
        
        # 精细阶段：原始GAAM模块
        if use_gaam:
            self.gaam = GraspAwareAttentionModule(
                channels=channels,
                use_edge=use_edge,
                use_center=use_center,
                use_width=use_width,
                use_angle=use_angle
            )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        # 残差连接的权重（可学习）
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            return_aux: 是否返回辅助信息（概率热图、峰值等）
        
        Returns:
            增强后的特征 [B, C, H, W]
            如果 return_aux=True，还返回 (prob_map, peaks, gaussian_attn)
        """
        enhanced = x
        prob_map = None
        peaks = None
        gaussian_attn = None
        
        # ============ 粗略阶段 ============
        if self.use_coarse_fine:
            # 1. 预测抓取概率热图
            prob_map = self.coarse_predictor(enhanced)  # [B, 1, H, W]
            
            # 2. 精细阶段：以峰值为中心生成高斯注意力
            enhanced, peaks, gaussian_attn = self.fine_attention(enhanced, prob_map)
        
        # ============ 精细阶段：原始GAAM ============
        if self.use_gaam:
            enhanced = self.gaam(enhanced)
        
        # ============ 最终融合 ============
        output = self.final_fusion(enhanced)
        
        # 残差连接
        output = self.residual_weight * x + output
        
        if return_aux and self.use_coarse_fine:
            return output, prob_map, peaks, gaussian_attn
        return output


class CFGAAMLoss(nn.Module):
    """
    粗到精GAAM的损失函数
    
    包含：
    1. 粗略阶段损失：概率热图与真实抓取位置的损失
    2. 精细阶段损失：高斯注意力区域的抓取质量损失
    """
    def __init__(self, coarse_weight: float = 1.0, fine_weight: float = 1.0):
        """
        Args:
            coarse_weight: 粗略阶段损失权重
            fine_weight: 精细阶段损失权重
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
            prob_map: 预测的概率热图 [B, 1, H, W]
            gt_pos: 真实抓取位置图 [B, 1, H, W]
            peaks: 检测到的峰值 [B, N, 2]
            gaussian_attn: 高斯注意力图 [B, 1, H, W]
        
        Returns:
            损失字典
        """
        # 粗略阶段损失：概率热图与真实位置的MSE
        coarse_loss = self.mse_loss(prob_map, gt_pos)
        
        # 精细阶段损失：高斯注意力区域的抓取质量
        # 在高斯注意力高的区域，抓取质量应该更高
        fine_loss = -torch.mean(gaussian_attn * gt_pos)  # 最大化高斯区域的质量
        
        total_loss = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss
        
        return {
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss,
            'total_loss': total_loss
        }

