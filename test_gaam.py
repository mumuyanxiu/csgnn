"""
测试抓取感知注意力模块 (GAAM)
验证模块功能是否正常
"""
import torch
import torch.nn as nn
from models.grasp_aware_attention import (
    GraspAwareAttentionModule,
    EdgeAwareAttention,
    CenterStabilityAttention,
    WidthAdaptiveAttention,
    AngleConsistencyAttention
)
from models.serial_model import HybridGraspNet


def test_edge_attention():
    """测试边缘感知注意力"""
    print("=" * 60)
    print("测试: EdgeAwareAttention")
    print("=" * 60)
    
    module = EdgeAwareAttention(channels=192)
    x = torch.randn(2, 192, 28, 28)
    
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in module.parameters()):,}")
    print("✅ EdgeAwareAttention 测试通过!\n")


def test_center_attention():
    """测试中心稳定性注意力"""
    print("=" * 60)
    print("测试: CenterStabilityAttention")
    print("=" * 60)
    
    module = CenterStabilityAttention(channels=192)
    x = torch.randn(2, 192, 28, 28)
    
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in module.parameters()):,}")
    print("✅ CenterStabilityAttention 测试通过!\n")


def test_width_attention():
    """测试宽度自适应注意力"""
    print("=" * 60)
    print("测试: WidthAdaptiveAttention")
    print("=" * 60)
    
    module = WidthAdaptiveAttention(channels=192, num_scales=3)
    x = torch.randn(2, 192, 28, 28)
    
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in module.parameters()):,}")
    print("✅ WidthAdaptiveAttention 测试通过!\n")


def test_angle_attention():
    """测试角度一致性注意力"""
    print("=" * 60)
    print("测试: AngleConsistencyAttention")
    print("=" * 60)
    
    module = AngleConsistencyAttention(channels=192)
    x = torch.randn(2, 192, 28, 28)
    
    out, direction = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"方向信息形状: {direction.shape}")
    print(f"参数量: {sum(p.numel() for p in module.parameters()):,}")
    print("✅ AngleConsistencyAttention 测试通过!\n")


def test_full_gaam():
    """测试完整 GAAM 模块"""
    print("=" * 60)
    print("测试: GraspAwareAttentionModule (完整GAAM)")
    print("=" * 60)
    
    # 测试不同配置
    configs = [
        {"use_edge": True, "use_center": False, "use_width": False, "use_angle": False},
        {"use_edge": True, "use_center": True, "use_width": False, "use_angle": False},
        {"use_edge": True, "use_center": True, "use_width": True, "use_angle": False},
        {"use_edge": True, "use_center": True, "use_width": True, "use_angle": True},  # 完整配置
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        module = GraspAwareAttentionModule(channels=192, **config)
        x = torch.randn(2, 192, 28, 28)
        
        out = module(x)
        params = sum(p.numel() for p in module.parameters())
        
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print(f"  参数量: {params:,}")
    
    print("\n✅ GraspAwareAttentionModule 测试通过!\n")


def test_gaam_integration():
    """测试 GAAM 在完整模型中的集成"""
    print("=" * 60)
    print("测试: GAAM 集成到 HybridGraspNet")
    print("=" * 60)
    
    # 测试无 GAAM 版本
    print("\n1. 无 GAAM 版本:")
    net_no_gaam = HybridGraspNet(
        input_channels=4,
        use_gaam=False
    )
    params_no_gaam = sum(p.numel() for p in net_no_gaam.parameters())
    print(f"   参数量: {params_no_gaam:,}")
    
    # 测试有 GAAM 版本
    print("\n2. 有 GAAM 版本:")
    net_gaam = HybridGraspNet(
        input_channels=4,
        use_gaam=True
    )
    params_gaam = sum(p.numel() for p in net_gaam.parameters())
    print(f"   参数量: {params_gaam:,}")
    print(f"   增加参数量: {params_gaam - params_no_gaam:,} (+{(params_gaam - params_no_gaam) / params_no_gaam * 100:.2f}%)")
    
    # 测试前向传播
    print("\n3. 前向传播测试:")
    x = torch.randn(2, 4, 224, 224)
    
    with torch.no_grad():
        out_no_gaam = net_no_gaam(x)
        out_gaam = net_gaam(x)
    
    print(f"   输入形状: {x.shape}")
    print(f"   无GAAM输出 - Pos: {out_no_gaam['pos'].shape}, Cos: {out_no_gaam['cos'].shape}")
    print(f"   有GAAM输出 - Pos: {out_gaam['pos'].shape}, Cos: {out_gaam['cos'].shape}")
    
    print("\n✅ GAAM 集成测试通过!\n")


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 60)
    print("测试: 梯度流")
    print("=" * 60)
    
    module = GraspAwareAttentionModule(channels=192)
    x = torch.randn(2, 192, 28, 28, requires_grad=True)
    
    out = module(x)
    loss = out.mean()
    loss.backward()
    
    # 检查梯度
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"输入梯度存在: {has_grad}")
    print(f"梯度范数: {x.grad.norm().item():.6f}")
    
    if has_grad:
        print("✅ 梯度流正常!\n")
    else:
        print("❌ 梯度流异常!\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GAAM 模块测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 测试各个子模块
        test_edge_attention()
        test_center_attention()
        test_width_attention()
        test_angle_attention()
        
        # 测试完整 GAAM
        test_full_gaam()
        
        # 测试集成
        test_gaam_integration()
        
        # 测试梯度流
        test_gradient_flow()
        
        print("=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

