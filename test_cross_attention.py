"""
测试 Cross-Attention Fusion 模块
验证功能和性能对比
"""
import torch
import time
from models.parallel_model import ParallelHybridGraspNet

def test_fusion_modes():
    """测试三种融合模式的功能和速度"""
    
    print("=" * 60)
    print("Cross-Attention Fusion 功能测试")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")
    
    # 测试输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 4, 224, 224).to(device)
    
    fusion_modes = ['simple', 'attention', 'cross_attention']
    
    for mode in fusion_modes:
        print(f"\n{'=' * 60}")
        print(f"测试模式: {mode.upper()}")
        print(f"{'=' * 60}")
        
        # 创建模型
        try:
            model = ParallelHybridGraspNet(
                in_chans=4,
                use_pretrained=False,  # 测试时不用预训练，节省时间
                swin_size='tiny',
                fusion_type=mode,
                num_heads=8 if mode == 'cross_attention' else 4
            ).to(device)
            
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n模型参数:")
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model(input_tensor)
                
                # 计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                num_iterations = 10
                
                for _ in range(num_iterations):
                    outputs = model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                avg_time = elapsed / num_iterations
                fps = batch_size / avg_time
                
            print(f"\n性能测试:")
            print(f"  平均推理时间: {avg_time*1000:.2f} ms")
            print(f"  吞吐量: {fps:.2f} FPS")
            
            print(f"\n输出形状:")
            for key, value in outputs.items():
                print(f"  {key}: {value.shape}")
            
            # 验证输出范围
            print(f"\n输出值范围:")
            print(f"  Q (质量): [{outputs['q'].min().item():.3f}, {outputs['q'].max().item():.3f}]")
            print(f"  A (角度): [{outputs['a'].min().item():.3f}, {outputs['a'].max().item():.3f}]")
            print(f"  W (宽度): [{outputs['w'].min().item():.3f}, {outputs['w'].max().item():.3f}]")
            
            print(f"\n✅ {mode.upper()} 模式测试通过!")
            
            # 清理显存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n❌ {mode.upper()} 模式测试失败!")
            print(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("所有测试完成!")
    print(f"{'=' * 60}")


def test_cross_attention_gradients():
    """测试 Cross-Attention 的梯度流"""
    
    print(f"\n{'=' * 60}")
    print("Cross-Attention 梯度流测试")
    print(f"{'=' * 60}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ParallelHybridGraspNet(
        in_chans=4,
        use_pretrained=False,
        fusion_type='cross_attention',
        num_heads=8
    ).to(device)
    
    # 创建随机输入和标签
    x = torch.randn(2, 4, 224, 224).to(device)
    
    # 模拟标签
    pos_gt = torch.rand(2, 1, 224, 224).to(device)
    cos_gt = torch.randn(2, 1, 224, 224).to(device)
    sin_gt = torch.randn(2, 1, 224, 224).to(device)
    width_gt = torch.rand(2, 1, 224, 224).to(device)
    
    yc = [pos_gt, cos_gt, sin_gt, width_gt]
    
    # 前向传播
    model.train()
    lossd = model.compute_loss(x, yc)
    
    print(f"\n损失值:")
    print(f"  总损失: {lossd['loss'].item():.4f}")
    for name, loss in lossd['losses'].items():
        print(f"  {name}: {loss.item():.4f}")
    
    # 反向传播
    lossd['loss'].backward()
    
    # 检查关键层的梯度
    gradient_check = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_check[name] = grad_norm
    
    # 打印 Cross-Attention 相关层的梯度
    print(f"\nCross-Attention 层梯度范数（前10个）:")
    cross_att_grads = {k: v for k, v in gradient_check.items() if 'cross_attention' in k}
    
    for i, (name, grad_norm) in enumerate(list(cross_att_grads.items())[:10]):
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
        print(f"  {short_name}: {grad_norm:.6f}")
    
    if cross_att_grads:
        avg_grad = sum(cross_att_grads.values()) / len(cross_att_grads)
        print(f"\n平均梯度范数: {avg_grad:.6f}")
        
        if avg_grad > 1e-6:
            print("✅ 梯度流正常!")
        else:
            print("⚠️  梯度过小，可能存在梯度消失")
    else:
        print("⚠️  未找到 Cross-Attention 层梯度")
    
    print(f"\n{'=' * 60}")


def compare_fusion_performance():
    """对比三种融合模式的性能"""
    
    print(f"\n{'=' * 60}")
    print("融合模式性能对比")
    print(f"{'=' * 60}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for mode in ['simple', 'attention', 'cross_attention']:
        model = ParallelHybridGraspNet(
            in_chans=4,
            use_pretrained=False,
            fusion_type=mode,
            num_heads=8
        ).to(device)
        
        # 参数量
        params = sum(p.numel() for p in model.parameters())
        
        # 推理速度
        x = torch.randn(4, 4, 224, 224).to(device)
        model.eval()
        
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(20):
                _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / 20
        
        results[mode] = {
            'params': params,
            'time': elapsed * 1000,  # ms
            'fps': 4 / elapsed
        }
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 打印对比表格
    print(f"\n{'模式':<20} {'参数量':<15} {'推理时间':<15} {'FPS':<10}")
    print("-" * 60)
    
    for mode, stats in results.items():
        print(f"{mode:<20} {stats['params']:>13,}  {stats['time']:>10.2f} ms  {stats['fps']:>8.1f}")
    
    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    print("\n🚀 开始测试 Cross-Attention Fusion 模块\n")
    
    # 测试1: 功能测试
    test_fusion_modes()
    
    # 测试2: 梯度流测试
    test_cross_attention_gradients()
    
    # 测试3: 性能对比
    compare_fusion_performance()
    
    print("\n✅ 所有测试完成!\n")

