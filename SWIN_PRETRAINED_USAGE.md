# 使用预训练 Swin Transformer 的 HybridGraspNet

## 🏗️ 架构说明

```
Input RGB-D [B, 4, 224, 224]
    ↓
CNN Backbone
    ├─→ F1 [B, 48, 112, 112]  (skip connection - 细节特征)
    ├─→ F2 [B, 96, 56, 56]    (送入 Swin + skip - 中层特征)
    └─→ F3 [B, 192, 28, 28]   (skip connection - 语义特征)
    
F2 [B, 96, 56, 56]
    ↓ Channel Adapter (96 → 3)
[B, 3, 56, 56]
    ↓ Upsample (56 → 224)
[B, 3, 224, 224]
    ↓ Swin Transformer (ImageNet-1K Pretrained)
[B, 768, 7, 7]
    ↓ Decoder
    ├─ Skip from F3 [192, 28, 28] (深层语义)
    ├─ Skip from F2 [96, 56, 56]  (中层特征)
    └─ Skip from F1 [48, 112, 112] (细节信息)
Output: 
  - Q (质量): [B, 1, 112, 112]
  - A (角度): [B, 2, 112, 112] (sin θ, cos θ)
  - W (宽度): [B, 1, 112, 112]
```

## 📦 模型参数

### 不使用预训练
```python
from models.serial_model import HybridGraspNet

model = HybridGraspNet(
    input_channels=4,
    use_pretrained=False,  # 从头训练
    swin_size='tiny'  # 'tiny', 'small', 'base'
)
```
- **参数量**: ~31.1M
  - CNN Backbone: 1.5M
  - Swin Transformer: 27.5M
  - Decoder: 2.1M (包含 F3 skip connection)

### 使用预训练（推荐）
```python
model = HybridGraspNet(
    input_channels=4,
    use_pretrained=True,  # 使用 ImageNet-1K 预训练
    swin_size='tiny'
)
```

## 🔧 下载预训练权重

### 方法1：自动下载（推荐）

```bash
# Windows (设置国内镜像)
set HF_ENDPOINT=https://hf-mirror.com
python train_ggcnn.py --network hybrid ...

# Linux
export HF_ENDPOINT=https://hf-mirror.com
python train_ggcnn.py --network hybrid ...
```

首次运行会自动下载 Swin-Tiny 权重（~110MB）到：
- Windows: `C:\Users\你的用户名\.cache\huggingface\hub\`
- Linux: `~/.cache/huggingface/hub/`

### 方法2：手动下载

1. 访问 Hugging Face 镜像站：
   - Swin-Tiny: https://hf-mirror.com/timm/swin_tiny_patch4_window7_224.ms_in1k
   - Swin-Small: https://hf-mirror.com/timm/swin_small_patch4_window7_224.ms_in1k
   - Swin-Base: https://hf-mirror.com/timm/swin_base_patch4_window7_224.ms_in1k

2. 下载 `model.safetensors` 文件

3. 放到缓存目录：
   ```
   C:\Users\24806\.cache\huggingface\hub\
       models--timm--swin_tiny_patch4_window7_224.ms_in1k\
           snapshots\main\
               model.safetensors
   ```

## 🚀 训练命令

```bash
python train_ggcnn.py \
    --network hybrid \
    --dataset cornell \
    --dataset-path /path/to/cornell \
    --use-depth 1 \
    --use-rgb 1 \
    --batch-size 8 \
    --epochs 50 \
    --description "Swin_Pretrained_Hybrid"
```

## 📊 消融实验建议

### 实验1：预训练 vs 随机初始化
```python
# 配置1：预训练（主方法）
model_pretrained = HybridGraspNet(input_channels=4, use_pretrained=True)

# 配置2：随机初始化（对比）
model_scratch = HybridGraspNet(input_channels=4, use_pretrained=False)
```

### 实验2：不同 Swin 规模
```python
# Swin-Tiny (快速，推荐)
model_tiny = HybridGraspNet(input_channels=4, swin_size='tiny')

# Swin-Small (更强)
model_small = HybridGraspNet(input_channels=4, swin_size='small')

# Swin-Base (最强，需要更多显存)
model_base = HybridGraspNet(input_channels=4, swin_size='base')
```

## ✅ 测试验证

```python
import torch
from models.serial_model import HybridGraspNet

# 创建模型（不使用预训练，快速测试）
model = HybridGraspNet(input_channels=4, use_pretrained=False)
model.eval()

# 测试输入
x = torch.randn(2, 4, 224, 224)

# 前向传播
with torch.no_grad():
    outputs = model(x, verbose=True)

# 检查输出
assert outputs['q'].shape == torch.Size([2, 1, 112, 112])
assert outputs['a'].shape == torch.Size([2, 2, 112, 112])
assert outputs['w'].shape == torch.Size([2, 1, 112, 112])

print("✓ 模型工作正常！")
```

## 📝 论文撰写要点

### 方法章节
- "We adopt the pre-trained Swin Transformer as the global feature aggregation module..."
- "The Swin Transformer, pre-trained on ImageNet-1K, provides rich visual representations..."
- "We fine-tune the entire network end-to-end on the grasp detection task..."

### 消融实验
| Method | Pretrained | Accuracy | IOU |
|--------|-----------|----------|-----|
| Baseline (CNN only) | No | - | - |
| Ours w/o pretrain | No | - | - |
| **Ours w/ pretrain** | **Yes** | **-** | **-** |

### 引用
```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and others},
  booktitle={ICCV},
  year={2021}
}
```

## ⚠️ 常见问题

### Q1: 网络连接问题
**A**: 使用国内镜像：`set HF_ENDPOINT=https://hf-mirror.com`

### Q2: 显存不足
**A**: 减小 batch size 或使用 `swin_size='tiny'`

### Q3: 预训练权重加载失败
**A**: 手动下载权重文件到缓存目录

## 🎯 性能对比

预期提升：
- **收敛速度**: 使用预训练可快 2-3倍
- **最终精度**: 预训练通常提高 2-5%
- **泛化能力**: 预训练模型泛化性更好

---

**模型已准备就绪！祝论文顺利！** 🚀

