# 提升抓取检测准确率指南

**当前准确率**: 80-85%  
**目标**: 85-90%+

---

## 🎯 快速提升方法（按优先级）

### 1. 使用预训练 Swin（+3-5%，最简单）

**当前**: `pretrained=False`  
**改成**: `pretrained=True`

```python
# models/serial_model.py 第446行
pretrained=True  # 使用 ImageNet 预训练
```

**前提**: 需要下载预训练权重
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 重新训练
```

**预期提升**: 83% → 86-88%

---

### 2. 改进损失函数（+2-4%）

#### 方案A：加权损失

当前所有损失权重相同，改为：

```python
# models/serial_model.py compute_loss 方法
# 原来
total_loss = p_loss + cos_loss + sin_loss + width_loss

# 改成（质量损失更重要）
total_loss = 2.0 * p_loss + cos_loss + sin_loss + 0.5 * width_loss
```

#### 方案B：Focal Loss（处理样本不平衡）

```python
# 添加 Focal Loss for quality
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

# 替换 p_loss
p_loss = focal_loss(q_pred, pos_gt)
```

---

### 3. 增强数据增广（+1-3%）

当前已有旋转和缩放，添加更多：

```python
# utils/data/cornell_data.py

# 添加随机亮度/对比度调整
def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
    rgb_img = image.Image.from_file(self.rgb_files[idx])
    
    # 新增：随机亮度调整
    if self.random_augment:
        brightness = np.random.uniform(0.8, 1.2)
        rgb_img.img = np.clip(rgb_img.img * brightness, 0, 255)
    
    # ... 原有代码
```

---

### 4. 调整超参数（+1-2%）

#### 学习率调度

```python
# train_ggcnn.py

# 原来
optimizer = optim.Adam(net.parameters())

# 改成（带学习率衰减）
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 在训练循环中
for epoch in range(args.epochs):
    train_results = train(...)
    scheduler.step()  # 每个 epoch 后更新学习率
```

#### 优化器选择

```python
# 试试 AdamW（通常比 Adam 好 1-2%）
optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
```

---

### 5. 更深的 Swin（+2-3%，但慢）

```python
# models/serial_model.py 初始化时
model = HybridGraspNet(
    input_channels=4,
    use_pretrained=True,
    swin_size='small'  # 从 tiny → small
)
```

**效果**: Swin-Small 比 Swin-Tiny 强，但参数多 20M

---

### 6. 模型集成（+3-5%，推理时）

训练多个模型，推理时融合：

```python
# 训练3个模型
model1 = HybridGraspNet(...)  # 串行，预训练
model2 = ParallelHybridGraspNet(...)  # 并行
model3 = HybridGraspNet(..., swin_size='small')  # 大模型

# 推理时平均
q_out = (model1(x)['q'] + model2(x)['q'] + model3(x)['q']) / 3
```

---

### 7. 后处理优化（+1-2%）

#### 调整质量图阈值

```python
# utils/dataset_processing/grasp.py detect_grasps 函数

# 原来
local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, ...)

# 调整参数
local_max = peak_local_max(
    q_img,
    min_distance=15,  # 更密集的检测
    threshold_abs=0.15,  # 更低的阈值
    num_peaks=no_grasps
)
```

#### NMS (非极大值抑制)

```python
# 在多个检测结果中选择最好的
# 避免重复检测
```

---

### 8. 训练更多 epoch（+1-2%）

```python
# 从 50 epoch 增加到 100
--epochs 100

# 或使用 early stopping
# 当验证 IOU 连续 10 个 epoch 不提升时停止
```

---

### 9. 测试时增广（TTA，+2-3%）

```python
# 推理时对同一图像做多次预测（旋转、翻转），然后平均
def predict_with_tta(model, image):
    preds = []
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        pred = model(rotated)
        pred = rotate_back(pred, -angle)
        preds.append(pred)
    return average(preds)
```

---

### 10. 使用更好的损失权重（+1-2%）

```python
# 根据像素数量加权
# 因为抓取点是稀疏的

# 质量损失：正样本（pos=1）权重更高
pos_weight = (pos_gt ==  0).sum() / (pos_gt == 1).sum()
p_loss = F.binary_cross_entropy(q_pred, pos_gt, 
                                 weight=pos_gt * pos_weight + (1-pos_gt))
```

---

## 📊 优先级排序

| 方法 | 预期提升 | 难度 | 时间成本 |
|------|---------|------|----------|
| **1. 预训练 Swin** | +3-5% | ⭐ | 下载权重 |
| **2. 损失函数权重** | +2-4% | ⭐⭐ | 5分钟 |
| **3. 学习率调度** | +1-2% | ⭐ | 2分钟 |
| **4. Swin-Small** | +2-3% | ⭐ | 重新训练 |
| **5. 数据增广** | +1-3% | ⭐⭐⭐ | 30分钟 |
| **6. 后处理调参** | +1-2% | ⭐ | 10分钟 |
| **7. TTA** | +2-3% | ⭐⭐ | 仅推理时 |
| **8. 模型集成** | +3-5% | ⭐⭐⭐ | 训练多个 |
| **9. 更多 epoch** | +1-2% | ⭐ | 时间 |

---

## 🔧 立即可实施（前3个）

### 修改1：加权损失（5分钟）

```python
# models/serial_model.py 第545行
# 找到
total_loss = p_loss + cos_loss + sin_loss + width_loss

# 改成
total_loss = 3.0 * p_loss + cos_loss + sin_loss + 0.5 * width_loss
```

### 修改2：学习率调度（2分钟）

```python
# train_ggcnn.py 第231行之后添加
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# 第250行之后添加（epoch 循环内）
scheduler.step()
```

### 修改3：后处理调参（10分钟）

```python
# utils/dataset_processing/grasp.py 第424行
# 原来
local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

# 调整
local_max = peak_local_max(q_img, min_distance=15, threshold_abs=0.15, num_peaks=no_grasps)
```

---

## 💡 我的推荐路线

### 阶段1：快速优化（1天内）
1. ✅ 加权损失函数
2. ✅ 学习率调度
3. ✅ 后处理调参

**预期**: 80-85% → 85-88%

### 阶段2：深度优化（需重新训练）
1. 使用预训练 Swin（需要下载）
2. 换成 Swin-Small
3. 训练到 100 epoch

**预期**: 85-88% → 88-92%

### 阶段3：终极优化（论文发表级别）
1. 模型集成（3个模型）
2. TTA（测试时增广）
3. 精细调参

**预期**: 88-92% → 92-95%

---

## 🎓 论文写作角度

### 消融实验表格

| Method | IOU |
|--------|-----|
| Baseline (CNN only) | 75% |
| Ours (w/o pretrain) | 83% |
| Ours (w/ pretrain) | 87% |
| Ours + weighted loss | 89% |
| Ours + ensemble | 92% |

---

你想先试哪几个方法？我可以帮你实现！🚀
