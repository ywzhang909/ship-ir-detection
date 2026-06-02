# NSLSR-LWIR 船舶检测训练记录

> 红外海事船舶检测 | YOLO11m | 空域+频域预处理 + 可学习融合网络
> 日期: 2026-05-28 | GPU: NVIDIA RTX 5080 Laptop (16GB VRAM) | CUDA 12.8

---

## 目录

1. [项目概览](#1-项目概览)
2. [Commit 日志](#2-commit-日志)
3. [运行对比总表](#3-运行对比总表)
4. [Run 1: 空域预处理](#4-run-1-空域预处理)
5. [Run 2: 频域预处理](#5-run-2-频域预处理)
6. [Run 3: 空频融合](#6-run-3-空频融合)
7. [融合架构详解](#7-融合架构详解)
8. [已知问题](#8-已知问题)
9. [下一步修改建议](#9-下一步修改建议)

---

## 1. 项目概览

### 目标
构建面向红外海事场景的舰船检测流水线，包含：
- **预处理**: 自动白平衡 (AWB) + 空域滤波 (top-hat/bottom-hat) + 频域滤波 (Butterworth BPF) + 图像增强 (CLAHE/Retinex/Unsharp)
- **融合网络**: 可学习的空域-频域特征融合模块 (`SpatialFrequencyFusion`)
- **检测模型**: YOLO11m (20M 参数)
- **实验追踪**: Weights & Biases (wandb)

### 数据集
- NSLSR-LWIR: 803 训练 + 243 验证 + 118 测试
- RGB 伪彩色红外图像 (512×640, 3 通道)
- 单类别: "ship"，共 559 个验证实例

### 核心文件

| 文件 | 用途 |
|------|------|
| `train.py` | 训练入口，含融合模式、wandb 集成、预处理缓存 |
| `preprocessing_module.py` | 预处理流水线：AWB、空域/频域滤波、增强、双路融合 |
| `fusion_module.py` | `SpatialFrequencyFusion` 网络 + 模型注入 + 自定义 Trainer |
| `configs/train_nslsr_optimized.yaml` | 训练超参数 |
| `configs/data_nslsr_lwir.yaml` | 数据集配置 |

---

## 2. Commit 日志

| # | 时间 | Hash | 描述 | 类型 |
|---|------|------|------|------|
| 1 | 09:07 | `0aee456` | Initial commit: YOLO11m ship detection pipeline (NSLSR SWIR training + SAHI inference) | init |
| 2 | 09:51 | `d09a4f5` | feat: IR preprocessing pipeline with AWB, freq/spatial filtering, enhancement | feature |
| 3 | 10:35 | `29bc0d8` | refactor: use Ultralytics built-in wandb integration instead of manual init | refactor |
| 4 | 10:41 | `e2ab2f0` | fix: dispatch correct params per sea_clutter method in process_stage() | fix |
| 5 | 11:49 | `75de5ab` | feat: integrate dual-stream fusion pipeline + train.py fusion mode | feature |
| 6 | 11:59 | `6291aaa` | fix: use FusionDetectionTrainer to inject fusion module at correct point | fix |
| 7 | 14:35 | `aac66c3` | fix: resolve all 12 issues identified in architecture review | fix |
| 8 | 15:19 | `712031a` | feat: fusion training run completed - YOLO11m+SpatialFrequencyFusion | result |

### Commit 2 — IR 预处理流水线
- 实现 `PreprocessingPipeline` 类
- 自动白平衡 (AWB): 基于海面角落灰度世界假设
- 空域海面杂波滤波: Top-hat + Bottom-hat 形态学滤波
- 频域海面杂波滤波: Butterworth 带通滤波器
- 图像增强: CLAHE, Retinex, Unsharp Mask
- 3 个预置配置: `awb_tophat_clahe`, `awb_butterworth_clahe`, `awb_butterworth_retinex`
- 文件: `preprocessing_module.py`

### Commit 3 — Wandb 集成重构
- 移除手动 wandb init/login/finish
- 使用 Ultralytics 内置 wandb 回调 (`yolo settings wandb=True`)
- 文件: `train.py`

### Commit 4 — 参数分发修复
- 修复 `process_stage()` 中不同海面杂波方法参数分发问题
- 确保每个方法使用正确的默认参数
- 文件: `preprocessing_module.py`

### Commit 5 — 融合流水线
- 创建 `fusion_module.py`: `SpatialFrequencyFusion` (3→16→16→3, SE-attention + residual)
- 双路预处理: `PreprocessingPipelineDual` + `awb_dual_fusion` 预置
- 训练参数: `--use-fusion`, `--fusion-hidden`, `--fusion-lr-scale`
- 预处理缓存: `data/preprocessed_awb_dual_fusion/`
- 文件: `fusion_module.py`, `preprocessing_module.py`, `train.py`

### Commit 6 — FusionDetectionTrainer 修复
- 使用 `make_fusion_trainer_class()` 创建自定义 Trainer
- 在正确位置注入融合模块 (model.0 的 Sequential 中)
- 解决加载预训练权重时 state dict 不匹配问题
- 文件: `fusion_module.py`

### Commit 7 — 架构审查修复 (12 issues)
- **C4**: 通道顺序语义修正 — `[orig, freq, spatial]` 堆叠顺序修复
- **C3**: 删除死代码 NWD loss (从来未被 Ultralytics 使用)
- **C5**: 输入校验 — `--use-fusion` 必须搭配 `awb_dual_fusion`
- **C1+C2**: Resume 修正 — `resume=True` 正确传递到 `model.train()`
- **H1**: 融合 LR 缩放 — 分离参数组，独立学习率
- **H3**: 残差 warmup — `residual_alpha` 从 0.0→0.5 线性增长 (前 10 epoch)
- **H2**: 融合监控回调 — wandb 记录 alpha、通道统计、MSE、梯度范数
- **M5**: `F.silu(inplace=True→False)` 适配 ONNX 导出
- **M1**: 修复缓存双层嵌套目录
- **M2**: 合并重复的插入函数
- **M3/M4/M7/H4**: 清理死变量、统一 dict 格式、扩展 known_keys、独立 wandb 字段
- 文件: `fusion_module.py`, `preprocessing_module.py`, `train.py`

### Commit 8 — 融合训练结果
- Run 3 完整训练结果
- 详见 [Run 3](#6-run-3-空频融合)
- 文件: `runs/` 目录结果

---

## 3. 运行对比总表

| 指标 | Run 1: 空域 | Run 2: 频域 | 🏆 Run 3: Fusion |
|------|:-----------:|:-----------:|:----------------:|
| **方法** | awb_tophat_clahe | awb_butterworth_clahe | awb_dual_fusion |
| **mAP50** | 0.9396 | 0.9465 | **0.9568** ✅ |
| **mAP50-95** | 0.6366 | 0.6433 | **0.6551** ✅ |
| **Precision** | 0.9358 | **0.9751** ✅ | 0.9569 |
| **Recall** | 0.9052 | 0.9302 | **0.9532** ✅ |
| **最佳 Epoch** | 132 | 126 | 118 |
| **总 Epochs** | 150 | 146 (early stop) | 138 (early stop) |
| **训练时间** | ~0.6h | ~0.6h | ~0.6h |
| **参数** | 20.0M | 20.0M | 20.0M + 3,361 |
| **推理速度** | — | — | 3.1ms/img |
| **ONNX 大小** | — | — | 76.7MB (126 layers) |
| **Wandb Run** | `fbl2s246` | `nslsr-lwir-yolo11m-awb_butterworth_clahe_20260528_104253` | `nslsr-lwir-yolo11m-awb_dual_fusion-fusion_20260528_144057` |

### 关键发现
1. **Recall 提升最大**: Fusion 相比空域 Recall 提升 +4.8%，相比频域提升 +2.3%
2. **频域 Precision 最优**: 纯频域方法 Precision=0.9751，但 Recall 较低
3. **Fusion 综合最优**: mAP50 和 mAP50-95 均为最高，同时 Recall 显著改善
4. **融合模块开销极小**: 仅 3,361 参数 (0.017% 总量)，推理 3.1ms

---

## 4. Run 1: 空域预处理

### 训练命令
```powershell
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_tophat_clahe --wandb --epochs 150 --batch 16 --imgsz 640
```

### 预处理流程
```
输入(RGB) → AWB(海面角落灰度世界) → Top-hat + Bottom-hat → CLAHE → 输出
```

### 最佳验证结果 (Epoch 132)

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|-------|--------|-----------|---|---|-------|----------|
| all | 243 | 559 | 0.936 | 0.905 | 0.940 | 0.637 |

### 学习曲线
- mAP50: 0.9396 @ epoch 132
- mAP50-95: 0.6366 @ epoch 132
- Precision: 0.9358
- Recall: 0.9052

### 分析
- 空域 Top-hat 能有效增强亮目标 (舰船)，但 Recall 偏低说明仍有漏检
- Precision 相对较低 (0.936)，可能存在虚警

---

## 5. Run 2: 频域预处理

### 训练命令
```powershell
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_butterworth_clahe --wandb --epochs 150 --batch 16 --imgsz 640
```

### 预处理流程
```
输入(RGB) → AWB(海面角落灰度世界) → Butterworth BPF → CLAHE → 输出
```

### 最佳验证结果 (Epoch 126)

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|-------|--------|-----------|---|---|-------|----------|
| all | 243 | 559 | 0.975 | 0.930 | 0.947 | 0.643 |

### 学习曲线
- mAP50: 0.9465 @ epoch 126
- mAP50-95: 0.6433 @ epoch 126
- Precision: 0.9751 (三 run 最高)
- Recall: 0.9302

### 分析
- 频域 Butterworth BPF 有效抑制了周期性海杂波，Precision 达到最高
- Recall 相比空域有改善 (+2.5%)，但仍存在漏检
- mAP50-95 略高于空域 (+0.7%)

---

## 6. Run 3: 空频融合

### 训练命令
```powershell
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-hidden 16 --fusion-lr-scale 10.0 --wandb --epochs 150 --batch 16 --imgsz 640
```

### 预处理流程
```
输入(RGB)
  ├→ AWB → Top-hat + Bottom-hat → CLAHE → [spatial_gray]
  ├→ AWB → Butterworth BPF → CLAHE → [freq_gray]
  └→ AWB → [orig_gray]

stack → [orig, freq, spatial] (BGR) → Fusion Model
```

### 最佳验证结果 (Epoch 118)

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|-------|--------|-----------|---|---|-------|----------|
| all | 243 | 559 | 0.957 | 0.953 | 0.957 | 0.655 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | YOLO11m (20,030,803 params) |
| 融合模块 | SpatialFrequencyFusion (3,361 params) |
| 融合隐藏维度 | 16 |
| 融合 LR 缩放 | 10.0× |
| 残差 warmup | 0.0→0.5 (前 10 epochs) |
| SE reduction | 4 |
| GPU 显存 | ~8.06 GB |
| batch size | 16 |
| 输入尺寸 | 640×640 |

### 学习曲线

| Epoch | mAP50 | mAP50-95 | P | R |
|-------|-------|----------|---|---|
| 83 | 0.9620 | 0.6316 | 0.951 | 0.950 |
| 86 | 0.9620 | 0.6456 | 0.948 | 0.953 |
| **118** | **0.9568** | **0.6551** | **0.957** | **0.953** |
| 116 | 0.9566 | 0.6536 | 0.959 | 0.952 |
| 136 | 0.9527 | 0.6518 | 0.950 | 0.948 |

### 分析
- **Recall 0.9532 为所有运行中最高** — 融合模块结合空域和频域特征，减少了漏检
- mAP50-95 0.6551 同样最优，说明定位精度也有改善
- Precision 0.9569 介于空域和频域之间，未出现 trade-off 恶化
- Epoch 83 达到最高 mAP50 (0.962) 但 mAP50-95 较低，最终 best.pt 由 mAP50-95 决定

### Wandb 监控指标
在 wandb 中可以查看每个 val epoch 的融合监控数据：
- `fusion/residual_alpha`: 残差连接权重 (0→0.5 warmup)
- `fusion/output_channel_{0,1,2}_mean` 和 `_std`: 输出通道统计
- `fusion/mse`: 输入输出 MSE
- `fusion/grad_ratio`: 融合参数梯度范数 / 骨干网络梯度范数

---

## 7. 融合架构详解

```
SpatialFrequencyFusion
──────────────────────
输入: 3通道 [spatial, freq, orig_gray]  (经 BGR→RGB 后)

┌─ Conv2d(3→16, k=3, s=1, p=1) + BN + SiLU
├─ Conv2d(16→16, k=3, s=1, p=1) + BN + SiLU
├─ SEBlock(16, reduction=4)
│    ├─ AdaptiveAvgPool → Conv(16→4) → SiLU → Conv(4→16) → Sigmoid
│    └─ Channel-wise multiply
├─ Conv2d(16→3, k=3, s=1, p=1)
└─ + residual_alpha × skip_connection → 输出 (3 通道)
```

### 集成方式
```
YOLO11m model.0:
  Sequential(
    (0): SpatialFrequencyFusion(3→3)   ← 插入的融合模块
    (1): Conv2d(3→64, k=3, s=2, p=1)   ← 原始 YOLO 首层
    ...
  )
```

### 关键设计决策
1. **残差连接 + warmup**: `residual_alpha` 从 0 线性增长至 0.5 (前 10 epoch)，使训练早期 backbone 看近似原始输入
2. **通道注意力 (SE)**: 自适应学习空域/频域特征的通道权重
3. **融合 LR 缩放**: 融合层使用 lr × 10，加速随机初始化参数的收敛
4. **3→3 映射**: 保持通道数不变，对 YOLO 架构透明

---

## 8. 已知问题

### 1. Wandb on_train_end 文件移动错误 (非致命)
```
FileNotFoundError: [WinError 3] 系统找不到指定的路径
```
**原因**: Windows 上 wandb 在 `on_train_end` 回调中移动临时文件时路径长度限制或反斜杠转义问题。
**影响**: 仅影响训练结束时的 PR 曲线日志，不影响训练过程数据和最佳模型保存。
**状态**: 未修复 — Ultralytics wandb 回调中的上游问题。

### 2. Fusion 训练 resume 测试不完整
Commit 7 中修复了 resume 功能，但仅通过代码审查验证，未进行实际的 resume→train 完整测试。

---

## 9. 下一步修改建议

### 短期 (立即执行)

#### S1. 修复 Wandb 文件移动错误
- **描述**: Windows 上 wandb `on_train_end` 中 `_plot_curve` 的 `shutil.move` 失败
- **方案**: 在 `train.py` 的 `_make_wandb_callback` 中增加 `on_train_end` 回调，在 Ultralytics 原生回调之前手动调用 `wandb.finish()`，或 patch 临时目录路径使用短路径
- **优先级**: 低 (非致命)
- **文件**: `train.py`

#### S2. 增加 NSLSR 测试集评估脚本
- **描述**: 目前仅有验证集结果，未在正式测试集 (118 张) 上评估
- **方案**: 编写 `evaluate_test.py`，加载 `best.pt` 在测试集上推理并计算指标
- **优先级**: 中
- **文件**: `scripts/evaluate_test.py` (新建)

#### S3. 输出 ONNX + 推理 demo
- **描述**: 将融合模型导出为 ONNX，编写轻量推理脚本
- **方案**: `predict.py` 已存在，需适配融合预处理流水线
- **优先级**: 中

### 中期 (改进实验)

#### M1. 融合模块架构搜索
- **当前**: hidden_dim=16, 2 层 Conv, SE reduction=4
- **尝试方向**:
  - 增大 hidden_dim (32, 64) 观察表达能力
  - 增加深度 (3-4 层 Conv) 观察是否过拟合
  - 替换 SE 为 CBAM 或简单 1×1 Conv
  - 尝试不同 residual_alpha target (0.3, 0.7, 1.0)
- **命令**:
  ```powershell
  uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-hidden 32 --fusion-lr-scale 10.0 --wandb --epochs 150
  ```

#### M2. 频域方法升级
- **当前**: Butterworth BPF (固定阶数 2, 截止频率 0.007/0.08)
- **尝试方向**:
  - 同态滤波 (Homomorphic filtering) 分离光照/反射分量
  - 小波变换去噪 (Wavelet denoising)
  - 引导滤波 (Guided filter) 边缘保持平滑
- **文件**: `preprocessing_module.py`

#### M3. YOLO 模型升级
- **当前**: YOLO11m (20M params)
- **尝试方向**: YOLO11l (25M) 或 YOLO11x (57M)，观察融合模块在大模型上是否仍有增益
- **风险**: 显存限制 (16GB)，可能需要降低 batch size

#### M4. 多尺度推理
- **描述**: 使用 SAHI (Slice Aided Hyper Inference) 或 TTA (Test Time Augmentation)
- **预期**: 可能提升小目标检测 Recall
- **文件**: `predict.py` 或新建推理脚本

#### M5. 数据增强消融
- **当前**: CLACHE 固定参数
- **尝试方向**: 随机增强参数 (随机 clip_limit, 随机 grid_size)，在训练中引入变化

### 长期 (研究方向)

#### L1. 在小目标场景独立验证
- 目前 512×640 图像中的船舶大多为中等以上尺寸
- 建议在更远距离/更小目标的独立数据集上验证融合模块效果

#### L2. 扩展到 SWIR 波段
- 已有 NSLSR-SWIR 数据，训练配置 `nslsr-swir-v1`
- SWIR 和 LWIR 物理特性不同，融合模块的泛化能力值得验证

#### L3. 时序检测 (视频流)
- 当前为单帧检测
- 可扩展为利用时序信息的检测 (如 YOLO + LSTM 或 3D Conv)
- 但需要视频序列标注数据

#### L4. 模型量化与部署
- ONNX 导出已完成 (76.7MB)
- 可进一步量化为 FP16/INT8 (TensorRT)，减小模型体积，提升推理速度
- 适配 NVIDIA Jetson 等边缘设备

---

## 附录 A: 配置文件

### `configs/train_nslsr_optimized.yaml`
```yaml
task: detect
mode: train
model: yolo11m.pt
epochs: 150
patience: 20
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
close_mosaic: 10
cos_lr: true
amp: true
device: 0
```

### `configs/data_nslsr_lwir.yaml`
```yaml
path: D:/Projects/AUVDect/data/nslsr_lwir
train: images/train
val: images/val
nc: 1
names: ['ship']
```

## 附录 B: Training Commands 速查

```powershell
# Run 1 — 空域
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_tophat_clahe --wandb --epochs 150

# Run 2 — 频域
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_butterworth_clahe --wandb --epochs 150

# Run 3 — 融合 (最终最佳)
uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --fusion-hidden 16 --fusion-lr-scale 10.0 --wandb --epochs 150

# 断点续训 (融合模式)
uv run python train.py --model runs/detect/.../weights/last.pt --resume --use-fusion --wandb

# 仅推理
uv run python predict.py --model runs/detect/.../weights/best.pt
```
