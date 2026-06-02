# StarNet / StarBlock Backbone

> **论文**: "Rewrite the Stars" (Xu Ma et al., CVPR 2024) — [arXiv:2403.19967](https://arxiv.org/abs/2403.19967)

## StarBlock 设计原理

StarBlock 的核心创新是 **星形操作 (Star Operation)**：两个并行 1×1 投影的逐元素乘法（element-wise multiplication），在无需传统激活函数的情况下产生非线性特征交互。

### 标准 StarBlock（论文原始设计）

```
输入 x
  ├── DWConv(3×3, c_in) → 两个 1×1 FC 分支 → element-wise multiply
  └── 残差连接
```

论文中的 StarNet 使用 4 阶段层次化架构，每个 Stage 由若干 StarBlock 堆叠构成，激活函数为 ReLU6，无 BatchNorm。

### YOLO 适配版 C3k2Star（本项目的实现）

由于 YOLO 的 C3k2 使用 C2f 拓扑，StarBlock 被嵌入 CSP 结构中：

```
cv1(1×1) → chunk(2) ─┬─ y0 ──────────────────────┐
                       └─ y1 → StarBlock × n ──────┤
                                                    ↓
                                             cv2(cat[...])
```

每个 **StarBlock** 内部：

```
         ┌── cv1(1×1) ──┐
x ───────┤              ├──→ [×] element-wise → DWConv(3×3) → cv3(1×1) → +x
         └── cv2(1×1) ──┘
```

**参数**: 13,376 参数 (dim=64)，与标准 Bottleneck 基本持平。

### 关键区别：StarNet vs VanillaNet

| 特性 | StarNet (Ma 2024) | VanillaNet (Chen 2023) |
|------|--------------------|------------------------|
| 核心算子 | Element-wise 乘法 | 深度卷积 + 极简激活 |
| 非线性来源 | 乘法交互 | 激活函数 (ReLU) |
| 设计哲学 | 特征交互 > 激活 | 极简架构 > 深度 |

## 实验结果 (T18)

- **状态**: 训练中
- **Backbone**: YOLO11l 全部 8 个 C3k2 → C3k2Star
- **假设**: StarBlock 的 element-wise 乘法比 3×3 卷积更适合红外小目标的特征编码

## 参考实现

- `yolo-training/starnet_backbone.py` — StarBlock + C3k2Star + 替换函数
- `yolo-training/train.py` — `--backbone starnet` 参数入口
