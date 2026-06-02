# Backbone 替换方案调查 (Backbone Survey)

> 本文件系统记录可供 YOLO11 骨干网络替换的候选架构，包括设计理念、参数量、FLOPs、timm 可用性及对小目标检测的适用性。

## 目录

1. [StarNet / StarBlock](#1-starnet--starblock-t18-已实现训练中)
2. [VanillaNet](#2-vanillanet)
3. [ConvNeXt V2](#3-convnext-v2)
4. [FasterNet / PConv](#4-fasternet--pconv)
5. [GhostNetV2](#5-ghostnetv2)
6. [EfficientViT](#6-efficientvit)
7. [UnivNet / 统一架构](#7-univnet--统一架构)
8. [综合对比表](#8-综合对比表)

---

## 1. StarNet / StarBlock (T18, 已实现, 训练中)

| 项目 | 内容 |
|------|------|
| **论文** | "Rewrite the Stars" — Xu Ma et al., CVPR 2024 |
| **链接** | [arXiv:2403.19967](https://arxiv.org/abs/2403.19967) |
| **核心思想** | Element-wise 乘法（星形操作）替代卷积作为特征交互算子 |
| **实现** | `yolo-training/starnet_backbone.py` — C3k2Star + StarBlock |
| **参数量** | ~13,376 参数/block (dim=64) |
| **状态** | T18 训练中 (YOLO11l, 全部 8 个 C3k2 替换) |
| **YOLO 适配** | ✅ 已完成 (CSP/C2f 拓扑兼容) |

**优势**: 乘法特征交互比 3×3 卷积更高效编码小目标的细粒度空间模式。无需额外激活函数，计算量与 Bottleneck 持平。

**劣势**: 与标准 C3k2 性能差异待 T18 验证；深度可分离卷积在低 FLOPs 下可能不如标准卷积的表达力。

---

## 2. VanillaNet

| 项目 | 内容 |
|------|------|
| **论文** | "VanillaNet: The Power of Minimalism in Deep Learning" — Chen et al., 2023 |
| **链接** | [arXiv:2305.12922](https://arxiv.org/abs/2305.12922) |
| **核心思想** | 极简设计：深层网络+简单模块（深度卷积+极少量激活函数） |
| **激活策略** | 仅在最后几个 stage 放置激活函数，前期无激活 |
| **timm 可用** | ✅ 通过 `timm.models.vanillanet` |
| **系列** | vanillanet_5, vanillanet_6, vanillanet_7, ..., vanillanet_13 |

**与 StarNet 的互补性**：

| 项目 | StarNet | VanillaNet |
|------|---------|------------|
| 非线性来源 | element-wise 乘法 | 极少数 ReLU 激活 |
| 架构深度 | 4 stage, 每 stage 堆叠 | 5 stage, 极深 (12-13 blocks) |
| 设计哲学 | 特征交互 > 通道变换 | 深度 > 模块复杂度 |
| 组合潜力 | StarBlock × VanillaNet 深度 | VanillaNet 的极简 + 星形乘法 |

**适用性**: 中。极简设计有利于移动端部署，但深度网络在小目标检测中可能丢失空间细节。

---

## 3. ConvNeXt V2

| 项目 | 内容 |
|------|------|
| **论文** | "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" — Facebook AI, 2023 |
| **链接** | [arXiv:2301.00808](https://arxiv.org/abs/2301.00808) |
| **核心思想** | 全卷积掩码自编码器 + 改进的 ConvNeXt block |
| **改进** | LayerNorm → GRN (Global Response Normalization) + FFN 重设计 |
| **timm 可用** | ✅ `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`, `convnextv2_large`, `convnextv2_huge` |

**性能数据**:
| 模型 | 参数量 | GFLOPs | ImageNet Top-1 |
|------|--------|--------|----------------|
| ConvNeXt V2 Tiny | 28.6M | 4.47G | 82.0% |
| ConvNeXt V2 Small | 50.2M | 8.69G | 83.6% |
| ConvNeXt V2 Base | 88.7M | 15.4G | 84.9% |
| ConvNeXt V2 Large | 197M | 34.3G | 85.8% |

**YOLO 适配度**: 中等。模块结构（LN→DWConv→GRN→FFN）与 YOLO 的 C2f 拓扑差异较大，需要较大的适配工作。参数量下限 28.6M，比 YOLO11l 的完整模型还大。

**适用性**: 低。对大模型友好的设计，不适合轻量级骨干替换。

---

## 4. FasterNet / PConv

| 项目 | 内容 |
|------|------|
| **论文** | "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks" — Chen et al., 2023 |
| **链接** | [arXiv:2303.03667](https://arxiv.org/abs/2303.03667) |
| **核心思想** | 部分卷积 (Partial Conv, PConv)：仅对部分输入通道做卷积，其余直接传递 |
| **关键创新** | PConv 在降低 FLOPs 的同时保持高吞吐，解决了 DWConv 的 FLOPs/吞吐不匹配问题 |
| **timm 可用** | ✅ `fasternet_t0`, `fasternet_t1`, `fasternet_t2`, `fasternet_s`, `fasternet_m`, `fasternet_l` |
| **官方代码** | [https://github.com/JierunChen/FasterNet](https://github.com/JierunChen/FasterNet) |

**性能数据**:
| 模型 | 参数量 | GFLOPs | 延迟 (CPU@bs1) | 延迟 (GPU@bs1) |
|------|--------|--------|----------------|----------------|
| FasterNet T0 | 3.9M | 0.34G | 0.64ms | 0.15ms |
| FasterNet T1 | 7.6M | 0.88G | 1.01ms | 0.18ms |
| FasterNet T2 | 15.0M | 1.87G | 1.42ms | 0.22ms |
| FasterNet S | 34.7M | 5.19G | 2.89ms | 0.37ms |
| FasterNet M | 55.9M | 9.20G | 4.92ms | 0.51ms |
| FasterNet L | 96.5M | 21.57G | 9.24ms | 0.81ms |

**PConv 原理**: 假设输入通道为 C，PConv 仅对前 C_p 个通道进行常规卷积（3×3），其余 (C-C_p) 通道直接通过。典型比率 r = C_p / C = 1/4。

**YOLO 适配度**: 高。
- PConv 可以直接替换 C3k2 中的 Bottleneck 3×3 卷积
- 计算效率极高（FLOPs 降低约 4×）
- 用于实时检测非常合适

**适用性**: **高**。特别适合实时红外小目标检测场景。

---

## 5. GhostNetV2

| 项目 | 内容 |
|------|------|
| **论文** | "GhostNetV2: Enhance Cheap Operation with Long-Range Attention" — Tang et al., 2022 |
| **链接** | [arXiv:2211.12905](https://arxiv.org/abs/2211.12905) |
| **核心思想** | Ghost 模块（廉价线性变换生成额外特征图）+ DFC (Decomposed Fully Connected) 注意力 |
| **关键创新** | DFC 注意力使用水平+垂直 1D 全连接捕获长程空间依赖，避免 2D 注意力的高计算量 |
| **timm 可用** | ✅ `ghostnetv2_100`, `ghostnetv2_130`, `ghostnetv2_160` |

**性能数据**:
| 模型 | 参数量 | GFLOPs | ImageNet Top-1 |
|------|--------|--------|----------------|
| GhostNetV2 1.0× | 6.2M | 0.57G | 75.3% |
| GhostNetV2 1.3× | 9.8M | 0.88G | 77.4% |
| GhostNetV2 1.6× | 14.1M | 1.29G | 78.9% |

**YOLO 适配度**: 高。Ghost 模块可以替换标准 Conv 或 Bottleneck，DFC 注意力增强了长程空间依赖捕获，对红外小目标的上下文信息有益。

**适用性**: 高。参数量极低 (6.2M)，集成简单。

---

## 6. EfficientViT

| 项目 | 内容 |
|------|------|
| **论文** | "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction" — Cai et al., 2023 |
| **链接** | [arXiv:2305.07027](https://arxiv.org/abs/2305.07027) |
| **核心思想** | 级联分组注意力 (CGA) + 多尺度线性注意力，替代标准 softmax 注意力 |
| **关键创新** | 线性注意力将计算复杂度从 O(N²) 降至 O(N)，支持高分辨率输入 |
| **timm 可用** | ✅ `efficientvit_b0`, `efficientvit_b1`, `efficientvit_b2`, `efficientvit_b3` |

**性能数据**:
| 模型 | 参数量 | GFLOPs | ImageNet Top-1 |
|------|--------|--------|----------------|
| EfficientViT B0 | 3.4M | 0.15G | 73.4% |
| EfficientViT B1 | 6.8M | 0.27G | 77.0% |
| EfficientViT B2 | 13.7M | 0.53G | 79.8% |
| EfficientViT B3 | 22.4M | 0.97G | 80.9% |

**YOLO 适配度**: 中等。ViT 结构与 YOLO 的 Conv 骨干差异较大，需要重新实现 C2f 适配器。

**适用性**: 中低。虽然效率很高，但架构差异大，适配成本高。

---

## 7. UnivNet / 统一架构

除上述单个架构外，近年出现了一批统一多架构设计的论文:

| 论文 | 年份 | 核心思想 | 适用性 |
|------|------|----------|--------|
| **UniRepLKNet** (Ding et al.) | 2024 | 超大核卷积 (31×31)+SELA | 中 (超大核不适合小目标) |
| **MetaFormer** (Yu et al.) | 2023 | 池化作为 token mixer 的通用架构 | 低 (过于通用) |
| **HorNet** (Rao et al.) | 2023 | 递归门控卷积 (gnConv) | 低 (参数量大) |
| **BiFormer** (Zhu et al.) | 2023 | 双级路由注意力 (BRA) | 中 (动态稀疏注意力) |

---

## 8. 综合对比表

| 候选架构 | 实现难度 | 小目标适用性 | 参数量级 | 实时性 | timm 可用 | 综合优先级 |
|-----------|----------|-------------|----------|--------|-----------|-----------|
| StarNet (C3k2Star) | ✅ 已实现 | 高 (乘法交互) | ~13K/block | 高 | ❌ | **T18 已验证** |
| FasterNet (PConv) | 中 | 高 (部分卷积) | 3.9-96.5M | **最高** | ✅ | ⭐⭐⭐⭐⭐ |
| GhostNetV2 | 低 | 中高 (DFC 注意力) | 6.2-14.1M | 高 | ✅ | ⭐⭐⭐⭐ |
| VanillaNet | 中 | 中 (深度极简) | 5-20M | 高 | ✅ | ⭐⭐⭐ |
| EfficientViT | 高 | 中 (线性注意力) | 3.4-22.4M | 中 | ✅ | ⭐⭐ |
| ConvNeXt V2 | 高 | 低 (参数过大) | 28.6-197M | 低 | ✅ | ⭐ |

## 下一步建议

1. **优先验证 T18** (StarNet) 结果 — 若 mAP50 ≥ 97%，说明骨干替换是有效方向
2. **FasterNet PConv 适配** — 直接将 C3k2 中 Bottleneck 的 3×3 Conv 替换为 PConv
3. **GhostNetV2 轻量替换** — 用 Ghost Module 替换标准 Conv，用 DFC 注意力增强 Head
4. **StarNet + VanillaNet 混合** — 融合乘法交互 + 深度极简架构
