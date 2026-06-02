# FocalNet — Focal Modulation Networks

## 论文信息
- **DOI**: `10.48550/arXiv.2203.11926`
- **标题**: "Focal Modulation Networks"
- **作者**: Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao
- **时间**: NeurIPS 2022 (arXiv: 2022-03-22 → NeurIPS 2022)
- **GitHub**: [microsoft/FocalNet](https://github.com/microsoft/FocalNet)
- **HuggingFace**: `microsoft/focalnet-tiny`

## StarNet 关联
FocalNet 是 StarNet 的直接先驱论文之一。StarNet 论文明确将 FocalNet 列为"star operation"的学术先驱，二者共享核心思想：**用 element-wise multiplication（逐元素乘法）替代自注意力**来实现上下文聚合。StarNet 将这一思想从 FocalNet 的 focal modulation 机制推广为通用的 high-dimensional kernel mapping 理论。

## 方法概要
FocalNet 用 **Focal Modulation** 完全替代 Transformer 中的自注意力机制，构建纯 attention-free 的视觉骨干网络。Focal Modulation 由三个组件组成：

1. **Hierarchical Contextualization**（分层上下文编码）：使用堆叠的逐深度卷积层（depth-wise conv），从短到长编码视觉上下文。
2. **Gated Aggregation**（门控聚合）：根据当前 query token 的内容选择性地聚集上下文信息。
3. **Element-wise Modulation**（逐元素调制）：用 affine transformation 将聚合后的上下文注入 query。

核心创新：将自注意力的 **"first interaction, last aggregation" (FILA)** 反转为 **"first aggregation, last interaction" (FALI)**，避免 O(N²) 复杂度。

## 模型规模

| 变体 | Embedding Dim | Hidden Sizes (stages) | Depths | Top-1 (1K) | Top-1 (22K→224) | Top-1 (22K→384) |
|------|--------------|----------------------|--------|------------|-----------------|-----------------|
| **Tiny** | 96 | (192, 384, 768, 768) | (2, 2, 6, 2) | **82.3%** | — | — |
| **Small** | 96 | — | — | 83.0% | — | — |
| **Base** | 96 | — | — | **83.9%** | — | — |
| **Large** | 128 | — | — | — | **87.3%** | — |
| **Huge** | 160 | — | — | — | — | — |

| 变体 | 参数量 | FLOPs |
|------|--------|-------|
| **Tiny** | ~28M | ~4.4G |
| Small | ~47M | ~7.6G |
| Base | ~83M | ~13G |
| Large | — | — |

> ⚠️ **参数量较大**：Tiny 变体 ~28M params，远超 StarNet-S1 (2.9M)。适合作为大型骨干网络替换方案，不适合轻量场景。

## 下游性能 (COCO)

| 任务 | 模型 | 结果 | 对比基线 |
|------|------|------|---------|
| Object Detection (Mask R-CNN) | FocalNet-B (1×) | **49.0 AP** | Swin-B (1×): 46.7, Swin-B (3×): 48.5 |
| Semantic Segmentation (UPerNet) | FocalNet-B | **50.5 mIoU** | Swin-B: 49.7 |
| Panoptic Segmentation | FocalNet-L + Mask2Former | **57.9 PQ** | — |
| Object Detection | FocalNet-H + DINO | **64.4 mAP** | Surpasses Swinv2-G, BEIT-3 |

## 关键结论
- FocalNet 用 focal modulation 完全替代自注意力，在相同计算预算下超越 SoTA Transformer 架构
- FocalNet-Tiny (28M) 参数量较大，**不适合作为本项目的轻量替换目标**
- StarNet 的理论工作（arXiv:2403.19967）从核方法角度重新解释了 FocalNet 的核心操作
- 本项目可借鉴的启示：**Focal Modulation 模式**可作为 attention-free 检测模块的设计参考

## 参考链接
- arXiv: [2203.11926](https://arxiv.org/abs/2203.11926)
- GitHub: [microsoft/FocalNet](https://github.com/microsoft/FocalNet)
- HuggingFace: [microsoft/focalnet-tiny](https://huggingface.co/microsoft/focalnet-tiny)
