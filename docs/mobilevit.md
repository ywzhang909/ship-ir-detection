# MobileViT — Light-weight, General-purpose, and Mobile-friendly Vision Transformer

## 论文信息
- **DOI**: `10.48550/arXiv.2110.02178`
- **标题**: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (ICLR 2022)
- **标题 (V2)**: Separable Self-attention for Mobile Vision Transformers (2022)
- **标题 (V3)**: MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local and Global Contexts (2023)
- **作者**: Sachin Mehta, Mohammad Rastegari (Apple)
- **时间**: ICLR 2022 (V1) / 2022 (V2) / 2023 (V3)
- **Zotero**: `ST5I6PT8`

## 方法概要
MobileViT 首次将 Transformer 以"卷积形式"引入移动端骨干网络：

1. **MobileViT Block**：将标准卷积替换为 → 3×3 Depthwise Conv → 1×1 Conv(升维) → 展开为 patches → MHSA → 折叠回特征图 → 1×1 Conv(降维)
2. **Transformer 作为卷积**：将自注意力看作全局卷积核——在不同的空间位置学习不同的聚合权重
3. 基础架构：MobileNetV2-style 的 inverted residual blocks + MobileViT blocks

### V2 改进
- **可分离自注意力 (Separable Self-Attention)**：降低 O(N²) 复杂度
- 更高效的融合策略

### V3 改进
- **简单有效的局部-全局融合**：改进早期 stage 的局部和全局特征融合
- 进一步降低推理延迟

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | 来源 |
|------|--------|-----------|-------|------|
| XXS  | ~1.3M | 0.36 | ~69.0% | V1 |
| XS   | ~2.3M | 0.67 | ~72.0% | V1 |
| S    | 5.9M  | 2.01 | 78.4%  | V1 |
| V2-S0| ~1.6M | —    | ~70.0% | V2 |
| V2-S1| ~3.0M | —    | ~74.0% | V2 |
| V2-S2| ~5.0M | —    | ~78.0% | V2 |

## StarNet 对比
- MobileViT XS (2.3M, 72%) vs StarNet-S1 (2.9M, 73.5%): 参数量相近但精度低 1.5%
- **核心设计差异**：MobileViT 是 CNN+Transformer 混合，StarNet 是纯卷积 + star operation
- MobileViT 的优势在于检测任务上已验证有效（COCO 检测比 MobileNetv3 +5.7% mAP）

## 关键结论
- MobileViT 是**最早且最成熟**的移动端 ViT 架构，生态完整（timm/HuggingFace 支持）
- 在 COCO 检测任务上表现突出，说明**混合 CNN-Transformer 架构对小目标检测有增益**
- V2/V3 的改进版本具有更好的效率/精度比
- **建议测试 MobileViT V2 S1 (~3.0M, ~74%)**：参数量与 StarNet-S1 接近，但架构本质不同
