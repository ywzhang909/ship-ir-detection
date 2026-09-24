# EdgeNeXt — Efficiently Amalgamated CNN-Transformer Architecture

## 论文信息
- **DOI**: `10.48550/arXiv.2206.10589`
- **标题**: EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications
- **作者**: Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Rao Muhammad Anwer, Fahad Shahbaz Khan
- **时间**: ECCV 2022 (Oral)
- **Zotero**: `EWL32V8D`

## 方法概要
EdgeNeXt 提出 **SDTA (Split Depth-wise Transpose Attention)** 编码器，将输入张量沿通道维度分组，并在每组内依次应用深度可分离卷积 + 通道维自注意力（transpose attention），实现线性复杂度的全局感受野。核心设计：

1. **SDTA 模块**：分组 → DConv 局部编码 → 通道维 Self-Attention 全局建模
2. **Transpose Attention**：在通道维度上计算注意力（而非空间维度），复杂度从 O(N²d) 降至 O(Nd²)，实现在移动设备上的高效推理
3. **多尺度特征学习**：前一组输出叠加到当前组，自适应学习多尺度表征

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | 备注 |
|------|--------|-----------|-------|------|
| XXS  | **1.3M** | 0.26 | 71.2% | **⭐ 超轻量候选** |
| XS   | **2.3M** | 0.54 | 74.9% | **⭐ 候选 — 轻量级** |
| S    | 5.6M | 1.26 | 79.4% | 效率最优, 但 >5M |
| B    | 18.5M | 3.84 | 82.5% | 超出轻量级范围 |

## StarNet 对比
- EdgeNeXt XXS (1.3M, 71.2%) vs StarNet-S1 (2.9M, 73.5%): **EdgeNeXt 用 45% 的参数达到接近的精度**
- EdgeNeXt XS (2.3M, 74.9%) vs StarNet-S1 (2.9M, 73.5%): **更少参数 + 更高 Top-1**
- **核心差异**：EdgeNeXt 利用 SDTA + Transpose Attention 实现全局建模，StarNet 通过 star operation 实现高维映射

## 目标检测应用
- EdgeNeXt-S 作为 SSDLite 骨干在 COCO 上达到 27.9 mAP，比 MobileViT-S 少 38% MAdds (2.1G vs 3.4G)
- 在 NVIDIA Jetson Nano 上，XXS 仅需 19.3ms 推理时间

## 关键结论
- **EdgeNeXt XXS (1.3M)** 和 **XS (2.3M)** 是 <3M 参数范围的最强 Transformer 候选之一
- Channel-wise Transpose Attention 设计天然适合红外小目标（感受野灵活）
- 与 StarNet-S1 同精度下参数节省 50%+
- **建议优先测试 EdgeNeXt XS (2.3M)**：平衡参数量和精度
