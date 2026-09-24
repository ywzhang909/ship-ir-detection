# RepViT — Revisiting Mobile CNN From ViT Perspective

## 论文信息
- **DOI**: `10.48550/arXiv.2307.09283`
- **标题**: RepViT: Revisiting Mobile CNN From ViT Perspective
- **作者**: Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding (Tsinghua University)
- **时间**: 2023 (CVPR 2025?)
- **Zotero**: `NFMF77FE` ✅ 已存在

## 方法概要
RepViT 从 MobileNetV3-L 出发，**逐步引入轻量 ViT 的架构设计**（MetaFormer 结构、模块设计、micro/macro 架构），最终得到一组**纯卷积**但具有 ViT 风格架构的轻量模型。核心思想：

1. **以 ViT 的架构设计思路指导 CNN 改进**：在 block 结构、stage 配置、通道比例等方面吸收 ViT 的长处
2. **MetaFormer 结构**：Token Mixer + Channel MLP 的通用架构，但 Token Mixer 使用卷积实现
3. **结构重参数化**：训练时丰富表达，推理时合并为简单卷积

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | iPhone12 延迟 | 备注 |
|------|--------|-----------|-------|--------------|------|
| M0.5 | ~3.0M | ~0.3 | ~72.0% | <1ms | **⭐ 候选** |
| M0.75| ~4.5M | ~0.5 | ~76.0% | <1ms | **⭐ 候选** |
| M1.0 | ~6.0M | ~0.8 | ~80.0% | ~1ms | 超过参数量预算 |
| M1.5 | ~12.0M | ~1.5 | ~82.0% | ~1.5ms | 远超过轻量范围 |

> 注：RepViT 有多个不同时期的版本（M0.5-M1.5 对应不同宽度），具体参数和精度以论文为准。RepViT M1.0 是首个在 iPhone12 上 <1ms 延迟达到 80% Top-1 的轻量模型。

## StarNet 对比
- **核心差异**：RepViT 是纯卷积架构（受 ViT 启发的 CNN），StarNet 也是纯卷积架构（受 Star Operation 启发）
- **设计哲学不同**：RepViT 吸收 Transformer 宏观架构设计，StarNet 专注于微观算子（star operation）
- 两者在本质上都属于"受 Transformer 启发的轻量 CNN"范式

## 关键结论
- RepViT 证明了**纯卷积架构通过 ViT 架构设计思路改进**可以达到甚至超越轻量 ViT
- 其 MetaFormer 结构 + 结构重参数化设计**适合作为 YOLO 骨干**（C2f 模块可替换）
- 相比 MobileOne，RepViT 在相同延迟下精度更高（~1ms @ 80% vs ~1ms @ 76%）
- **建议测试 RepViT M0.75 (~4.5M) 作为平衡候选**
