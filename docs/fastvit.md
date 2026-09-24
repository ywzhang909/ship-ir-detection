# FastViT — A Fast Hybrid Vision Transformer using Structural Reparameterization

## 论文信息
- **DOI**: `10.48550/arXiv.2303.14189`
- **标题**: FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization
- **作者**: Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, Anurag Ranjan (Apple)
- **时间**: ICCV 2023
- **Zotero**: `NXG4ZS5K`

## 方法概要
FastViT 是 Apple 推出的混合 Vision Transformer，核心创新是 **RepMixer**：

1. **RepMixer Token Mixer**：使用结构重参数化将 skip-connection 移除，降低内存访问成本
2. **Train-time Over-parameterization**：训练时使用丰富化结构，推理时为简单卷积
3. **大核卷积**：在 FFN 和 patch embedding 中使用大核卷积（7×7）提升鲁棒性
4. **混合架构**：早期 stage 使用卷积捕获局部特征，后期 stage 使用自注意力捕获全局依赖

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | iPhone12 延迟(ms) | 备注 |
|------|--------|-----------|-------|-------------------|------|
| T8   | ~3.1M  | ~0.7      | ~76.0%† | 0.7 | **⭐ 候选** — 轻量级 |
| T12  | ~5.0M  | ~1.2      | ~78.0%† | 1.0 | 中等候选 |
| S12  | 9.5M   | 1.8       | 79.8%  | 1.4 | 超出轻量范围 |

† T 系列精度为近似值（移动端蒸馏训练结果）

## StarNet 对比
- FastViT T8 (~3.1M, ~76%) vs StarNet-S1 (2.9M, 73.5%): **参数量相近但精度高 2.5%**
- **设计哲学差异**：FastViT 是混合 Transformer，StarNet 是纯卷积 + star operation
- FastViT 的 RepMixer + 大核卷积设计使其对小目标的鲁棒性更强（论文在 ImageNet-C 上表现优异）

## 目标检测应用
- FastViT 在 COCO 检测任务上进行了验证，作为骨干网络效果超越同量级 EfficientNet
- **结构重参数化设计使其非常适合与 YOLO 检测头搭配**

## 关键结论
- **FastViT T8 (~3.1M) 是最有希望 <5M 的 Transformer 候选之一**
- RepMixer 的结构重参数化与 MobileOne 一脉相承，对小目标训练可能有益
- 大核卷积的设计对红外舰船检测的杂波抑制有潜在帮助
- **建议优先测试 FastViT T8**：参数量接近 StarNet-S1 但精度预期更高
