# MobileOne — One Millisecond Mobile Backbone

## 论文信息
- **DOI**: `10.48550/arXiv.2206.04040`
- **标题**: MobileOne: An Improved One millisecond Mobile Backbone
- **作者**: Xinchao Wang, et al. (Apple)
- **时间**: CVPR 2023
- **GitHub**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)

## 方法概要
提出 Multi-purpose Input-Wise Transform (MWT)，通过 train-time over-parameterization + inference-time reparameterization，实现训练时丰富的表达能力和推理时的极致效率。

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | 备注 |
|------|--------|-----------|-------|------|
| S0 | **2.1M** | 0.57 | 71.4% | **⭐ 候选** — 低于 StarNet-S1 |
| S1 | 4.8M | 0.85 | 75.9% | **⭐ 候选** |
| S2 | 7.8M | 1.35 | 77.4% | 中等候选 |
| S3 | 10.1M | 2.08 | 78.1% | 超出轻量级 |
| S4 | 14.8M | 2.46 | 79.4% | — |

## StarNet 对比
- S0 (2.1M, 71.4%) vs StarNet-S1 (2.9M, 73.5%): MobileOne-S0 更小但准确率低 2.1%
- **Train-time over-parameterization 的优势**：MobileOne 在训练时使用丰富的参数空间，推理时简化。这对小目标特征学习可能有额外帮助
- **建议**：S0 和 S1 都适合作为 StarNet-S1 的轻量对照实验

## 关键结论
- Over-parameterized training 对红外小目标可能有益（更强的特征学习容量）
- S0 参数量最小，S1 性能/效率平衡最优
- Reparameterization 技术可直接集成到 YOLO11 C3k2 模块中
