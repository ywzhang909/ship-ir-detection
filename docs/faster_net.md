# FasterNet — Partial Convolution for Higher FLOPS

## 论文信息
- **DOI**: `10.48550/arXiv.2303.03667`
- **标题**: Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
- **作者**: Jierun Chen, et al.
- **时间**: CVPR 2023
- **Zotero**: `CMVYX5JX`
- **GitHub**: [JierunChen/FasterNet](https://github.com/JierunChen/FasterNet)

## 方法概要
核心发现：**低 FLOPs ≠ 快**。提出 Partial Convolution (PConv)，将标准 conv 拆分为 "global conv" + "local conv"，通过减少实际运算量但保持高 FLOP 计数来实现更快的 GPU 推理速度。

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | 备注 |
|------|--------|-----------|-------|------|
| T0 | 3.5M | 0.47 | 73.2% | **⭐ 候选** — 略大于 StarNet-S1 |
| T1 | 5.9M | 0.87 | 76.2% | 中等候选 |
| T2 | 9.1M | 1.30 | 78.0% | 超出轻量范围 |
| S | 14.0M | 1.80 | 79.2% | — |
| M | 25.5M | 3.10 | 81.0% | — |

## StarNet 对比
- FasterNet-T0 (3.5M) vs StarNet-S1 (2.9M): Top-1 73.2% vs 73.5%，StarNet 略优
- 核心差异：FasterNet 追求高 FLOPS + GPU 并行度，StarNet 追求低参数量 + star operation
- **建议**：FasterNet-T0 可作为 StarNet-S1 的对照候选测试

## 关键结论
- Partial Convolution 是提升 GPU 推理效率的 novel 设计
- T0 变体规模接近 StarNet-S1，适合在 YOLO11 骨干中做对比实验
