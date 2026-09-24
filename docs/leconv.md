# LeConv — Decomposed Convolutional Kernel for Long-Range Vision

## 论文信息
- **DOI**: `10.48550/arXiv.2407.02488`
- **标题**: LeConv: Decomposed Convolutional Kernel for Long-Range Embedded Vision
- **作者**: Yuqi Wang, et al. (Tsinghua)
- **时间**: 2024
- **Zotero**: `5FMNE5L6`
- **Zotero**: `5FMNE5L6`
- **GitHub**: [THU-MIG/LeConv](https://github.com/THU-MIG/LeConv)

## StarNet 关联
LeConv 与 StarNet 共享"轻量化 conv 内核设计"的理念，但走不同路线：LeConv 用**自适应池化 + element-wise scaling** 实现长感受野，StarNet 用**star operation** 实现高维特征映射。

## 方法概要
将标准卷积分解为三步：
1. **Spatial Pooling** — 自适应内核的池化，捕获长距离特征
2. **Local Convolution** — 局部卷积提取细粒度特征
3. **Element-wise Scaling** — 逐元素缩放融合

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 | 备注 |
|------|--------|-----------|-------|------|
| T | **0.9M** | 0.18 | 74.0% | **⭐ 超强候选** — 极轻量 |
| S | **2.9M** | 0.66 | 78.9% | **⭐ 候选** — 与 StarNet-S1 同规模 |
| B | 8.7M | 2.40 | 82.7% | 中等候选 |

## StarNet 对比
- **LeConv-T (0.9M, 74.0%) vs StarNet-S1 (2.9M, 73.5%)**: LeConv-T 用仅 1/3 参数就达到相近准确率和更高 Top-1！**这是非常值得测试的候选**
- LeConv-S (2.9M, 78.9%) vs StarNet-S1 (2.9M, 73.5%): 同参数量下 LeConv-S 高 5.4%！
- **核心差异**：LeConv 用空间池化捕获长感受野，StarNet 用 star operation 映射高维空间
- 两者结合（LeConv 骨干 + StarNet blocks）可能是未来的研究方向

## 关键结论
- LeConv-T 是本项目最轻量级的候选（0.9M params），精度甚至超过 StarNet-S1
- LeConv-S 与 StarNet-S1 同规模但精度显著更高
- **强烈建议**将 LeConv-T 和 LeConv-S 加入实验计划
