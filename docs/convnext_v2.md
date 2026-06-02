# ConvNeXt V2 — Modern ConvNet with Masked Autoencoders

## 论文信息
- **DOI**: `10.48550/arXiv.2301.00808`
- **标题**: ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
- **作者**: Sanghyun Woo, et al. (Meta AI)
- **时间**: CVPR 2023
- **Zotero**: `FVUG9X64`
- **timm**: `convnextv2_*`

## StarNet 关联
ConvNeXt V2 代表"现代化卷积网络"的路线，用 LayerNorm + GELU + 大核卷积重构 ConvNet。StarNet 则从操作层面（star operation）重新思考卷积，两条路线正交。

## 方法概要
- FCMAE (Fully Convolutional Masked Autoencoder) 预训练
- GRN (Global Response Normalization) 层
- 现代 CNN 设计：LayerNorm, GELU activation, large kernel depthwise conv

## 模型规模
| 变体 | Params | FLOPs (G) | Top-1 |
|------|--------|-----------|-------|
| Atto | 3.7M | 0.55 | 70.5% |
| Femto | 5.2M | 0.78 | 74.2% |
| Pico | 9.1M | 1.37 | 76.1% |
| Nano | 15.6M | 2.43 | 79.5% |
| Base | 88.5M | 15.45 | 85.0% |

## StarNet 对比
- V2-Atto (3.7M, 70.5%) vs StarNet-S1 (2.9M, 73.5%): StarNet 以更少参数取得更高精度
- V2-Femto (5.2M, 74.2%): 精度超过 StarNet-S1 但参数量大 80%
- **建议**：V2-Atto 规模接近但精度更低，V2-Femto 精度更优但代价更大

## 关键结论
- ConvNeXt V2-Atto 参数量 3.7M，与 StarNet-S1 接近
- GRN 层和 MAE 预训练是特色，但需要大量预训练资源
- 对于本项目（16GB VRAM, 有限数据），ConvNeXt V2 可能需要从 timm 加载预训练权重
