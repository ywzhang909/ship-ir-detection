# EfficientFormer — Vision Transformers at MobileNet Speed

## 论文信息
- **DOI**: `10.48550/arXiv.2206.01191`
- **标题**: EfficientFormer: Vision Transformers at MobileNet Speed (NeurIPS 2022)
- **标题 (V2)**: Rethinking Vision Transformers for MobileNet Size and Speed (ICCV 2023)
- **作者**: Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren (Snap Inc.)
- **时间**: NeurIPS 2022 (V1) / ICCV 2023 (V2)
- **Zotero V1**: `SFURNZPR`
- **Zotero V2**: `GX3T6A7W`

## 方法概要 (V1)
EfficientFormer 重新审视 ViT 中低效的算子设计（如非必要的 reshape、冗余的 MHSA），提出 **维度一致（dimension-consistent）纯 Transformer** 架构，避免昂贵的维度变换。通过延迟驱动的网络剪枝，获得一系列适合移动设备的模型。

## 方法概要 (V2)
EfficientFormerV2 进一步引入：
1. **Unified FFN**：将早期 stage 的 PoolFormer 替换为统一 FFN 块，改善信息流动
2. **细粒度联合搜索**：同时优化延迟和参数量，搜索高效微架构
3. **改进的注意力机制**：低延迟注意力

## 模型规模
### V2 变体
| 变体 | Params | FLOPs (G) | Top-1 | 备注 |
|------|--------|-----------|-------|------|
| S0  | **3.6M** | 0.40 | 74.0%† | **⭐ 候选** — 参数量略高于 StarNet-S1 |
| S1  | ~5.0M | 0.60 | ~76.0%† | 中等候选 |
| S2  | ~6.3M | 0.80 | ~77.5%† | 略超轻量范围 |

† 蒸馏训练结果，无蒸馏精度约低 1-1.5%

### V1 变体（参数量较大）
| 变体 | Params | FLOPs (G) | Top-1 |
|------|--------|-----------|-------|
| L1 | 12.2M | 1.30 | 79.2% |
| L3 | 31.0M | 2.57 | 81.4% |
| L7 | 51.0M | 4.85 | 83.3% |

## StarNet 对比
- **EfficientFormerV2 S0 (3.6M, 74.0%) vs StarNet-S1 (2.9M, 73.5%)**：参数量多 24%，精度相近
- **核心差异**：Transformer 自注意力 vs StarNet 的 star operation
- V2 S0 虽然略大，但纯 Transformer 架构为检测头提供了全局感受野

## 关键结论
- V2 S0 (3.6M) 是唯一参数量接近 StarNet-S1 的纯 Transformer 候选
- **V1 的 L 系列参数量过大（>12M）**，不适合轻量级骨干测试
- 检测任务适配需注意 Transformer 特征金字塔与 YOLO 检测头的对齐
- 建议作为对照实验，验证"纯 Transformer 骨干在红外小目标检测上能否达到 CNN 骨干同等精度"
