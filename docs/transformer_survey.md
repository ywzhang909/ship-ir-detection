# 轻量级 Transformer 视觉模型综述

> 本文件汇总其他值得关注的轻量 Transformer 及相关架构，作为 EdgeNeXt / FastViT / EfficientFormer / MobileViT / RepViT 等主力候选的补充参考。

## 1. SBCFormer — Single Board Computer Transformer (2023)

- **标题**: SBCFormer: Lightweight Network Capable of Full-size ImageNet Classification at 1 FPS on Single Board Computers
- **DOI**: `10.48550/arXiv.2311.03776`
- **作者**: Xiangyong Lu, M. Suganuma, Takayuki Okatani
- **时间**: 2023
- **Zotero**: `ZKB6M6GR`
- **概要**: 专为单板计算机（树莓派等）设计的轻量 Transformer。通过低秩近似和剪枝策略，在极低计算预算下达到可用精度。
- **参数量**: ~2-4M（具体变体待确认）
- **对项目的价值**: 极低计算需求适合边缘部署，但 SBC 优化可能牺牲检测精度。

## 2. iFormer — Integrating ConvNet and Transformer For Mobile Application (2025)

- **标题**: iFormer: Integrating ConvNet and Transformer For Mobile Application
- **DOI**: — (arXiv 2025)
- **作者**: Chuanyang Zheng
- **时间**: 2025
- **Zotero**: `LHGTN9AD` ✅ 已存在（含 PDF）
- **概要**: 提出卷积-Transformer 深度融合方式，在移动端实现高效的特征提取。
- **参数量**: 待确认（<5M 级别）
- **对项目的价值**: 2025 年最新架构，但成熟度有待验证。

## 3. MoCoViT — Mobile Convolutional Vision Transformer (2022)

- **标题**: MoCoViT: Mobile Convolutional Vision Transformer
- **DOI**: `10.48550/arXiv.2206.05247`
- **作者**: Hailong Ma, Xin Xia, Xing Wang, Xuefeng Xiao, Jiashi Li, Minghang Zheng
- **时间**: 2022
- **Zotero**: `JUVGRHG3`
- **概要**: 移动端卷积视觉 Transformer，通过卷积和自注意力的精细化协作降低计算量。
- **对项目的价值**: 与 MobileViT 同期工作，可作为对照参考。

## 4. Mobile-Former (2022) 

> ⚠️ 未在 Zotero 中找到对应条目，可后续补充导入。

- **标题**: Mobile-Former: Bridging MobileNet and Transformer
- **DOI**: `10.48550/arXiv.2201.02233`
- **作者**: Chunjiang Ge, et al.
- **时间**: 2022 (CVPR 2023?)
- **概要**: 提出双向桥接（Bidirectional Bridge）连接 MobileNet 和 Transformer，实现局部-全局特征的并行计算与交互。
- **参数量**: 508M MAdds 版本约 ~3-4M params
- **对项目的价值**: 双向桥接设计理论上非常适合同时捕获舰船的局部细节和全局上下文。

## 5. MobileMamba — 轻量多感受野视觉 Mamba (2024)

- **标题**: MobileMamba: Lightweight Multi-receptive Visual Mamba Network
- **作者**: Haoyang He, Jiangning Zhang, Yuxuan Cai, et al.
- **时间**: 2024
- **Zotero**: `ZZZ3J2IV`
- **概要**: 使用 State Space Model (SSM/Mamba) 替代传统注意力机制，以线性复杂度实现长距离依赖建模。
- **参数量**: ~3.5M（具体变体待确认）
- **对项目的价值**: SSM 的线性复杂度对高分辨率红外图像处理有潜力；但 YOLO 集成需额外适配工作。

## 优先级排序

| 优先级 | 模型 | 参数量 | 推荐理由 |
|--------|------|--------|----------|
| ★★★ | Mobile-Former | ~3-4M | 双向桥接设计，局部-全局交互好 |
| ★★☆ | iFormer | <5M | 2025 最新, 但成熟待验证 |
| ★★☆ | SBCFormer | ~2-4M | 极致轻量 |
| ★☆☆ | MoCoViT | ~3-5M | 对照参考 |
| ★☆☆ | MobileMamba | ~3.5M | SSM 新范式, 集成成本高 |
