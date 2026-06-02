# Star-YOLO — Lightweight Detail-Aware IR Ship Detection

## 论文信息
- **DOI**: `10.1364/AO.580648`
- **标题**: Star-YOLO: A lightweight and detail-aware network for infrared ship detection
- **作者**: (待补充)
- **时间**: 2026, Applied Optics
- **Zotero**: `JXQJH5MV`
- **PDF**: ❌ 无 PDF 附件

## 与本项目直接相关
这是**与我们项目完全同领域**的论文 —— 用 StarNet 作为 YOLOv11 的 backbone 进行红外舰船检测。

## 方法概要
- 使用 StarNet 作为骨干网络替换 YOLOv11 的原始 backbone
- 加入 detail-aware 模块增强小目标特征
- 针对红外舰船检测任务优化网络结构

## 关键借鉴点
1. **骨干替换方案** — Star-YOLO 证明了 StarNet 在红外舰船检测领域的可行性，为我们 T18 实验提供理论支持
2. **轻量设计** — 面向边缘部署的轻量方案
3. **小目标优化** — detail-aware 模块可作为我们 Neck 层面的改进方向

> 本文与我们的 T18 "骨干网络替换 (StarNet)" 实验高度同向，值得深入对比实验设置。
