# RT-DETR: Real-Time DEtection TRansformer

> **论文**: DOI:10.48550/arXiv.2304.08069, "RT-DETR: A DETR-based Real-Time Object Detector", S. Li et al., arXiv 2023

## 摘要

RT-DETR（Real-Time DEtection TRansformer）由 S. Li 等人于 2023 年提出，旨在弥合基于 Transformer 的端到端目标检测器与实时检测需求之间的鸿沟。该方法将 DETR（Deformable DETR）的编码器-解码器架构与 YOLO 风格的多尺度级联特征金字塔相结合，实现了兼具 Transformer 检测精度与实时推理速度的统一框架。RT-DETR 的核心设计包括：一个辅助分类分支以加速训练收敛、一个混合编码器（Hybrid Encoder）采用可变形注意力（Deformable Attention）进行多尺度特征交互、以及一个辅助解码器头（Auxiliary Decoder Head）以加速匹配与收敛。与 DETR 系列模型不同，RT-DETR 摒弃了传统的锚框设计和 NMS 后处理，通过 Transformer 解码器直接输出检测结果，实现了真正的端到端检测。实验结果表明，RT-DETR 在 COCO 数据集上以接近 YOLO 系列的速度达到了 Transformer 级别的高精度。

## 红外舰艇检测中的应用

在红外舰艇检测研究中，RT-DETR 被规划为未来探索方向（T21 实验），旨在替换当前 YOLO 系列模型中的检测头结构，以 Transformer 解码器替代传统的卷积检测头。这一替换需要改变模型的基线架构——从 YOLO11 过渡至 RT-DETR 框架。RT-DETR 的端到端设计可消除 NMS 后处理环节，避免红外小目标在重叠检测中被误抑制的问题。此外，其混合编码器中的可变形注意力机制能够自适应地关注红外舰艇目标的稀疏特征区域，理论上对复杂海面背景下的低信噪比目标检测具有天然优势。然而，RT-DETR 的 Transformer 解码器结构在高分辨率红外图像上的计算效率仍待进一步验证，特别是在显存受限的移动端推理平台上。
