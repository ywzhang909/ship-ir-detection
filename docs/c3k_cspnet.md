# C3K / CSPNet

> **论文**: DOI:10.48550/arXiv.2207.04670, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors", C.-Y. Wang et al., arXiv 2022
> 
> CSPNet: DOI:10.48550/arXiv.1911.11929, "CSPNet: A New Backbone that can Enhance Learning Capability of CNN", C.-Y. Wang et al., CVPR 2020

### 摘要

CSPNet (Cross Stage Partial Network) 的核心思想是：在每一阶段将特征图沿着通道维度分成两部分，一部分经过密集的 Bottleneck 操作，另一部分作为快捷连接直接保留，最后将两部分拼接起来。这种梯度聚合策略有效解决了深度网络中梯度信息重复计算的问题。在标准的残差网络中，大量梯度信息在不同层之间被重复使用，导致计算效率低下；CSPNet 通过特征分割和跨阶段拼接，使得梯度流可以沿着两条不同的路径传播，从而在减少计算量的同时保持甚至提升网络的表征能力。

C3K 是基于 CSPNet 拓扑结构的具体实现，使用 3×3 卷积构建 Bottleneck 模块。其具体结构为：cv1 对输入进行通道变换，然后送入由 3×3 卷积组成的 Bottleneck 序列处理，处理结果与来自 cv1 的快捷路径拼接，最后通过 cv3 进行通道融合输出。C3k2 是 YOLO11 中的变体版本，对内部结构进行了微调以适应 YOLO11 的整体架构设计。

CSP 风格的设计在现代目标检测网络中得到了广泛应用，YOLOv4、YOLOv5、YOLOv7、YOLOv8 乃至 YOLO11 均采用了类似的跨阶段特征聚合结构。其成功主要归因于：通过特征分割减少了冗余梯度计算，通过跨阶段拼接维持了丰富的梯度流信息，以及通过端到端训练实现了特征的高效复用。

### 红外舰艇检测中的应用

在本项目中，实验 T12 将 YOLO11l 中所有的 8 个 C3k2 模块替换为 C3k 模块（卷积核大小 k=3），旨在评估不同的 CSP 变体对红外舰艇检测性能的影响。

实验结果显示 mAP50 从基线提升至 97.77%，但参数量从 25.4M 增加至 28.9M（增幅 13.8%）。这表明 C3k 提供了更强的特征提取能力，但代价是显著的计算开销增加。mAP50 的提升说明 3×3 卷积的 Bottleneck 在特征提取方面确实优于 C3k2 的默认配置；然而 mAP50-95 的下降则提示，C3k 可能对中等难度的检测样本产生了过拟合效应，导致其在更严格的 IoU 阈值下表现欠佳。参数量的显著增加也是实际部署中需要权衡的因素，特别是在资源受限的边缘计算场景中。
