# 背景建模与海杂波抑制 — 参考文献汇编
## 供插入《红外海面舰艇识别深度模型改进方法综述》

> 最终版本 | 2026-05-28 | 所有 DOI 已验证
> 共 18 篇论文，按方法子类分组

---

## A. 综述与方法论调研

**1.** Ma, D., Dong, L., Xu, W., & Zhang, Y. (2023). Recent advancements in long-distance marine infrared target detection: Latest method and future perspectives. *Infrared Physics & Technology*, 133, 104729.
DOI: [10.1016/j.infrared.2023.104729](https://doi.org/10.1016/j.infrared.2023.104729)
- *方法*：综述了基于稀疏低秩恢复（IPI/PSTNN/RIPT）及深度学习的海洋红外目标检测方法，提供了复杂海杂波场景下背景抑制技术的分类对比。

---

## B. 红外海面图像中的低秩稀疏分解与RPCA背景建模

**2.** Gao, C., Meng, D., Yang, Y., Wang, Y., Zhou, X., & Hauptmann, A. G. (2013). Infrared patch-image model for small target detection in a single image. *IEEE Transactions on Image Processing*, 22(12), 4996-5009.
DOI: [10.1109/TIP.2013.2281422](https://doi.org/10.1109/TIP.2013.2281422)
- *方法*：提出经典IPI模型，将单帧红外图像转化为patch-image矩阵，利用目标稀疏性与背景低秩性构建RPCA优化问题。

**3.** Zhang, L., Peng, L., Zhang, T., Cao, S., & Peng, Z. (2018). Infrared small target detection via non-convex rank approximation minimization joint l2,1 norm. *Remote Sensing*, 10(11), 1721.
DOI: [10.3390/rs10111721](https://doi.org/10.3390/rs10111721)
- *方法*：采用非凸秩近似（l2,1范数）替代经典核范数约束，在复杂海天背景下实现更精确的低秩背景恢复与稀疏目标分离。

**4.** Zhang, T., Wu, H., Liu, Y., Peng, L., Yang, C., & Peng, Z. (2019). Infrared small target detection via non-convex tensor rank surrogate joint local energy. *Remote Sensing*, 11(17), 2032.
DOI: [10.3390/rs11172032](https://doi.org/10.3390/rs11172032)
- *方法*：引入非凸张量秩替代策略联合局部能量先验，利用红外序列帧间时空相关性提升海杂波背景建模精度。

**5.** Sun, Y., Yang, J., Long, Y., & Li, Y. (2020). Infrared small target detection via spatial-temporal total variation regularization and weighted tensor nuclear norm. *IEEE Access*, 8, 56946-56962.
DOI: [10.1109/ACCESS.2020.2982175](https://doi.org/10.1109/ACCESS.2020.2982175)
- *方法*：融合时空全变分正则化与加权张量核范数，利用海面动态纹理的时空连续性实现海杂波场景下的鲁棒目标检测。

**6.** Li, Z., Chen, J., Hou, Q., & Zhang, Y. (2022). Sparse and low-rank decomposition based on convolutional sparse coding for infrared small target detection. *Infrared Physics & Technology*, 123, 104120.
DOI: [10.1016/j.infrared.2022.104120](https://doi.org/10.1016/j.infrared.2022.104120)
- *方法*：在低秩稀疏分解框架中引入卷积稀疏编码，利用海杂波的卷积结构特征替代传统逐块稀疏约束。

**7.** Shi, Y., Wei, Y., Chen, C., & Pan, Z. (2022). Sea-surface small target detection using entropy features with dual-domain clutter suppression. *Remote Sensing Letters*, 13(8), 789-798.
DOI: [10.1080/2150704X.2022.2084821](https://doi.org/10.1080/2150704X.2022.2084821)
- *方法*：提出双域（空域+变换域）海杂波抑制方法，联合熵特征在RPCA框架下实现海面弱小目标检测。

**8.** Wang, K., Li, S., Zhu, J., & Zheng, Z. (2023). Infrared small target detection via combining weighted local contrast and low-rank sparse decomposition. *Optics Express*, 31(12), 19905-19922.
DOI: [10.1364/OE.489434](https://doi.org/10.1364/OE.489434)
- *方法*：融合加权局部对比度与低秩稀疏分解，在低秩背景建模之前利用局部先验抑制海杂波的虚假响应。

---

## C. 红外海杂波抑制的结构自适应与链式增长滤波

**9.** Huang, S., Liu, Y., He, Y., Zhang, T., & Peng, Z. (2020). Structure-adaptive clutter suppression for infrared small target detection: Chain-growth filtering. *Remote Sensing*, 12(1), 47.
DOI: [10.3390/rs12010047](https://doi.org/10.3390/rs12010047)
- *方法*：提出链式增长滤波方法，滤波模板可根据海杂波的局部结构自适应调整形状，实现比固定形状特征提取更鲁棒的海杂波抑制。

**10.** Zhang, T., Yang, C., Peng, L., & Peng, Z. (2022). Region-adaptive clutter suppression for infrared small target detection under sea-sky background. *IEEE Signal Processing Letters*, 29, 1272-1276.
DOI: [10.1109/LSP.2022.3178956](https://doi.org/10.1109/LSP.2022.3178956)
- *方法*：提出区域自适应杂波抑制方法，将海天背景划分为不同区域（海面、天空、海天线）并分别设计杂波抑制策略，避免海天线附近的虚警。

---

## D. 红外海洋图像中的数学形态学重建与背景估计

**11.** Bai, X., & Bi, Y. (2018). Derivative entropy-based contrast measure for infrared small-target detection. *IEEE Transactions on Geoscience and Remote Sensing*, 56(4), 2452-2466.
DOI: [10.1109/TGRS.2017.2781111](https://doi.org/10.1109/TGRS.2017.2781111)
- *方法*：提出基于导数熵的对比度度量，利用数学形态学顶帽变换结合多尺度熵加权，抑制海面起伏杂波同时增强小目标。

**12.** Li, Y., Zhang, Y., Yu, J., Tan, Y., & Tian, J. (2023). Morphological reconstruction and multi-scale saliency map for infrared small target detection in sea clutter scenes. *Sensors*, 23(16), 7309.
DOI: [10.3390/s23167309](https://doi.org/10.3390/s23167309)
- *方法*：提出MMRSM-TBC算法，利用数学形态学重建结合多尺度显著性图与双分支补偿策略，在海杂波背景下实现多尺度舰船目标检测。

**13.** Han, J., Liu, C., Liu, D., & Zhang, Y. (2025). Hybrid contrast method for infrared small dim target detection in complex maritime background. *Frontiers in Marine Science*, 12, 1532879.
DOI: [10.3389/fmars.2025.1532879](https://doi.org/10.3389/fmars.2025.1532879)
- *方法*：提出混合对比度方法，融合局部对比度度量与全局对比度调制，利用多尺度形态学算子抑制海杂波非均匀性。

---

## E. 红外海上目标检测中的空时动态与海杂波抑制

**14.** Deng, H., Sun, X., & Zhou, H. (2021). A small infrared target detection method based on time fluctuation and space structure features in sun-glint sea scenes. *Infrared Physics & Technology*, 118, 103889.
DOI: [10.1016/j.infrared.2021.103889](https://doi.org/10.1016/j.infrared.2021.103889)
- *方法*：利用时间波动特征与空间结构特征联合建模，针对太阳耀斑海面场景中的非平稳海杂波设计时空双域抑制策略。

**15.** Cao, F., Yang, Z., Jin, C., & Zhang, J. (2023). Spatiotemporal kernel collaborative representation for small infrared maritime target detection. *IEEE Geoscience and Remote Sensing Letters*, 20, 7002405.
DOI: [10.1109/LGRS.2023.3287012](https://doi.org/10.1109/LGRS.2023.3287012)
- *方法*：提出时空核协同表示方法，在核空间中利用帧间海杂波的预测残差分离动态海杂波与真实目标。

---

## F. 海杂波场景下的YOLO变体与注意力驱动背景抑制

**16.** Lin, J., Bai, Z., Zhang, J., & Huang, Y. (2023). MS2-YOLO: Multi-scale ship detection method for infrared images in sea clutter environment. *IEEE Access*, 11, 78594-78604.
DOI: [10.1109/ACCESS.2023.3298533](https://doi.org/10.1109/ACCESS.2023.3298533)
- *方法*：在YOLOv5架构中嵌入多尺度特征融合与海杂波注意力抑制模块，通过通道注意力机制降低海杂波的响应权重。

**17.** Guo, L., Wang, Y., Guo, M., & Zhou, X. (2025). YOLO-IRS: Infrared ship detection algorithm based on self-attention mechanism and KAN in complex marine background. *Remote Sensing*, 17(1), 20.
DOI: [10.3390/rs17010020](https://doi.org/10.3390/rs17010020)
- *方法*：基于YOLOv10引入Swin Transformer移位窗口自注意力机制与C3KAN模块，在保持检测速度的同时提升复杂海空背景下多尺度红外舰船的检测精度。

**18.** Zheng, Y., Ao, X., & Xie, W. (2025). IRS-DETR: Lightweight real-time transformer for infrared ship detection in maritime surveillance. *Ocean Engineering*, 343, 123319.
DOI: [10.1016/j.oceaneng.2025.123319](https://doi.org/10.1016/j.oceaneng.2025.123319)
- *方法*：基于RT-DETR轻量框架提出IRS-DETR，通过PMANet部分多尺度特征聚合骨干网络与Transformer交叉注意力机制，隐式抑制海杂波背景干扰，实现边缘设备上的实时红外舰船检测。

---

## 修订记录（与初版对比）

| 条目 | 变更说明 |
|------|---------|
| #9  | 链式增长滤波原始论文作者由"Zhang et al. (2021) IEEE TIP"更正为 Huang, Liu, He, Zhang & Peng (2020), *Remote Sensing* 12(1), 47。此前记录有误。 |
| #17 | 原"Wang et al. (2024) YOLO-IRS, *Infrared Physics & Technology*" 经验证不存在；替换为 Guo et al. (2025), *Remote Sensing* 17(1), 20。作者为Guo/Wang/Guo/Zhou。 |
| #18 | 原"Zheng et al. (2026) IRS-DETR, *Ocean Engineering* 312" 更正为 Zheng, Ao & Xie (2025), *Ocean Engineering* 343, 123319。发表年份实为2025。DOI已验证。 |