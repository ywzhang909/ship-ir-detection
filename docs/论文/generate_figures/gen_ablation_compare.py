# -*- coding: utf-8 -*-
"""图：模型改进消融实验 — 不同消融下的识别结果（带置信度）+ 对比图。

数据来源（真实，禁止伪造）：
  - 识别结果图：docs/论文/figures/detection/det_{model}.png（10 个消融模型，
    标准推理 conf=0.05，含 YOLO 默认带置信度标注框）。
  - 检测统计：docs/论文/figures/detection/inference_stats.json
    （每模型 frame0/1/2 的 detections / avg_conf / max_conf / min_conf / inference_ms）。

模型显示名（论文用语，不用 run 码）：
  baseline_yolo11m     → YOLO11m + 空频融合 基线
  starnet_s1           → StarNet S1
  T1_fusion_h32_l3_cbam→ YOLO11m + 大空频融合架构（过拟合）
  T2_fusion_residual07 → YOLO11m + 空频残差搜索架构（过拟合）
  T4_wavelet_dual      → YOLO11m + 小波双流融合
  T5_yolo11l_fusion    → YOLO11l + 空频融合
  T7_augment_jitter    → YOLO11m + 空频融合 + 增强双随机化
  T16_crrp             → YOLO11l + 空频融合 + CRRP
  T20_dynamic_head     → YOLO11l + 动态检测头
  T22_large            → YOLO11l + 大容量融合

输出（docs/论文/figures/detection/）：
  1. fig13_ablation_detection_compare.png
     10 模型 × 1 列，每格为该模型识别结果图（带置信度框），
     标题=方法/模型名 + 检出数 + 均置信度。
  2. fig14_ablation_confidence_compare.png
     双面板：左=各模型检出数(frame0)，右=各模型均置信度(frame0)，条形对比。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # ship/
FIG_DIR = REPO_ROOT / "docs" / "论文" / "figures" / "detection"
GEN_DIR = Path(__file__).resolve().parent
STATS = FIG_DIR / "inference_stats.json"

sys.path.insert(0, str(GEN_DIR))    # common_style

import common_style                      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402

# ---------------------------------------------------------------------------
# 模型 → 显示名（论文用语，无 run 码）
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "baseline_yolo11m": "YOLO11m + 空频融合 基线",
    "starnet_s1": "StarNet S1",
    "T1_fusion_h32_l3_cbam": "YOLO11m + 大空频融合架构（过拟合）",
    "T2_fusion_residual07": "YOLO11m + 空频残差搜索架构（过拟合）",
    "T4_wavelet_dual": "YOLO11m + 小波双流融合",
    "T5_yolo11l_fusion": "YOLO11l + 空频融合",
    "T7_augment_jitter": "YOLO11m + 空频融合 + 增强双随机化",
    "T16_crrp": "YOLO11l + 空频融合 + CRRP",
    "T20_dynamic_head": "YOLO11l + 动态检测头",
    "T22_large": "YOLO11l + 大容量融合",
}

# 模型 → 实际检测图文件名（f0 帧，带置信度框）
MODEL_IMG = {
    "baseline_yolo11m": "det_baseline_f0.png",
    "starnet_s1": "det_starnet_f0.png",
    "T1_fusion_h32_l3_cbam": "det_T1_fusion_f0.png",
    "T2_fusion_residual07": "det_T2_fusion_f0.png",
    "T4_wavelet_dual": "det_T4_wavelet_f0.png",
    "T5_yolo11l_fusion": "det_T5_yolo11l_f0.png",
    "T7_augment_jitter": "det_T7_jitter_f0.png",
    "T16_crrp": "det_T16_crrp_f0.png",
    "T20_dynamic_head": "det_T20_dyn_f0.png",
    "T22_large": "det_T22_large_f0.png",
}

with open(STATS, encoding="utf-8") as f:
    stats = json.load(f)

# 模型顺序（按论文叙述：基线 → 过拟合消融 → 有效改进 → 其他）
ORDER = [
    "baseline_yolo11m",
    "T1_fusion_h32_l3_cbam",
    "T2_fusion_residual07",
    "T4_wavelet_dual",
    "T5_yolo11l_fusion",
    "T7_augment_jitter",
    "T16_crrp",
    "T20_dynamic_head",
    "T22_large",
    "starnet_s1",
]
for m in ORDER:
    assert m in MODEL_DISPLAY, f"{m} 缺显示名"
    assert m in stats, f"{m} 缺统计"
    assert m in MODEL_IMG, f"{m} 缺图片文件名"
    img = FIG_DIR / MODEL_IMG[m]
    assert img.exists() and img.stat().st_size > 10240, f"检测图缺失: {img}"

# ---------------------------------------------------------------------------
# 图 1：消融模型识别结果对比（带置信度）
#   10 模型 → 5 行 × 2 列，每格 det_{m}.png + 标题（方法名 + 检出/均置信度）
# ---------------------------------------------------------------------------
det_imgs = {}
for m in ORDER:
    img = cv2.imread(str(FIG_DIR / MODEL_IMG[m]))
    assert img is not None
    det_imgs[m] = img

N = len(ORDER)  # 10
rows, cols = 5, 2
fig, axes = plt.subplots(rows, cols, figsize=(16, 3.4 * rows))
fig.suptitle(
    "模型改进消融实验：不同消融下的识别结果（带置信度，conf=0.05，frame0）",
    fontsize=15, y=0.998,
)
for idx, m in enumerate(ORDER):
    r, c = idx // cols, idx % cols
    ax = axes[r, c]
    s = stats[m]["frame0"]
    disp = cv2.cvtColor(det_imgs[m], cv2.COLOR_BGR2RGB)
    ax.imshow(disp)
    ax.set_title(
        f"{MODEL_DISPLAY[m]}\n检出 {s['detections']} | 均置信度 {s['avg_conf']*100:.1f}% | 最高 {s['max_conf']*100:.1f}%",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#4C72B0")
        spine.set_linewidth(0.8)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out1 = FIG_DIR / "fig13_ablation_detection_compare.png"
common_style.save_fig(fig, out1, dpi=200)
print(f"已保存: {out1}")

# ---------------------------------------------------------------------------
# 图 2：检出数 + 均置信度 条形对比（frame0）
# ---------------------------------------------------------------------------
labels = [MODEL_DISPLAY[m].split("（")[0] for m in ORDER]  # 短标签
det_counts = [stats[m]["frame0"]["detections"] for m in ORDER]
avg_confs = [stats[m]["frame0"]["avg_conf"] * 100 for m in ORDER]
max_confs = [stats[m]["frame0"]["max_conf"] * 100 for m in ORDER]

# 颜色：基线/过拟合 红，有效改进 蓝，其他 紫
color_map = {
    "baseline_yolo11m": common_style.PALETTE[0],
    "T1_fusion_h32_l3_cbam": common_style.PALETTE[3],
    "T2_fusion_residual07": common_style.PALETTE[3],
    "T4_wavelet_dual": common_style.PALETTE[0],
    "T5_yolo11l_fusion": common_style.PALETTE[0],
    "T7_augment_jitter": common_style.PALETTE[2],
    "T16_crrp": common_style.PALETTE[2],
    "T20_dynamic_head": common_style.PALETTE[4],
    "T22_large": common_style.PALETTE[3],
    "starnet_s1": common_style.PALETTE[5],
}
colors = [color_map[m] for m in ORDER]

x = np.arange(len(ORDER))
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
bars1 = ax1.bar(x, det_counts, color=colors)
ax1.set_ylabel("检出目标数（frame0）")
ax1.set_title("各消融模型检出目标数（conf=0.05，frame0）")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
for b, v in zip(bars1, det_counts):
    ax1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
ax1.set_ylim(0, max(det_counts) * 1.15)

bars2 = ax2.bar(x, avg_confs, color=colors)
ax2.set_ylabel("均置信度（%）")
ax2.set_title("各消融模型均置信度 / 最高置信度（frame0）")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
for b, v in zip(bars2, avg_confs):
    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
# 最高置信度折线叠加
ax2.plot(x, max_confs, "o-", color="#333333", lw=1.5, ms=5, label="最高置信度")
ax2.set_ylim(0, max(max_confs) * 1.2)
ax2.legend(loc="upper right", fontsize=9)

fig2.tight_layout(rect=[0, 0, 1, 0.97])
out2 = FIG_DIR / "fig14_ablation_confidence_compare.png"
common_style.save_fig(fig2, out2, dpi=200)
print(f"已保存: {out2}")

print("\n===== 消融模型 frame0 识别统计 =====")
print(f"{'模型':40s}  检出   均置信度   最高置信度")
for m in ORDER:
    s = stats[m]["frame0"]
    print(f"{MODEL_DISPLAY[m]:40s}  {s['detections']:>3}   {s['avg_conf']*100:6.1f}%   {s['max_conf']*100:6.1f}%")
print(f"\n共生成 2 张消融对比图（图13/图14）。")
