# -*- coding: utf-8 -*-
"""图：特征提取/预处理方法 — 各方法处理后图片 + 处理后特征分析。

对 3 张真实 NSLSR 验证帧（data/weights_staging/frames/val/val_batch{0,1,2}_pred_101_clean.jpg，
1920×1008）运行 yolo-training/preprocessing_module.py 的 15 种特征提取/预处理方法
（AVAILABLE_PREPROCESS 全集），真实代码、真实帧、禁止伪造。

输出（docs/论文/figures/preprocess/）：
  1. 每张验证帧 × 每种方法的处理后单图：
        frame{i}_{method}.png      （单流方法为 2D 灰度，双流方法为 3 通道）
  2. 一张 3×6 对比总图：
        fig11_preprocess_all_methods.png
        行=3 张验证帧，列=15 种方法（含原始），共 18 面板
  3. 处理后特征分析统计 JSON：
        preprocess_feature_stats.json
        每个方法每帧的 mean/std(对比度)/entropy/峰值直方图占比等，
        用于正文"处理后特征分析"。

复用 common_style.save_fig（CJK 无豆腐块、DPI=200）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # ship/
YOLO_DIR = REPO_ROOT / "yolo-training"
FIG_DIR = REPO_ROOT / "docs" / "论文" / "figures" / "preprocess"
GEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(YOLO_DIR))   # preprocessing_module
sys.path.insert(0, str(GEN_DIR))    # common_style

import common_style                      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import preprocessing_module as pm        # noqa: E402

# ---------------------------------------------------------------------------
# 源帧：3 张真实 NSLSR 验证帧
# ---------------------------------------------------------------------------
FRAMES = [
    REPO_ROOT / "data" / "weights_staging" / "frames" / "val" / "val_batch0_pred_101_clean.jpg",
    REPO_ROOT / "data" / "weights_staging" / "frames" / "val" / "val_batch1_pred_101_clean.jpg",
    REPO_ROOT / "data" / "weights_staging" / "frames" / "val" / "val_batch2_pred_101_clean.jpg",
]

# 方法显示名（论文用语，不用 run 码；与图注一致）
METHOD_DISPLAY = {
    "none": "原始",
    "tophat_clahe": "空域Top-hat+CLAHE",
    "tophat_detail": "Top-hat+底帽",
    "dog_retinex": "DoG高斯差分",
    "retinex_gamma": "Retinex+Gamma",
    "median_clahe": "中值+CLAHE",
    "bilateral_gamma": "双边滤波+Gamma",
    "ssl_adaptive": "海天线自适应",
    "awb_tophat_clahe": "AWB+空域Top-hat+CLAHE",
    "awb_butterworth_clahe": "AWB+频域Butterworth+CLAHE",
    "awb_butterworth_retinex": "AWB+频域Butterworth+Retinex",
    "awb_dual_fusion": "AWB+双流空频融合",
    "awb_wavelet_clahe": "AWB+小波+CLAHE",
    "awb_wavelet_dual_fusion": "AWB+小波双流融合",
    "awb_rpca_dual_fusion": "AWB+RPCA双流融合",
}

# 列顺序：原始 + 14 个方法（共 15 列 → 用 3 行 × 6 列 排版，前 15 个占 3 行）
COL_ORDER = [
    "none", "tophat_clahe", "awb_tophat_clahe", "awb_butterworth_clahe",
    "awb_wavelet_clahe", "awb_dual_fusion",
    "tophat_detail", "dog_retinex", "retinex_gamma", "median_clahe",
    "bilateral_gamma", "ssl_adaptive",
    "awb_butterworth_retinex", "awb_wavelet_dual_fusion", "awb_rpca_dual_fusion",
]
assert len(COL_ORDER) == 15
assert set(COL_ORDER) == set(pm.AVAILABLE_PREPROCESS.keys())

# ---------------------------------------------------------------------------
# 真实运行 15 种方法 × 3 帧
# ---------------------------------------------------------------------------
FIG_DIR.mkdir(parents=True, exist_ok=True)

# results[frame_idx][method] = np.ndarray (2D or 3D)
results = {i: {} for i in range(len(FRAMES))}
stats = {}  # method -> {frame_i -> {mean,std,entropy,p99,peak_frac}}

for fi, frame_path in enumerate(FRAMES):
    bgr = cv2.imread(str(frame_path))
    assert bgr is not None, f"无法读取帧: {frame_path}"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for name in pm.AVAILABLE_PREPROCESS:
        pipe, _desc, _params = pm.get_preprocessing_pipeline(name)
        if isinstance(pipe, pm.PreprocessingPipelineDual):
            out = pipe(bgr)          # 3 通道
        else:
            out = pipe(gray)         # 2D
        assert out is not None and out.size > 0, f"{name} 帧{fi} 渲染失败"
        results[fi][name] = out
        # 保存处理后单图
        out_path = FIG_DIR / f"frame{fi}_{name}.png"
        cv2.imwrite(str(out_path), out)

        # 特征统计（对灰度做；3 通道取均值灰度）
        g = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
        hist = hist / hist.sum()
        entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        p99 = float(np.percentile(g, 99))
        peak_frac = float(hist.max())
        stats.setdefault(name, {})[f"frame{fi}"] = {
            "mean": round(float(g.mean()), 3),
            "std": round(float(g.std()), 3),
            "contrast_ratio": round(float(g.std() / (g.mean() + 1e-6)), 4),
            "entropy": round(entropy, 4),
            "p99": round(p99, 2),
            "peak_hist_frac": round(peak_frac, 4),
            "out_shape": list(g.shape) + ([3] if out.ndim == 3 else []),
        }

# ---------------------------------------------------------------------------
# 处理后特征分析 JSON
# ---------------------------------------------------------------------------
import json
stats_path = FIG_DIR / "preprocess_feature_stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "frames": [p.name for p in FRAMES],
            "methods": [METHOD_DISPLAY[n] for n in COL_ORDER],
            "method_keys": COL_ORDER,
            "stats": stats,
        },
        f, ensure_ascii=False, indent=2,
    )
print(f"已保存特征统计: {stats_path}")

# ---------------------------------------------------------------------------
# 竖向排版：行=方法(15)、列=帧(3)
# ---------------------------------------------------------------------------
N_METHODS = len(COL_ORDER)
N_FRAMES = len(FRAMES)              # 3
fig, axes = plt.subplots(N_METHODS, N_FRAMES, figsize=(4.2 * N_FRAMES, 1.45 * N_METHODS))
fig.suptitle(
    "特征提取/预处理方法：各方法处理后图片（行=方法，列=验证帧 0/1/2）",
    fontsize=15, y=0.998,
)

for r, name in enumerate(COL_ORDER):
    for c in range(N_FRAMES):
        ax = axes[r, c]
        out = results[c][name]
        if out.ndim == 3:
            disp = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            ax.imshow(disp)
        else:
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#4C72B0")
            spine.set_linewidth(0.6)
        if c == 0:
            ax.set_ylabel(METHOD_DISPLAY[name], fontsize=8, rotation=0,
                          labelpad=8, va="center")

for c in range(N_FRAMES):
    axes[0, c].set_title(f"帧{c}", fontsize=10, pad=6)

fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
out_path = FIG_DIR / "fig11_preprocess_all_methods.png"
common_style.save_fig(fig, out_path, dpi=200)
print(f"已保存对比总图(竖向): {out_path}")

# ---------------------------------------------------------------------------
# 处理后特征分析图（对比度/熵 随方法变化，3 帧均值）
# ---------------------------------------------------------------------------
method_keys = COL_ORDER
labels = [METHOD_DISPLAY[n] for n in method_keys]
mean_vals = []
std_vals = []
ent_vals = []
for n in method_keys:
    ms = [stats[n][f"frame{i}"]["mean"] for i in range(N_FRAMES)]
    ss = [stats[n][f"frame{i}"]["std"] for i in range(N_FRAMES)]
    es = [stats[n][f"frame{i}"]["entropy"] for i in range(N_FRAMES)]
    mean_vals.append(sum(ms) / len(ms))
    std_vals.append(sum(ss) / len(ss))
    ent_vals.append(sum(es) / len(es))

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
x = np.arange(len(method_keys))
ax1.bar(x, mean_vals, color=common_style.PALETTE[0])
ax1.set_ylabel("均值（灰度）")
ax1.set_title("各方法处理后图像灰度均值（3 帧平均）")
ax1.set_xticks(x)
ax1.set_xticklabels([l[:5] for l in labels], rotation=45, ha="right", fontsize=8)
ax2.bar(x, std_vals, color=common_style.PALETTE[3])
ax2.set_ylabel("标准差（对比度）")
ax2.set_title("各方法处理后图像标准差（对比度，3 帧平均）")
ax2.set_xticks(x)
ax2.set_xticklabels([l[:5] for l in labels], rotation=45, ha="right", fontsize=8)
ax3.bar(x, ent_vals, color=common_style.PALETTE[2])
ax3.set_ylabel("信息熵")
ax3.set_title("各方法处理后图像信息熵（3 帧平均）")
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
out_path2 = FIG_DIR / "fig12_preprocess_feature_analysis.png"
common_style.save_fig(fig2, out_path2, dpi=200)
print(f"已保存特征分析图: {out_path2}")

print("\n===== 处理后特征分析摘要（3 帧平均）=====")
print(f"{'方法':34s}  均值     标准差   熵")
for n, mv, sv, ev in zip(method_keys, mean_vals, std_vals, ent_vals):
    print(f"{METHOD_DISPLAY[n]:34s}  {mv:7.2f}  {sv:6.2f}  {ev:6.3f}")
print(f"\n共生成 {len(FRAMES)*len(pm.AVAILABLE_PREPROCESS)} 张处理后单图 + 2 张汇总图。")
