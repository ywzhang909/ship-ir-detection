# -*- coding: utf-8 -*-
"""数据图表生成脚本（fig4–fig9，共 6 张）。

唯一数据来源：docs/论文/data_tables.md（论文唯一数字真相库）。
所有数字逐字转写，禁止四舍五入/发明/推算；禁止伪造训练曲线（本脚本只画
指标柱状图/排名图/差距图，不画任何训练曲线）。

输出（docs/论文/figures/）：
    fig4_run123_val_metrics.png    Run1/2/3 验证指标柱状图
    fig5_test_mAP50_ranking.png    全实验测试集 mAP50 排名
    fig6_generalization_gap.png    验证-测试泛化差距
    fig7_loss_ablation.png         损失函数消融
    fig8_sahi_vs_standard.png      SAHI vs 标准推理
    fig9_dual_randomization.png    双随机化策略对比（augment-jitter vs preprocess-jitter）

运行（从 generate_figures/ 目录）：
    /home/ws/code/ship/.venv/bin/python gen_data_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common_style

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# ---------------------------------------------------------------------------
# 数据（逐字转写自 data_tables.md，禁止改动）
# ---------------------------------------------------------------------------

# 表3：Run1/2/3 验证指标（小数 0–1）
RUN123_VAL = {
    "Run 1": {"mAP50": 0.9396, "mAP50-95": 0.6366, "P": 0.9358, "R": 0.9052},
    "Run 2": {"mAP50": 0.9465, "mAP50-95": 0.6433, "P": 0.9751, "R": 0.9302},
    "Run 3": {"mAP50": 0.9568, "mAP50-95": 0.6551, "P": 0.9569, "R": 0.9532},
}
RUN123_LABEL = {
    "Run 1": "Run 1\nTop-hat+Bottom-hat\n+CLAHE",
    "Run 2": "Run 2\nButterworth BPF\n+CLAHE",
    "Run 3": "Run 3\nDual-stream\n+SpatialFrequencyFusion",
}

# 表4：全实验测试集 mAP50 排名（降序，%）
RANKING = [
    ("T16", "CRRP + YOLO11l + Fusion", 98.13),
    ("T5", "YOLO11l + Fusion", 97.87),
    ("T14", "ASFF Neck", 97.87),
    ("T12", "C3K3 骨干替换", 97.77),
    ("T4", "DWT Wavelet + Fusion", 97.45),
    ("T11", "FECA 注意力", 96.64),
    ("T7", "Augment jitter 0.3 + Fusion", 96.63),
    ("T25", "Preprocess jitter + YOLO11m + Fusion", 96.47),
    ("T19", "C2EMA 注意力插入 Neck", 96.27),
    ("T28", "SAHI 集成（yolo11l）", 94.63),
    ("Run 2", "Butterworth BPF + CLAHE", 93.97),
    ("T10", "NWD 损失", 93.42),
    ("T9", "WIoU v3 损失", 93.18),
    ("Run 1", "Top-hat + CLAHE", 92.66),
    ("T8", "Shape-IoU 损失", 86.25),
    ("Run 3", "Fusion 早期版本", 52.30),
    ("T1", "大融合架构（h=32,l=3,CBAM）", 37.67),
    ("T2", "残差强度 0.7", 37.67),
]
OVERFIT_IDS = {"Run 3", "T1", "T2"}  # 严重过拟合（Δ < −37）

# 表5：验证-测试泛化差距（%，Δ 为百分点）
GAP = [
    ("Run 1", 93.96, 92.66, -1.3),
    ("Run 2", 94.65, 93.97, -0.7),
    ("Run 3", 95.68, 52.30, -43.4),
    ("T8", 88.43, 86.25, -2.2),
    ("T1", 95.47, 37.67, -57.8),
    ("T2", 95.47, 37.67, -57.8),
    ("T4", 95.49, 97.45, 2.0),
    ("T5", 97.20, 97.87, 0.7),
    ("T7", 97.23, 96.63, -0.6),
    ("T10", 94.00, 93.42, -0.6),
    ("T11", 95.20, 96.64, 1.4),
    ("T12", 97.04, 97.77, 0.7),
    ("T14", 97.00, 97.87, 0.9),
    ("T15", 95.15, 96.64, 1.5),
    ("T16", 97.85, 98.13, 0.3),
    ("T25", 97.11, 96.47, -0.6),
    ("T9", 93.50, 93.18, -0.32),
]

# 表8：损失函数消融（test %）
LOSS = [
    ("CIoU（基线）", "T5", 97.87, 68.02, 97.07, 94.58),
    ("Shape-IoU", "T8", 86.25, 56.71, 90.68, 77.23),
    ("WIoU v3", "T9", 93.18, 60.85, 92.80, 86.64),
    ("NWD", "T10", 93.42, 62.01, None, None),  # T10 测试 P/R 无记录
]

# 表6：SAHI vs 标准（test mAP50 %）
SAHI = [
    ("Run 1", 92.66, 30.84, -61.82),
    ("Run 2", 93.97, 61.34, -32.63),
    ("Run 3", 52.30, 14.92, -37.38),
    ("T1", 37.67, 89.75, 52.08),
    ("T2", 37.67, 89.75, 52.08),
    ("T4", 97.45, 87.49, -9.96),
    ("T5", 97.87, 77.24, -20.63),
    ("T7", 96.63, 79.45, -17.18),
    ("T8", 86.25, 8.97, -77.28),
    ("T9", 93.18, 59.81, -33.37),
]

# 表3：双随机化策略（Run3 基线 / T7 augment-jitter / T25 preprocess-jitter）
# val 为小数（表3），此处按表5 口径 ×100 转百分数（93.96% = 0.9396，精确换算）
DUAL = [
    ("Run 3\n（Fusion 基线）", 95.68, 52.30, 25.33),
    ("T7\naugment-jitter=0.3", 97.23, 96.63, 65.46),
    ("T25\npreprocess-jitter", 97.11, 96.47, 68.50),
]


# ---------------------------------------------------------------------------
# 绘图辅助
# ---------------------------------------------------------------------------
def _fmt(v: float, nd: int = 2) -> str:
    """保留 nd 位小数的字符串（不四舍五入数据本身，仅格式化显示）。"""
    return f"{v:.{nd}f}"


def _bar_labels(ax, bars, fmt: str = "{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


# ---------------------------------------------------------------------------
# fig4：Run1/2/3 验证指标柱状图
# ---------------------------------------------------------------------------
def fig4_run123_val_metrics():
    metrics = ["mAP50", "mAP50-95", "P", "R"]
    runs = ["Run 1", "Run 2", "Run 3"]
    x = np.arange(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, run in enumerate(runs):
        vals = [RUN123_VAL[run][m] for m in metrics]
        bars = ax.bar(
            x + (i - 1) * width, vals, width,
            color=common_style.PALETTE[i], label=run,
        )
        _bar_labels(ax, bars, "{:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("验证集指标（0–1）")
    ax.set_title("Run 1/2/3 验证集指标对比（表3）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.text(
        0.99, 0.02,
        "Run1: 空域杂波抑制（Top-hat+Bottom-hat+CLAHE）\n"
        "Run2: 频域杂波抑制（Butterworth BPF+CLAHE）\n"
        "Run3: 空频融合（Dual-stream+SpatialFrequencyFusion, h=16）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="#555555",
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig4_run123_val_metrics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig5：全实验测试集 mAP50 排名（水平条形，降序）
# ---------------------------------------------------------------------------
def fig5_test_mAP50_ranking():
    ids = [r[0] for r in RANKING]
    vals = [r[2] for r in RANKING]
    labels = [f"{r[0]}  {r[1]}" for r in RANKING]

    fig, ax = plt.subplots(figsize=(9, 7.2))
    colors = [
        common_style.PALETTE[2] if i == 0 else          # T16 最优 → 绿
        common_style.PALETTE[3] if ids[i] in OVERFIT_IDS else  # 严重过拟合 → 红
        common_style.PALETTE[0]                          # 其余 → 蓝
        for i in range(len(ids))
    ]
    bars = ax.barh(np.arange(len(ids)), vals, color=colors, height=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:.2f}%",
            xy=(v, b.get_y() + b.get_height() / 2),
            xytext=(3, 0), textcoords="offset points",
            va="center", fontsize=8,
        )

    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # 最高分在顶部
    ax.set_xlim(0, 108)
    ax.set_xlabel("测试集 mAP50（%）")
    ax.set_title("全实验测试集 mAP50 排名（表4，降序）")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=common_style.PALETTE[2], label="最优（T16, 98.13%）"),
            Patch(color=common_style.PALETTE[0], label="正常实验"),
            Patch(color=common_style.PALETTE[3], label="严重过拟合（Run3/T1/T2）"),
        ],
        loc="lower right", fontsize=8,
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig5_test_mAP50_ranking.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig6：验证-测试泛化差距（分组柱状 + Δ 标注）
# ---------------------------------------------------------------------------
def fig6_generalization_gap():
    ids = [g[0] for g in GAP]
    val = [g[1] for g in GAP]
    test = [g[2] for g in GAP]
    delta = [g[3] for g in GAP]

    x = np.arange(len(ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - width / 2, val, width, color=common_style.PALETTE[0], label="验证集 mAP50")
    b2 = ax.bar(x + width / 2, test, width, color=common_style.PALETTE[1], label="测试集 mAP50")
    _bar_labels(ax, b1, "{:.2f}")
    _bar_labels(ax, b2, "{:.2f}")

    # Δ 标注（百分点）
    for xi, d in zip(x, delta):
        color = common_style.PALETTE[3] if d < -37 else "#333333"
        ax.annotate(
            f"Δ{d:+.2f}" if d != -0.32 else "Δ−0.32",
            xy=(xi, max(val[xi], test[xi])),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=8.5, rotation=45, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("mAP50（%）")
    ax.set_title("验证集 vs 测试集 mAP50 泛化差距（表5，Δ=测试−验证，百分点）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.text(
        0.99, 0.02,
        "红色 Δ 为严重过拟合（Δ<−37）：Run3（−43.4）、T1/T2（−57.8）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color=common_style.PALETTE[3],
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig6_generalization_gap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig7：损失函数消融（test mAP50 / mAP50-95）
# ---------------------------------------------------------------------------
def fig7_loss_ablation():
    names = [l[0] for l in LOSS]
    exp = [l[1] for l in LOSS]
    m50 = [l[2] for l in LOSS]
    m5095 = [l[3] for l in LOSS]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - width / 2, m50, width, color=common_style.PALETTE[0], label="test mAP50")
    b2 = ax.bar(x + width / 2, m5095, width, color=common_style.PALETTE[4], label="test mAP50-95")
    _bar_labels(ax, b1, "{:.2f}")
    _bar_labels(ax, b2, "{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n（{e}）" for n, e in zip(names, exp)], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("测试集指标（%）")
    ax.set_title("损失函数消融（表8）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.text(
        0.99, 0.02,
        "注：T8（Shape-IoU）为 YOLO11m + awb_tophat_clahe（无融合，epochs 200），\n"
        "T5/T9/T10 为 YOLO11l + Fusion，配置不同，对比时需注明。\n"
        "T10（NWD）测试 P/R 源文件无记录，未绘制。",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        color="#555555",
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig7_loss_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig8：SAHI vs 标准推理（test mAP50）
# ---------------------------------------------------------------------------
def fig8_sahi_vs_standard():
    ids = [s[0] for s in SAHI]
    std = [s[1] for s in SAHI]
    sahi = [s[2] for s in SAHI]
    delta = [s[3] for s in SAHI]

    x = np.arange(len(ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    b1 = ax.bar(x - width / 2, std, width, color=common_style.PALETTE[0], label="标准推理 mAP50")
    b2 = ax.bar(x + width / 2, sahi, width, color=common_style.PALETTE[2], label="SAHI mAP50")
    _bar_labels(ax, b1, "{:.2f}")
    _bar_labels(ax, b2, "{:.2f}")

    for xi, d in zip(x, delta):
        color = common_style.PALETTE[2] if d > 0 else common_style.PALETTE[3]
        ax.annotate(
            f"Δ{d:+.2f}",
            xy=(xi, max(std[xi], sahi[xi])),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("测试集 mAP50（%）")
    ax.set_title("SAHI 滑窗 vs 标准推理（表6，切片 512×512，overlap=0.2）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.text(
        0.99, 0.02,
        "SAHI 仅对严重过拟合的 T1/T2 有 Recall 恢复效果（Δ+52.08），\n"
        "对泛化良好模型均造成 mAP50 下降（−9.96 ~ −77.28）。",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="#555555",
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig8_sahi_vs_standard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig9：双随机化策略对比（Run3 基线 / T7 augment-jitter / T25 preprocess-jitter）
# ---------------------------------------------------------------------------
def fig9_dual_randomization():
    names = [d[0] for d in DUAL]
    val = [d[1] for d in DUAL]
    test = [d[2] for d in DUAL]
    test95 = [d[3] for d in DUAL]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - width, val, width, color=common_style.PALETTE[0], label="验证集 mAP50")
    b2 = ax.bar(x, test, width, color=common_style.PALETTE[1], label="测试集 mAP50")
    b3 = ax.bar(x + width, test95, width, color=common_style.PALETTE[4], label="测试集 mAP50-95")
    _bar_labels(ax, b1, "{:.2f}")
    _bar_labels(ax, b2, "{:.2f}")
    _bar_labels(ax, b3, "{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("指标（%）")
    ax.set_title("双随机化策略对比（表3：Run3 基线 / T7 / T25）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.text(
        0.99, 0.02,
        "三者均为 20.0M + 3,361 融合配置。\n"
        "T7（augment-jitter=0.3）与 T25（preprocess-jitter）均修复 Run3 严重过拟合\n"
        "（测试 mAP50：52.30% → 96.63% / 96.47%）。",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="#555555",
    )
    common_style.save_fig(fig, FIGURES_DIR / "fig9_dual_randomization.png")
    plt.close(fig)


def main() -> None:
    fig4_run123_val_metrics()
    fig5_test_mAP50_ranking()
    fig6_generalization_gap()
    fig7_loss_ablation()
    fig8_sahi_vs_standard()
    fig9_dual_randomization()
    print("6 张图已生成到", FIGURES_DIR)


if __name__ == "__main__":
    sys.exit(main())