# -*- coding: utf-8 -*-
"""生成论文架构示意图：Fig1 整体流水线 / Fig2 空频融合模块 / Fig10 CRRP 增强流程。

数字来源：docs/论文/data_tables.md（唯一数字真相库）+ 源码
（fusion_module.py / crrp_augment.py）。复用 common_style.py 统一风格与 CJK 字体，
禁止伪造参数、禁止占位符。

输出（docs/论文/figures/）：
    fig1_pipeline.png      整体技术路线图
    fig2_fusion_module.png 空频融合模块结构
    fig10_crrp.png         CRRP 增强流程
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_style  # noqa: E402  (导入即完成 CJK 字体配置)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# 调色板（common_style 统一 6 色）
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN = common_style.PALETTE

# 中性色
INK = "#2B2B2B"
LIGHT_BG = "#F5F6F8"


# ---------------------------------------------------------------------------
# 绘制辅助
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, fc, ec, title=None, lines=None, title_fs=12,
         lw=1.6, title_color="white", rounded=0.08, zorder=3):
    """圆角矩形 + 可选标题条 + 多行文本。x,y 为左下角。lines: (text, fs, weight, color)"""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(p)
    if title:
        bar_h = 0.55
        bar = FancyBboxPatch(
            (x, y + h - bar_h), w, bar_h,
            boxstyle=f"round,pad=0.02,rounding_size={rounded}",
            linewidth=0, edgecolor="none", facecolor=ec, zorder=zorder + 1,
        )
        ax.add_patch(bar)
        ax.text(x + w / 2, y + h - bar_h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=title_color, zorder=zorder + 2)
    if lines:
        n = len(lines)
        top = y + h - (0.55 if title else 0.12)
        bottom = y + 0.12
        step = (top - bottom) / max(n, 1)
        for i, (txt, fsz, wgt, col) in enumerate(lines):
            yy = top - step * (i + 0.5)
            ax.text(x + w / 2, yy, txt, ha="center", va="center",
                    fontsize=fsz, fontweight=wgt, color=col, zorder=zorder + 2)


def _arrow(ax, x1, y1, x2, y2, color=INK, lw=2.0, ms=16, ls="-", rad=0.0, zorder=2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def _new_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Fig1 整体技术路线图
# ---------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = _new_ax((20, 10.5), (0, 20), (0, 10.5))

    # 顶部：数据集信息条
    _box(ax, 0.4, 9.35, 19.2, 0.85, LIGHT_BG, BLUE,
         lines=[
             ("数据集 NSLSR-LWIR：红外伪彩色 RGB · 512×640 · 1 类（ship）· 目标 5–30 px",
              12, "bold", INK),
             ("训练集 803 张 ｜ 验证集 243 张 ｜ 测试集 118 张（277 实例）",
              10.5, "normal", "#555555"),
         ])

    # 五个阶段
    stage_w, gap = 3.62, 0.28
    x0 = 0.4
    xs = [x0 + i * (stage_w + gap) for i in range(5)]
    y_stage, h_stage = 4.35, 4.7

    # 1 数据预处理
    _box(ax, xs[0], y_stage, stage_w, h_stage, "#EAF0F8", BLUE,
         title="① 数据预处理",
         lines=[
             ("AWB 白平衡（corner 0.06, gray 128）", 10.5, "bold", INK),
             ("空域：Top-hat/Bottom-hat", 9.5, "normal", "#333333"),
             ("（kernel=31）+ CLAHE", 9.5, "normal", "#333333"),
             ("（clip_limit=3.0, 8×8）", 9.5, "normal", "#333333"),
             ("频域：Butterworth BPF", 9.5, "normal", "#333333"),
             ("（cutoff=8–80, order=2）", 9.5, "normal", "#333333"),
             ("小波：DWT db4", 9.5, "normal", "#333333"),
             ("（LL×0.3, LH/HL×1.5, HH×0.3）", 9.5, "normal", "#333333"),
         ])
    # 2 数据增强
    _box(ax, xs[1], y_stage, stage_w, h_stage, "#FDF1E7", ORANGE,
         title="② 数据增强",
         lines=[
             ("CRRP 2× 离线增强", 10.5, "bold", INK),
             ("（旋转 ±30°, 缩放", 9.5, "normal", "#333333"),
             ("10–50×5–20 px）", 9.5, "normal", "#333333"),
             ("augment-jitter 0.3", 9.5, "normal", "#333333"),
             ("Mosaic", 9.5, "normal", "#333333"),
             ("（close_mosaic=15）", 9.5, "normal", "#333333"),
         ])
    # 3 空频融合
    _box(ax, xs[2], y_stage, stage_w, h_stage, "#EBF5EC", GREEN,
         title="③ 空频融合",
         lines=[
             ("SpatialFrequencyFusion", 10.5, "bold", INK),
             ("h=16, layers=2, SE", 9.5, "normal", "#333333"),
             ("参数量 3,361", 9.5, "bold", "#2E6B34"),
             ("残差 α warmup", 9.5, "normal", "#333333"),
             ("0 → 0.5（10 epochs）", 9.5, "normal", "#333333"),
         ])
    # 4 检测网络
    _box(ax, xs[3], y_stage, stage_w, h_stage, "#FBE9EA", RED,
         title="④ 检测网络",
         lines=[
             ("YOLO11l（25.4M）", 10.5, "bold", INK),
             ("AdamW lr0=0.0008", 9.5, "normal", "#333333"),
             ("Cosine lrf=0.01", 9.5, "normal", "#333333"),
             ("150 epochs · batch 16", 9.5, "normal", "#333333"),
             ("IoU 0.2 · AMP", 9.5, "normal", "#333333"),
             ("早停 patience=20", 9.5, "normal", "#333333"),
         ])
    # 5 后处理
    _box(ax, xs[4], y_stage, stage_w, h_stage, "#F0EDF7", PURPLE,
         title="⑤ 后处理",
         lines=[
             ("标准推理 imgsz=640", 10.5, "bold", INK),
             ("SAHI 512×512", 9.5, "normal", "#333333"),
             ("（overlap=0.2）", 9.5, "normal", "#333333"),
             ("SAHI 评估无效", 9.5, "bold", "#6B5B95"),
             ("（94.63%）", 9.5, "normal", "#333333"),
         ])

    # 阶段间箭头
    for i in range(4):
        _arrow(ax, xs[i] + stage_w + 0.02, y_stage + h_stage / 2,
               xs[i + 1] - 0.02, y_stage + h_stage / 2, color=INK, lw=2.2)

    # 底部：最终结果
    _box(ax, 0.4, 0.55, 19.2, 2.6, "#FFFDF5", BROWN,
         title="最终模型 T16：CRRP + YOLO11l + Fusion（测试集 NSLSR-LWIR）",
         lines=[
             ("test mAP50 = 98.13% ｜ test mAP50-95 = 68.23%",
              13, "bold", "#7A5C2E"),
             ("val mAP50 = 97.85% ｜ 验证-测试差距 Δ = +0.3 ｜ test P = 94.98% ｜ test R = 95.70%",
              10.5, "normal", "#555555"),
         ])
    # 后处理 → 结果 箭头
    _arrow(ax, xs[4] + stage_w / 2, y_stage - 0.02,
           xs[4] + stage_w / 2, 3.2, color=INK, lw=2.2)

    common_style.save_fig(fig, FIGURES_DIR / "fig1_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig2 空频融合模块结构
# ---------------------------------------------------------------------------
def fig2_fusion_module():
    fig, ax = _new_ax((12.5, 13.5), (0, 12.5), (0, 13.5))

    cx = 4.6
    bw, bh = 5.6, 1.05

    # 输入
    _box(ax, cx - bw / 2, 11.9, bw, bh, "#EAF0F8", BLUE,
         title="输入（H×W×3）",
         lines=[("［空间域 ｜ 频率域 ｜ 原始灰度］", 10.5, "bold", INK)])
    # Conv1
    _box(ax, cx - bw / 2, 10.35, bw, bh, "#FDF1E7", ORANGE,
         title="Conv1",
         lines=[("Conv2d(3→16, k=3) + BN + SiLU", 10.5, "bold", INK)])
    # Conv2
    _box(ax, cx - bw / 2, 8.8, bw, bh, "#FDF1E7", ORANGE,
         title="Conv2",
         lines=[("Conv2d(16→16, k=3) + BN + SiLU", 10.5, "bold", INK)])
    # SE 注意力
    _box(ax, cx - bw / 2, 6.55, bw, 1.75, "#EBF5EC", GREEN,
         title="SE 注意力（r=4）",
         lines=[
             ("AdaptiveAvgPool2d(1)", 9.5, "normal", "#333333"),
             ("FC(16→4) → SiLU → FC(4→16) → Sigmoid", 9.5, "bold", "#2E6B34"),
         ])
    # Conv_out
    _box(ax, cx - bw / 2, 5.0, bw, bh, "#FBE9EA", RED,
         title="Conv_out",
         lines=[("Conv2d(16→3, k=3)", 10.5, "bold", INK)])
    # 残差相加
    _box(ax, cx - bw / 2, 3.45, bw, bh, "#F0EDF7", PURPLE,
         title="残差相加",
         lines=[("out = x + α·x（α 可学习）", 10.5, "bold", INK)])
    # 输出
    _box(ax, cx - bw / 2, 1.9, bw, bh, "#FFFDF5", BROWN,
         title="输出（H×W×3）",
         lines=[("3 通道融合图 → YOLO 首层 Conv", 10.5, "bold", "#7A5C2E")])

    # 主链路箭头
    _arrow(ax, cx, 11.9, cx, 11.42, color=INK, lw=2.2)
    _arrow(ax, cx, 10.35, cx, 9.87, color=INK, lw=2.2)
    _arrow(ax, cx, 8.8, cx, 8.32, color=INK, lw=2.2)
    _arrow(ax, cx, 6.55, cx, 6.07, color=INK, lw=2.2)
    _arrow(ax, cx, 5.0, cx, 4.52, color=INK, lw=2.2)
    _arrow(ax, cx, 3.45, cx, 2.97, color=INK, lw=2.2)

    # 残差连接（右侧曲线箭头）
    _arrow(ax, cx + bw / 2 + 0.05, 12.42, cx + bw / 2 + 0.05, 3.97,
           color=GREEN, lw=2.0, ls="--", rad=0.0)
    ax.text(cx + bw / 2 + 0.35, 8.2, "残差连接\nα·x", ha="left", va="center",
            fontsize=10, fontweight="bold", color="#2E6B34")

    # 右侧：参数量与搜索空间
    rx = 10.6
    _box(ax, rx - 1.55, 8.6, 3.1, 4.3, LIGHT_BG, INK,
         title="配置与搜索空间",
         lines=[
             ("参数量 3,361", 10.5, "bold", INK),
             ("（h=16, l=2, SE）", 9.5, "normal", "#555555"),
             ("hidden ∈ {16, 32, 64}", 9.5, "normal", "#333333"),
             ("layers ∈ {2, 3, 4}", 9.5, "normal", "#333333"),
             ("attention ∈ {SE, CBAM}", 9.5, "normal", "#333333"),
             ("α init ∈ [0, 1]", 9.5, "normal", "#333333"),
         ])
    # 残差 warmup 说明
    _box(ax, rx - 1.55, 5.6, 3.1, 2.5, "#EBF5EC", GREEN,
         title="残差 α warmup",
         lines=[
             ("0 → 0.5", 10.5, "bold", "#2E6B34"),
             ("10 epochs 线性上升", 9.5, "normal", "#333333"),
             ("fusion_lr_scale 独立", 9.5, "normal", "#333333"),
         ])

    # 底部：插入位置说明
    _box(ax, 0.4, 0.35, 11.7, 1.05, LIGHT_BG, BLUE,
         lines=[
             ("插入位置：model.model[0] = Sequential(SpatialFrequencyFusion, Conv)",
              10, "normal", "#333333"),
             ("位于 YOLO 首层卷积之前，stride=1 保持分辨率",
              10, "normal", "#333333"),
         ])

    common_style.save_fig(fig, FIGURES_DIR / "fig2_fusion_module.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig10 CRRP 增强流程
# ---------------------------------------------------------------------------
def fig10_crrp():
    fig, ax = _new_ax((16, 9.5), (0, 16), (0, 9.5))

    # 顶部：输入说明
    _box(ax, 0.4, 8.55, 15.2, 0.75, LIGHT_BG, BLUE,
         lines=[
             ("输入：NSLSR-LWIR 训练集（803 张，伪标签：中心坐标 + 估计宽高）",
              11, "bold", INK),
         ])

    # 第一行：步骤 1-3
    y1, h1 = 5.6, 2.5
    w1, g1 = 4.8, 0.4
    x1s = [0.4 + i * (w1 + g1) for i in range(3)]

    _box(ax, x1s[0], y1, w1, h1, "#EAF0F8", BLUE,
         title="① 提取舰船实例",
         lines=[
             ("从训练图像裁剪", 10, "normal", "#333333"),
             ("舰船 patch（8–60 px）", 10, "normal", "#333333"),
             ("保留类别与宽高比", 10, "normal", "#333333"),
         ])
    _box(ax, x1s[1], y1, w1, h1, "#FDF1E7", ORANGE,
         title="② 构建舰船库",
         lines=[
             ("汇总全部实例", 10, "normal", "#333333"),
             ("（≤1000 patches）", 10, "normal", "#333333"),
             ("宽高比 4.0 / 3.0 / 2.0", 10, "normal", "#333333"),
         ])
    _box(ax, x1s[2], y1, w1, h1, "#EBF5EC", GREEN,
         title="③ 旋转 + 缩放",
         lines=[
             ("随机旋转 ±30°", 10, "bold", "#2E6B34"),
             ("缩放至 10–50 × 5–20 px", 10, "normal", "#333333"),
             ("（INTER_LINEAR）", 10, "normal", "#333333"),
         ])

    # 第一行箭头
    _arrow(ax, x1s[0] + w1 + 0.02, y1 + h1 / 2, x1s[1] - 0.02, y1 + h1 / 2, color=INK, lw=2.2)
    _arrow(ax, x1s[1] + w1 + 0.02, y1 + h1 / 2, x1s[2] - 0.02, y1 + h1 / 2, color=INK, lw=2.2)

    # 第二行：步骤 4-6
    y2, h2 = 2.6, 2.5
    x2s = [0.4 + i * (w1 + g1) for i in range(3)]

    _box(ax, x2s[0], y2, w1, h2, "#FBE9EA", RED,
         title="④ 随机粘贴",
         lines=[
             ("随机位置粘贴", 10, "normal", "#333333"),
             ("边缘混合 0.85/0.15", 10, "bold", "#A03A40"),
             ("（alpha 混合）", 10, "normal", "#333333"),
         ])
    _box(ax, x2s[1], y2, w1, h2, "#F0EDF7", PURPLE,
         title="⑤ 更新标签",
         lines=[
             ("新实例加入标签", 10, "normal", "#333333"),
             ("YOLO 格式：", 10, "normal", "#333333"),
             ("class cx cy w h", 10, "bold", "#6B5B95"),
         ])
    _box(ax, x2s[2], y2, w1, h2, "#FFFDF5", BROWN,
         title="⑥ 输出增强集",
         lines=[
             ("2× 离线增强", 10, "bold", "#7A5C2E"),
             ("copies=2 · max_instances=2", 9.5, "normal", "#333333"),
             ("control_ratio=0.2 · seed=42", 9.5, "normal", "#333333"),
         ])

    # 第二行箭头 + 行间箭头
    _arrow(ax, x2s[0] + w1 + 0.02, y2 + h2 / 2, x2s[1] - 0.02, y2 + h2 / 2, color=INK, lw=2.2)
    _arrow(ax, x2s[1] + w1 + 0.02, y2 + h2 / 2, x2s[2] - 0.02, y2 + h2 / 2, color=INK, lw=2.2)
    _arrow(ax, x1s[2] + w1 / 2, y1 - 0.02, x2s[2] + w1 / 2, y2 + h2 + 0.02, color=INK, lw=2.2)

    # 底部：结果
    _box(ax, 0.4, 0.35, 15.2, 1.75, "#FFFDF5", BROWN,
         title="T16 实测结果（CRRP + YOLO11l + Fusion）",
         lines=[
             ("test mAP50 = 98.13% ｜ test mAP50-95 = 68.23% ｜ val mAP50 = 97.85% ｜ Δ = +0.3",
              11.5, "bold", "#7A5C2E"),
         ])
    _arrow(ax, x2s[2] + w1 / 2, y2 - 0.02, x2s[2] + w1 / 2, 2.15, color=INK, lw=2.2)

    common_style.save_fig(fig, FIGURES_DIR / "fig10_crrp.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig1_pipeline()
    fig2_fusion_module()
    fig10_crrp()
    print("3 张架构示意图已生成到", FIGURES_DIR)
