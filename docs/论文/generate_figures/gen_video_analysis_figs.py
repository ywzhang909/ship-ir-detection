# -*- coding: utf-8 -*-
"""视频识别分析图生成（gen_video_analysis_figs.py）。

读取 ship_detector/export_results.py 产出的检测统计 JSON（每视频一份），
生成视频识别分析报告所需的「分析图片」：

    video_v1_detection_overview.png    两段视频的抽帧检测总览（带置信度框）
    video_v2_confidence_distribution.png  置信度分布直方图
    video_v3_detection_timeline.png     逐帧检出数 / 置信度时间轴
    video_v4_contrast_analysis.png      原图 vs CLAHE 增强（解释域差与漏检成因）

全部复用 common_style 保证图风一致、中文无豆腐块。

用法（在 docs/论文/ 下执行）:
    uv run --project /home/ws/code/ship python generate_figures/gen_video_analysis_figs.py \
        --stats ../output/video_detection/report_A.json ../output/video_detection/report_B.json

无 stats 时脚本打印用法并以非 0 退出，不会产出坏图。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402

import common_style  # noqa: E402  必须在 pyplot 之前导入以注册 CJK 字体
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = common_style.PALETTE


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------


def load_stats(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" not in payload:
        raise ValueError(f"{path} 缺少 summary 字段，不是合法的检测统计文件")
    payload.setdefault("per_frame", [])
    payload.setdefault("run", {})
    return payload


def label_of(stats: dict) -> str:
    """用文件名（视频名）作为图例标签。"""
    src = stats.get("run", {}).get("source") or ""
    return Path(src).stem or "clip"


def all_confidences(stats: dict) -> list[float]:
    return [
        float(d["confidence"])
        for fr in stats.get("per_frame", [])
        for d in fr.get("detections", [])
    ]


def sampled_indices(stats: dict, k: int = 4) -> list[int]:
    """从 per_frame 里挑 k 个均匀分布的帧号（优先挑有检出的帧）。"""
    pf = stats.get("per_frame", [])
    if not pf:
        return []
    with_det = [fr["frame_index"] for fr in pf if fr.get("count", 0) > 0]
    pool = with_det if with_det else [fr["frame_index"] for fr in pf]
    if len(pool) <= k:
        return pool
    step = (len(pool) - 1) / (k - 1)
    return [pool[int(round(i * step))] for i in range(k)]


def read_frame(video: str | Path, index: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def enhance_clahe(frame: np.ndarray) -> np.ndarray:
    """灰度 + CLAHE，用于展示暗弱小目标的可分离性。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def displayable(frame: np.ndarray, gain: float = 2.6) -> np.ndarray:
    """把极暗的 IR 帧提亮到肉眼可辨，仅用于出图。"""
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 图 1：抽帧检测总览
# ---------------------------------------------------------------------------


def fig_detection_overview(stats_list: list[dict], out_dir: Path) -> Path:
    rows = len(stats_list)
    cols = 4
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 2.5 * rows), squeeze=False
    )

    for r, stats in enumerate(stats_list):
        name = label_of(stats)
        ann = stats.get("run", {}).get("output_video")
        src = stats.get("run", {}).get("source")
        # 优先读标注视频（框已烧录进去），没有则退回原视频
        video = ann if ann and Path(ann).exists() else src
        idxs = sampled_indices(stats, cols)
        pf_by_idx = {fr["frame_index"]: fr for fr in stats.get("per_frame", [])}

        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(idxs) or not video:
                ax.axis("off")
                continue
            idx = idxs[c]
            frame = read_frame(video, idx)
            if frame is None:
                ax.axis("off")
                continue
            ax.imshow(displayable(frame))
            rec = pf_by_idx.get(idx, {})
            n = rec.get("count", 0)
            best = max(
                (d["confidence"] for d in rec.get("detections", [])), default=None
            )
            title = f"帧 {idx}  检出 {n}"
            if best is not None:
                title += f"  最高置信度 {best:.0%}"
            ax.set_title(title, fontsize=10, color=PAL[3] if n else "#666666")
            for spine in ax.spines.values():
                spine.set_edgecolor(PAL[3] if n else "#cccccc")
                spine.set_linewidth(2.0 if n else 1.0)

        axes[r][0].set_ylabel(
            f"{name}\n{stats.get('run', {}).get('frame_size', ['', ''])}",
            fontsize=11,
            rotation=0,
            ha="right",
            va="center",
            labelpad=26,
        )

    fig.suptitle("图 V1  视频抽帧检测总览（蓝框=检出舰船，标题含最高置信度）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return common_style.save_fig(fig, out_dir / "video_v1_detection_overview.png")


# ---------------------------------------------------------------------------
# 图 2：置信度分布
# ---------------------------------------------------------------------------


def fig_confidence_distribution(stats_list: list[dict], out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for i, stats in enumerate(stats_list):
        confs = all_confidences(stats)
        name = label_of(stats)
        color = PAL[i % len(PAL)]

        ax1.hist(confs, bins=20, range=(0, 1), alpha=0.65, label=f"{name} (n={len(confs)})",
                 color=color, edgecolor="white")
        if confs:
            ax1.axvline(float(np.mean(confs)), color=color, linestyle="--", linewidth=1.6,
                        label=f"{name} 均值 {np.mean(confs):.2f}")

        pf = [fr for fr in stats.get("per_frame", []) if fr.get("count", 0) > 0]
        if pf:
            xs = [fr["frame_index"] for fr in pf]
            best = [max(d["confidence"] for d in fr["detections"]) for fr in pf]
            ax2.plot(xs, best, linewidth=1.3, color=color, label=name)

    ax1.set_xlabel("置信度")
    ax1.set_ylabel("检出框数量")
    ax1.set_title("检出置信度分布")
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, linestyle=":")

    ax2.set_xlabel("帧号")
    ax2.set_ylabel("该帧最高置信度")
    ax2.set_title("逐帧最高置信度走势（有检出的帧）")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linestyle=":")

    fig.suptitle("图 V2  检出置信度分布与逐帧走势", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return common_style.save_fig(fig, out_dir / "video_v2_confidence_distribution.png")


# ---------------------------------------------------------------------------
# 图 3：检出时间轴 / 检出率
# ---------------------------------------------------------------------------


def fig_detection_timeline(stats_list: list[dict], out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    for i, stats in enumerate(stats_list):
        color = PAL[i % len(PAL)]
        name = label_of(stats)
        pf = stats.get("per_frame", [])
        xs = [fr["frame_index"] for fr in pf]
        ys = [fr.get("count", 0) for fr in pf]
        fps = stats.get("run", {}).get("fps") or 30.0

        ax1.plot(xs, ys, linewidth=1.4, color=color, label=name)
        ax2.plot([x / fps for x in xs], ys, linewidth=1.4, color=color, label=name)

        summ = stats.get("summary", {})
        rate = summ.get("detection_rate")
        if rate is not None:
            ax1.annotate(
                f"检出率 {rate:.0%}",
                xy=(xs[-1] if xs else 0, ys[-1] if ys else 0),
                xytext=(-6, 10),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=color,
            )

    ax1.set_ylabel("该帧检出舰船数")
    ax1.set_title("逐帧检出数（按帧号）")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, linestyle=":")

    ax2.set_xlabel("时间（秒）")
    ax2.set_ylabel("该帧检出舰船数")
    ax2.set_title("逐帧检出数（按时间）")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linestyle=":")

    fig.suptitle("图 V3  检测结果时间轴", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return common_style.save_fig(fig, out_dir / "video_v3_detection_timeline.png")


# ---------------------------------------------------------------------------
# 图 4：原图 vs CLAHE 增强（域差分析）
# ---------------------------------------------------------------------------


def fig_contrast_analysis(stats_list: list[dict], out_dir: Path) -> Path:
    picks: list[tuple[str, np.ndarray, float]] = []
    for stats in stats_list:
        src = stats.get("run", {}).get("source")
        idxs = sampled_indices(stats, 2)
        if not src or not Path(src).exists():
            continue
        for idx in idxs[:2]:
            frame = read_frame(src, idx)
            if frame is None:
                continue
            rec = next(
                (fr for fr in stats.get("per_frame", []) if fr["frame_index"] == idx), {}
            )
            best = max(
                (d["confidence"] for d in rec.get("detections", [])), default=0.0
            )
            picks.append((f"{label_of(stats)} 帧{idx}\n最高置信度 {best:.0%}", frame, best))

    if not picks:
        raise RuntimeError("没有可用的源视频帧，无法生成对比图")

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.0), squeeze=False)

    for c, (title, frame, best) in enumerate(picks):
        axes[0][c].imshow(displayable(frame, 2.6))
        axes[0][c].set_title(f"原图（提亮显示）\n{title}", fontsize=9)
        axes[1][c].imshow(enhance_clahe(frame))
        axes[1][c].set_title("CLAHE 增强（灰度均衡）", fontsize=9)
        for r in (0, 1):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

    fig.suptitle(
        "图 V4  原图与 CLAHE 增强对比：解释远距离暗弱小目标为何难以检出",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return common_style.save_fig(fig, out_dir / "video_v4_contrast_analysis.png")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="生成视频识别分析图")
    ap.add_argument(
        "--stats",
        nargs="+",
        required=True,
        help="export_results.py 产出的检测统计 JSON（每视频一份）",
    )
    ap.add_argument(
        "--out-dir",
        default=str(PAPER_DIR / "figures" / "video"),
        help="图片输出目录，默认 docs/论文/figures/video/",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.stats]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[错误] 找不到统计文件: {[str(m) for m in missing]}", file=sys.stderr)
        print("请先运行 ship_detector/export_results.py 生成统计 JSON。", file=sys.stderr)
        return 1

    stats_list = [load_stats(p) for p in paths]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in (
        fig_detection_overview,
        fig_confidence_distribution,
        fig_detection_timeline,
        fig_contrast_analysis,
    ):
        try:
            print("[生成]", fn(stats_list, out_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {fn.__name__}: {exc}", file=sys.stderr)
            raise

    print(f"[完成] 分析图输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())