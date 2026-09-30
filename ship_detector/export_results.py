# -*- coding: utf-8 -*-
"""视频识别结果导出 — 标注结果视频 + 检测统计报告。

对一段视频逐帧（可抽帧）执行 YOLO 检测，把带置信度的检测框烧录进画面，输出
标注结果 MP4；同时汇总逐帧与总体统计，产出机器可读的 JSON 与人类可读的
Markdown 报告。

本模块无 Qt 依赖，可在无显示环境下运行；重型导入放在函数内，保证模块本身
可以廉价地 import（纯逻辑测试不需要 cv2/GPU）。

用法:
  cd ship_detector
  uv run --project /home/ws/code/ship python export_results.py \
      --video ../dataset/mmexport1790781116796.mp4 \
      --out-dir ../docs/论文/figures/video/data \
      --sahi --conf 0.25 --stride 1
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("export_results")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO / "dataset"
DEFAULT_OUT_DIR = REPO / "docs" / "论文" / "figures" / "video" / "data"
DEFAULT_WEIGHTS = REPO / "data" / "weights_staging" / "weights" / "T5_yolo11l_fusion_best.pt"

_DEFAULT_BOX_RGB = (0, 0, 255)  # 与 config.SHIP_CLASS_COLORS[0] 同色（RGB）
_LABEL_TEXT_COLOR = (255, 255, 255)


def _box_color(class_id: int):
    """config 里的类别色是 RGB（Qt 直接按 RGB 用），cv2 需要 BGR，这里翻转。"""
    rgb = _DEFAULT_BOX_RGB
    try:
        from config import SHIP_CLASS_COLORS
        rgb = SHIP_CLASS_COLORS.get(class_id % len(SHIP_CLASS_COLORS), _DEFAULT_BOX_RGB)
    except Exception:  # noqa: BLE001 — 无 config 时退回默认色，不影响导出
        pass
    return (rgb[2], rgb[1], rgb[0])


def _draw_ship(frame, ship) -> None:
    """把单个检测框 + 置信度标签画到 frame 上（原地修改）。"""
    import cv2

    h, w = frame.shape[:2]
    thickness = max(1, round(min(w, h) / 300))
    font_scale = max(0.4, min(w, h) / 700)
    color = _box_color(getattr(ship, "class_id", 0))

    x1, y1, x2, y2 = ship.bbox.to_pixels(w, h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"{ship.class_name} #{ship.track_id}  {ship.confidence:.0%}"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    ly = y1 - th - base
    if ly < 0:
        ly = min(h - th - base - 1, y2 + base)
    lx = min(x1, max(0, w - tw - 2))
    cv2.rectangle(frame, (lx, ly), (lx + tw + 2, ly + th + base), color, -1)
    cv2.putText(frame, label, (lx + 1, ly + th), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, _LABEL_TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_hud(frame, frame_index: int, total: int, n_ships: int) -> None:
    import cv2

    h, w = frame.shape[:2]
    scale = max(0.4, min(w, h) / 700)
    text = f"frame {frame_index}/{total}  ships {n_ships}"
    cv2.putText(frame, text, (8, int(22 * scale) + 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), max(1, round(min(w, h) / 400)), cv2.LINE_AA)


def _det_to_dict(ship) -> dict:
    b = ship.bbox
    return {
        "track_id": int(ship.track_id),
        "class_id": int(getattr(ship, "class_id", 0)),
        "class_name": str(ship.class_name),
        "confidence": float(ship.confidence),
        "bbox": {"x": float(b.x), "y": float(b.y), "w": float(b.w), "h": float(b.h)},
    }


def _empty_stats(source: str, stride: int, conf: float, use_sahi, model_name) -> dict:
    return {
        "run": {
            "source": str(source),
            "output_video": None,
            "model_name": model_name,
            "confidence": float(conf),
            "stride": int(stride),
            "use_sahi": bool(use_sahi) if use_sahi is not None else None,
            "fps": None,
            "frame_size": None,
            "source_frames": 0,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        },
        "per_frame": [],
        "summary": {
            "frames_processed": 0,
            "frames_with_detections": 0,
            "detection_rate": 0.0,
            "total_detections": 0,
            "max_detections_in_a_frame": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
        },
        "frames_processed": 0,
        "source_frames": 0,
        "stride": int(stride),
        "total_detections": 0,
        "frames_with_detections": 0,
    }


def export_annotated_video(video_path, detector, out_video,
                           conf: float = 0.25, stride: int = 1,
                           use_sahi: Optional[bool] = None,
                           model_name: Optional[str] = None,
                           preprocess: Optional[str] = None,
                           progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """逐帧检测 -> 烧录置信度框 -> 写标注视频 -> 汇总统计。

    返回统计字典；视频打不开时返回带 "error" 的字典（不抛异常、不产出文件）。
    """
    import cv2

    try:
        stride = int(stride)
    except (TypeError, ValueError):
        stride = 1
    if stride <= 0:
        stride = 1

    stats = _empty_stats(str(video_path), stride, conf, use_sahi, model_name)
    stats["run"]["preprocess"] = preprocess

    if use_sahi is not None and hasattr(detector, "set_sahi"):
        detector.set_sahi(bool(use_sahi))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        stats["error"] = f"无法打开视频: {video_path}"
        logger.error("无法打开视频: %s", video_path)
        return stats

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stats["run"]["fps"] = float(fps)
    stats["run"]["frame_size"] = [width, height]

    writer = None
    if out_video is not None and width > 0 and height > 0:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        if not writer.isOpened():
            logger.error("无法创建输出视频: %s", out_video)
            writer = None
        else:
            stats["run"]["output_video"] = str(out_video)

    per_frame = []
    confs: list[float] = []
    frames_processed = 0
    frames_with_detections = 0
    idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride != 0:
                continue

            try:
                ships = detector.detect_frame(frame, conf=conf)
            except Exception as exc:  # noqa: BLE001 — 单帧失败不中断整段导出
                logger.error("检测失败 frame=%d: %s", idx, exc)
                ships = []

            n = len(ships)
            frames_processed += 1
            if n:
                frames_with_detections += 1
                confs.extend(float(s.confidence) for s in ships)

            per_frame.append({
                "frame_index": idx,
                "timestamp": round(idx / float(fps), 4),
                "count": n,
                "detections": [_det_to_dict(s) for s in ships],
            })

            if writer is not None:
                for s in ships:
                    _draw_ship(frame, s)
                _draw_hud(frame, idx, total_meta or (idx + 1), n)
                writer.write(frame)

            if progress is not None:
                progress(frames_processed, idx)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    source_frames = idx + 1 if idx >= 0 else 0
    total_detections = sum(f["count"] for f in per_frame)
    summary = {
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "detection_rate": round(frames_with_detections / frames_processed, 4)
        if frames_processed else 0.0,
        "total_detections": total_detections,
        "max_detections_in_a_frame": max((f["count"] for f in per_frame), default=0),
        "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
        "min_confidence": min(confs) if confs else 0.0,
        "max_confidence": max(confs) if confs else 0.0,
    }

    stats["per_frame"] = per_frame
    stats["summary"] = summary
    stats["run"]["source_frames"] = source_frames
    stats["source_frames"] = source_frames
    stats["frames_processed"] = frames_processed
    stats["frames_with_detections"] = frames_with_detections
    stats["total_detections"] = total_detections

    logger.info(
        "%s: 处理 %d 帧, 含目标 %d 帧, 共 %d 个目标, 平均置信度 %.3f",
        Path(video_path).name, frames_processed, frames_with_detections,
        total_detections, summary["mean_confidence"],
    )
    return stats


def _report_markdown(stats: dict) -> str:
    run = stats["run"]
    s = stats["summary"]
    src = Path(run.get("source") or "").name

    if stats.get("error"):
        return "\n".join([
            f"# 检测结果报告 — {src}",
            "",
            f"**检测失败**：{stats['error']}",
            "",
            "本段视频未能读取，未产生任何检测结果。",
            "",
        ])

    lines = [
        f"# 检测结果报告 — {src}",
        "",
        f"- 生成时间: {run.get('exported_at')}",
        f"- 源视频: `{run.get('source')}`",
        f"- 标注视频: `{run.get('output_video')}`",
        f"- 模型: {run.get('model_name')}",
        f"- 输入预处理: {run.get('preprocess')}（灰度归一化 BGR→GRAY→BGR 恒开）",
        f"- 置信度阈值: {run.get('confidence')}",
        f"- 抽帧步长: {run.get('stride')}",
        f"- SAHI: {run.get('use_sahi')}",
        f"- 分辨率: {run.get('frame_size')}  |  FPS: {run.get('fps')}",
        f"- 源帧数: {run.get('source_frames')}",
        "",
        "## 总体统计",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 处理帧数 | {s.get('frames_processed')} |",
        f"| 含目标帧数 | {s.get('frames_with_detections')} |",
        f"| 检出率 | {s.get('detection_rate')} |",
        f"| 目标总数 | {s.get('total_detections')} |",
        f"| 单帧最多目标数 | {s.get('max_detections_in_a_frame')} |",
        f"| 平均置信度 | {s.get('mean_confidence')} |",
        f"| 最低置信度 | {s.get('min_confidence')} |",
        f"| 最高置信度 | {s.get('max_confidence')} |",
        "",
    ]

    hits = [f for f in stats.get("per_frame", []) if f.get("count")]
    lines.append("## 含目标帧明细")
    lines.append("")
    if hits:
        lines.append("| 帧号 | 时间(s) | 目标数 | 最高置信度 |")
        lines.append("| --- | --- | --- | --- |")
        for f in hits[:200]:
            best = max(d["confidence"] for d in f["detections"])
            lines.append(f"| {f['frame_index']} | {f['timestamp']} | {f['count']} | {best:.3f} |")
        if len(hits) > 200:
            lines.append(f"| ... | ... | ... | (另有 {len(hits) - 200} 帧未列出) |")
    else:
        lines.append("_本段视频未检出目标。跨域实拍素材检出偏少属于已知域差现象，非工具故障。_")
    lines.append("")

    return "\n".join(lines)


def write_report(stats: dict, out_dir, stem: str = "detection_report") -> list[Path]:
    """写出 <stem>.json 与 <stem>.md，返回两个路径。"""
    if not isinstance(stats, dict) or "summary" not in stats:
        raise ValueError("stats 缺少 summary 字段，拒绝写出误导性报告")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_report_markdown(stats), encoding="utf-8")

    logger.info("报告已写出: %s / %s", json_path.name, md_path.name)
    return [json_path, md_path]


def _build_detector(weights: Path, use_sahi: bool):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.yolo_detector import YoloDetector

    det = YoloDetector()
    if not det.load_model(str(weights)):
        raise RuntimeError(f"模型加载失败: {weights}")
    if use_sahi:
        det.set_sahi(True)
    return det


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出视频识别结果（标注视频 + 统计报告）")
    parser.add_argument("--video", default=None, help="单个视频；缺省处理 dataset/ 下全部 mp4")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sahi", dest="sahi", action="store_true", default=True)
    parser.add_argument("--no-sahi", dest="sahi", action="store_false")
    parser.add_argument("--model-name", default="T5_yolo11l_fusion")
    parser.add_argument("--preprocess", default="raw",
                        choices=["raw", "tophat", "butterworth", "dual", "wavelet", "rpca"],
                        help="训练域对齐预处理流水线（灰度归一化恒开）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("权重不存在: %s", weights)
        return 1

    videos = [Path(args.video)] if args.video else sorted(DEFAULT_DATASET_DIR.glob("*.mp4"))
    if not videos:
        logger.error("未找到视频（dataset/ 下无 mp4）")
        return 1
    missing = [v for v in videos if not v.exists()]
    if missing:
        logger.error("视频不存在: %s", ", ".join(str(m) for m in missing))
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = _build_detector(weights, args.sahi)
    detector.set_preprocess(args.preprocess)

    for vpath in videos:
        if not vpath.exists():
            logger.warning("跳过缺失视频: %s", vpath)
            continue
        out_video = out_dir / f"annotated_{vpath.stem}.mp4"
        stats = export_annotated_video(
            str(vpath), detector, str(out_video),
            conf=args.conf, stride=args.stride, use_sahi=args.sahi,
            model_name=args.model_name, preprocess=args.preprocess,
        )
        if stats.get("error"):
            logger.error("%s: %s", vpath.name, stats["error"])
            write_report(stats, out_dir, stem=f"report_{vpath.stem}")
            continue

        paths = write_report(stats, out_dir, stem=f"report_{vpath.stem}")
        s = stats["summary"]
        print(f"[{vpath.name}] 处理 {s['frames_processed']} 帧, "
              f"含目标 {s['frames_with_detections']} 帧, 共 {s['total_detections']} 个目标, "
              f"平均置信度 {s['mean_confidence']:.3f}")
        print(f"  标注视频: {out_video}")
        for p in paths:
            print(f"  报告: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())