# -*- coding: utf-8 -*-
"""无头（offscreen）GUI 截图生成 — 论文「软件设计与使用」章节配图。

在 ship_detector/ 目录内以顶层模块导入方式运行（与 main.py 一致，core/ui/config
均按顶层包解析）。加载最佳 large 模型（YOLO11l + 空频融合，即论文 §6 最佳 T5 =
T5_yolo11l_fusion），对以下两类素材渲染主窗口并截图：

  1. dataset/ 下的真实 MP4 —— 定位抽帧 + SAHI 检测，渲染带置信度的检测框；
  2. data/weights_staging/frames/val 下的 3 张 NSLSR 真实验证帧（原有行为，保留）。

另存一张空态（未检测）主界面截图，并刷新 README 首页配图。

用法:
  cd ship_detector
  QT_QPA_PLATFORM=offscreen uv run --project /home/ws/code/ship python render_gui_screenshots.py

可选参数:
  --videos a.mp4 b.mp4   只渲染指定视频（默认 dataset/ 下全部 *.mp4）
  --no-stills            跳过 NSLSR 静态验证帧
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("render_gui")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_DEVICE_PIXEL_RATIO", "1")

REPO = Path(__file__).resolve().parent.parent  # 仓库根 (ship/)
SHIP_DETECTOR_DIR = REPO / "ship_detector"
WEIGHTS_DIR = REPO / "data" / "weights_staging" / "weights"
FRAMES_DIR = REPO / "data" / "weights_staging" / "frames" / "val"
DATASET_DIR = REPO / "dataset"
OUT_DIR = REPO / "docs" / "论文" / "figures" / "software"
README_SHOT = SHIP_DETECTOR_DIR / "resources" / "screenshot.png"

BEST_MODEL_GLOB = "T5_yolo11l_fusion_best.pt"

VIDEO_CONF = 0.25
STILL_CONF = 0.25

VAL_FRAMES = ["val_batch0_pred_101_clean.jpg",
              "val_batch1_pred_101_clean.jpg",
              "val_batch2_pred_101_clean.jpg"]

VAL_DISPLAY = {
    "val_batch0_pred_101_clean.jpg": "NSLSR 验证帧 val_batch0",
    "val_batch1_pred_101_clean.jpg": "NSLSR 验证帧 val_batch1",
    "val_batch2_pred_101_clean.jpg": "NSLSR 验证帧 val_batch2",
}


def pick_screenshot_frames(total_frames: int, count: int = 3) -> list[int]:
    """在整段视频上均匀挑 count 个抽帧索引。

    始终包含第 0 帧；索引唯一、升序、且落在 [0, total_frames) 内。
    空片或非法参数返回空列表。
    """
    total = int(total_frames)
    count = int(count)
    if total <= 0 or count <= 0:
        return []
    if total == 1 or count == 1:
        return [0]

    n = min(count, total)
    step = (total - 1) / (n - 1)
    picks = {int(round(i * step)) for i in range(n)}
    return sorted({0} | {p for p in picks if 0 <= p < total})


def _load_bgr(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def render_video_frames(win, app, detector, video_path, frame_indices,
                        conf, out_dir, prefix) -> list[Path]:
    """定位抽帧 -> 检测 -> 渲染 -> 截图。返回实际写出的 PNG 路径列表。

    这里刻意不用 QTimer 播放: 离屏环境下推理是阻塞的, 播放时钟会严重漂移,
    抽帧与目标帧对不上。直接 seek 到目标帧才能确定性复现。
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        logger.warning("无法打开视频: %s", video_path)
        return made

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                logger.warning("抽帧读取失败: %s frame=%s", video_path, idx)
                continue
            try:
                ships = detector.detect_frame(frame, conf=conf)
            except Exception as exc:  # noqa: BLE001 — 单帧失败不拖垮整段
                logger.error("检测失败 %s frame=%s: %s", video_path, idx, exc)
                continue

            win.canvas.set_frame(frame)
            win.canvas.set_ships(ships)
            app.processEvents()

            png = out_dir / f"{prefix}_f{int(idx):05d}.png"
            if win.grab().save(str(png)):
                made.append(png)
                logger.info("已渲染 %s (frame=%s, 目标数=%d)", png.name, idx, len(ships))
    finally:
        cap.release()

    return made


def _video_frame_count(video_path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _detection_aware_frames(detector, video_path, total, count, conf) -> list[int]:
    """挑 count 个「更可能有检出」的抽帧索引，用于配图。

    把扫描点切成 count 个桶，每桶取置信度最高的那一帧；桶内没有检出时退回该桶
    的中间帧。这样既尽量让配图体现识别能力，又不会假装每帧都有目标。
    """
    import cv2

    scan = max(count * 8, 24)
    cands = pick_screenshot_frames(total, count=scan)
    if not cands:
        return []

    best: dict[int, float] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return pick_screenshot_frames(total, count=count)
    try:
        for idx in cands:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            ships = detector.detect_frame(frame, conf=conf)
            best[idx] = max((s.confidence for s in ships), default=0.0)
    finally:
        cap.release()

    n = len(cands)
    picks = []
    for b in range(count):
        lo = b * n // count
        hi = max(lo + 1, (b + 1) * n // count)
        bucket = cands[lo:hi]
        if not bucket:
            continue
        picks.append(max(bucket, key=lambda i: (best.get(i, 0.0), -i)))
    return sorted(set(picks))


def _render_video_screenshots(win, app, detector, videos, out_dir) -> list[Path]:
    """遍历视频，对每段抽帧检测并截图。"""
    made: list[Path] = []
    for vpath in videos:
        vpath = Path(vpath)
        if not vpath.exists():
            logger.warning("跳过缺失视频: %s", vpath)
            continue

        total = _video_frame_count(vpath)
        if total <= 0:
            logger.warning("视频无可读帧: %s", vpath)
            continue

        indices = _detection_aware_frames(detector, vpath, total, count=3, conf=VIDEO_CONF)
        if not indices:
            logger.warning("无可用抽帧: %s", vpath)
            continue

        # 抽帧位置可能随参数变化, 先清掉该视频上一轮留下的截图, 避免孤儿图
        prefix = f"gui_{vpath.stem}"
        for stale in out_dir.glob(f"{prefix}_f*.png"):
            stale.unlink()

        shots = render_video_frames(
            win=win, app=app, detector=detector,
            video_path=str(vpath), frame_indices=indices,
            conf=VIDEO_CONF, out_dir=out_dir, prefix=prefix,
        )
        made.extend(shots)
        logger.info("%s: 抽帧 %s -> 截图 %d 张", vpath.name, indices, len(shots))

    return made


def _render_still_frames(win, app, detector, out_dir) -> list[Path]:
    """原有行为：3 张 NSLSR 验证帧 + 空态主界面。"""
    made: list[Path] = []
    for name in VAL_FRAMES:
        fpath = FRAMES_DIR / name
        if not fpath.exists():
            logger.warning("缺少验证帧: %s", fpath)
            continue
        frame = _load_bgr(fpath)
        ships = detector.detect_frame(frame, conf=STILL_CONF)
        win.canvas.load_image(str(fpath))
        win.canvas.set_ships(ships)
        app.processEvents()
        png = out_dir / f"gui_{Path(name).stem}_detected.png"
        if win.grab().save(str(png)):
            made.append(png)
        logger.info("已保存: %s (%d 目标)", png.name, len(ships))

    if (FRAMES_DIR / VAL_FRAMES[0]).exists():
        win.canvas.set_ships([])
        win.canvas.load_image(str(FRAMES_DIR / VAL_FRAMES[0]))
        app.processEvents()
        empty = out_dir / "gui_main_empty.png"
        if win.grab().save(str(empty)):
            made.append(empty)

    return made


def _render_readme_shot(win, app, detector, video_path) -> Path | None:
    """渲染一张最具代表性的主界面图，用作 README 头图。"""
    import cv2

    total = _video_frame_count(video_path)
    if total <= 0:
        logger.warning("README 头图: 无法读取 %s", video_path)
        return None

    best_idx, best_ships = 0, []
    for idx in pick_screenshot_frames(total, count=7):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        ships = detector.detect_frame(frame, conf=VIDEO_CONF)
        if sum(s.confidence for s in ships) > sum(s.confidence for s in best_ships):
            best_idx, best_ships = idx, ships

    if not best_ships:
        logger.warning("README 头图: 未找到含目标的帧，保留原图")
        return None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    win.canvas.set_frame(frame)
    win.canvas.set_ships(best_ships)
    win.canvas.status_message.emit(f"视频: {Path(video_path).name}  第 {best_idx} 帧")
    app.processEvents()
    README_SHOT.parent.mkdir(parents=True, exist_ok=True)
    if win.grab().save(str(README_SHOT)):
        logger.info("README 头图已更新: %s (frame=%s, %d 目标)",
                    README_SHOT, best_idx, len(best_ships))
        return README_SHOT
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="离屏渲染舰船识别系统界面截图")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="只渲染指定视频（默认 dataset/ 下全部 *.mp4）")
    parser.add_argument("--no-stills", action="store_true", help="跳过 NSLSR 静态帧")
    parser.add_argument("--preprocess", default="raw",
                        choices=["raw", "tophat", "butterworth", "dual", "wavelet", "rpca"],
                        help="训练域对齐预处理流水线（灰度归一化恒开）")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(SHIP_DETECTOR_DIR))
    os.chdir(str(SHIP_DETECTOR_DIR))

    from PySide6.QtWidgets import QApplication

    from core.yolo_detector import YoloDetector
    from ui.main_window import MainWindow

    app = QApplication([])
    app.setStyle("Fusion")

    cands = sorted(WEIGHTS_DIR.glob(BEST_MODEL_GLOB))
    if not cands:
        logger.error("未找到最佳模型: %s", WEIGHTS_DIR / BEST_MODEL_GLOB)
        return 1
    best = cands[0]

    detector = YoloDetector()
    if not detector.load_model(str(best)):
        logger.error("最佳模型加载失败: %s", best)
        return 1
    detector.set_preprocess(args.preprocess)
    detector.set_sahi(True)
    logger.info("输入预处理: %s (灰度归一化恒开)", args.preprocess)

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()

    win.detector = detector
    win.canvas.set_ships([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = args.videos if args.videos else sorted(DATASET_DIR.glob("*.mp4"))
    if videos:
        shots = _render_video_screenshots(win, app, detector, videos, OUT_DIR)
        logger.info("视频截图完成: %d 张", len(shots))
    else:
        logger.warning("dataset/ 下未找到 mp4，跳过视频截图")

    if not args.no_stills:
        _render_still_frames(win, app, detector, OUT_DIR)

    if videos:
        _render_readme_shot(win, app, detector, videos[-1])

    logger.info("截图完成 -> %s", OUT_DIR)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())