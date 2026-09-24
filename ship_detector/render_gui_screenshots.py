"""无头（offscreen）GUI 截图生成 — 论文「软件设计与使用」章节配图。

在 ship_detector/ 目录内以顶层模块导入方式运行（与 main.py 一致，core/ui/config
均按顶层包解析）。加载最佳 large 模型（YOLO11l + 空频融合，即 论文 §6 最佳 T5 =
T5_yolo11l_fusion），对 3 张 NSLSR 真实验证帧执行标准检测，渲染主窗口并逐帧截图，
另存一张空态（未检测）主界面截图。

用法:
  cd ship_detector
  QT_QPA_PLATFORM=offscreen uv run ../render_dir_gui_screenshots.py
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("render_gui")

# 必须在使用任何 PySide6 QtWidgets 之前设置（保证无显示环境可跑）
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_DEVICE_PIXEL_RATIO", "1")

REPO = Path(__file__).resolve().parent.parent.parent  # 仓库根 (ship/)
SHIP_DETECTOR_DIR = REPO / "ship_detector"
WEIGHTS_DIR = REPO / "data" / "weights_staging" / "weights"
FRAMES_DIR = REPO / "data" / "weights_staging" / "frames" / "val"
OUT_DIR = REPO / "docs" / "论文" / "figures" / "software"

# 最佳 large 模型：T5_yolo11l_fusion（空频融合）—— 全实验最佳 large（mAP50 98.13%）
BEST_MODEL_GLOB = "T5_yolo11l_fusion_best.pt"

VAL_FRAMES = ["val_batch0_pred_101_clean.jpg",
              "val_batch1_pred_101_clean.jpg",
              "val_batch2_pred_101_clean.jpg"]

# 数据源显示名（论文配图说明用）
VAL_DISPLAY = {
    "val_batch0_pred_101_clean.jpg": "NSLSR 验证帧 val_batch0",
    "val_batch1_pred_101_clean.jpg": "NSLSR 验证帧 val_batch1",
    "val_batch2_pred_101_clean.jpg": "NSLSR 验证帧 val_batch2",
}


def _load_bgr(path: Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def main():
    # 修正 cwd + 顶层 import 环境（与 main.py 运行时一致）
    sys.path.insert(0, str(SHIP_DETECTOR_DIR))
    os.chdir(str(SHIP_DETECTOR_DIR))

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    from core.yolo_detector import YoloDetector
    from ui.main_window import MainWindow

    app = QApplication([])
    app.setStyle("Fusion")

    # 定位最佳模型
    cands = sorted(WEIGHTS_DIR.glob(BEST_MODEL_GLOB))
    if not cands:
        logger.error("未找到最佳模型: %s", WEIGHTS_DIR / BEST_MODEL_GLOB)
        return 1
    best = cands[0]

    detector = YoloDetector()
    if not detector.load_model(str(best)):
        logger.error("最佳模型加载失败: %s", best)
        return 1

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()

    # 将最佳模型注入主窗口检测器（GUI 默认模型非最佳）
    win.detector = detector
    win.canvas.set_ships([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 逐帧检测 + 截图
    for name in VAL_FRAMES:
        fpath = FRAMES_DIR / name
        if not fpath.exists():
            logger.warning("缺少验证帧: %s", fpath)
            continue
        frame = _load_bgr(fpath)
        ships = detector.detect_frame(frame, conf=0.25)
        win.canvas.load_image(str(fpath))
        win.canvas.set_ships(ships)
        app.processEvents()
        png = OUT_DIR / f"gui_{Path(name).stem}_detected.png"
        win.grab().save(str(png))
        logger.info("已保存: %s (%d 目标)", png.name, len(ships))

    # 空态主界面（未检测）对照
    win.canvas.set_ships([])
    win.canvas.load_image(str(FRAMES_DIR / VAL_FRAMES[0]))
    app.processEvents()
    win.grab().save(str(OUT_DIR / "gui_main_empty.png"))

    logger.info("截图完成 -> %s", OUT_DIR)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
