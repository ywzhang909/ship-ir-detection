"""离屏渲染 ship_detector GUI 截图（论文 §7 软件设计与使用 配图）。

与 main.py 完全相同的运行上下文：cwd = ship_detector/、顶层 `core`/`ui` 包导入。
把最佳 large 模型（T5_yolo11l_fusion = YOLO11l + 空频融合，§5A 表 10/图 15 的最佳 large）
加载进 GUI 的 YoloDetector，对 3 张 NSLSR 验证预测帧（val_batch{0,1,2}_pred_101_clean.jpg）
执行 GUI 检测并把 DetectedShip 注入中央画布，然后 win.grab() 离屏截图。

用法（仓库根）:
    QT_QPA_PLATFORM=offscreen uv run python yolo-training/scripts/render_gui_screenshots.py

输出 -> docs/论文/figures/software/gui_detect{N}.png
"""
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent  # ship/
_SHIP_DET = _REPO / "ship_detector"
_MODELS_W = _REPO / "data" / "weights_staging" / "weights"
_FRAMES = [
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch0_pred_101_clean.jpg",
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch1_pred_101_clean.jpg",
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch2_pred_101_clean.jpg",
]
OUT_DIR = _REPO / "docs" / "论文" / "figures" / "software"


def _pick_best_pt() -> Path:
    """T5_yolo11l_fusion_best.pt = YOLO11l + 空频融合（最佳 large）。优先精确命中，
    否则回退到任一 *fusion_best.pt（large 表示在论文 §5A 表 10 中 rank#1）。"""
    exact = _MODELS_W / "T5_yolo11l_fusion_best.pt"
    if exact.exists():
        return exact
    cands = sorted(_MODELS_W.glob("T5_*fusion_best.pt"))
    if cands:
        return cands[0]
    cands = sorted(_MODELS_W.glob("*fusion_best.pt"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"未找到 fusion best-large 权重: {_MODELS_W}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("render")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 必须在 QApplication 前
    sys.path.insert(0, str(_SHIP_DET))
    os.chdir(str(_SHIP_DET))  # 与 main.py 一致：core/ui/config 作为顶层包解析

    import numpy as np
    import cv2
    from PySide6.QtWidgets import QApplication

    from core.yolo_detector import YoloDetector
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    best = _pick_best_pt()
    log.info("最佳 large 模型: %s", best.name)

    detector = YoloDetector()
    if not detector.load_model(str(best)):
        log.error("模型加载失败: %s", best)
        return 1

    win = MainWindow()
    win.resize(1400, 900)
    # 用 GUI 的 YoloDetector 统一驱动（与界面「开始检测」同一路径）
    win.detector = detector
    win.show()
    app.processEvents()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, fpath in enumerate(_FRAMES):
        if not fpath.exists():
            log.warning("跳过缺失帧: %s", fpath)
            continue
        frame = cv2.imread(str(fpath))
        ships = detector.detect_frame(frame, conf=0.25)
        log.info("帧 %d (%s): 检出 %d 目标", i, fpath.name, len(ships))

        win.canvas.load_image(str(fpath))
        win.canvas.set_ships(ships)
        app.processEvents()

        png = OUT_DIR / f"gui_detect{i}.png"
        win.grab().save(str(png))
        log.info("已保存 %s", png.name)

    # 主界面空窗口（软件整体布局示意）
    win.canvas.set_ships([])
    app.processEvents()
    png = OUT_DIR / "gui_main_window.png"
    win.grab().save(str(png))
    log.info("已保存 %s", png.name)

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
