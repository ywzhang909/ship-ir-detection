"""论文「软件设计与使用」GUI 渲染脚本 — 离屏截图生成。

对 ship_detector 的 PySide6 界面(与 main.py 完全相同的顶层导入方式)做离屏渲染:
加载最优模型 T5_yolo11l_fusion_best.pt (YOLO11l + 空频融合 · 最佳 large · §5.2/表8),
对 3 张三源 NSLSR 验证帧执行目标检测(空频融合已内置), 经 SAHI 检测逻辑(§软件检测流程)
在中央画布上绘制检测框, 整窗 grab() 截图输出到 docs/论文/figures/software/。

用法(在 ship_detector/ 目录下执行, 与 main.py 一致):
    QT_QPA_PLATFORM=offscreen uv run python scripts/render_gui_screenshots.py
"""
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

_REPO = Path(__file__).resolve().parent.parent  # ship_detector/
_BEST = Path("/home/ws/code/ship/data/weights_staging/weights/T5_yolo11l_fusion_best.pt")
_VAL = Path("/home/ws/code/ship/data/weights_staging/frames/val")
_FRAMES = [_VAL / f"val_batch{i}_pred_101_clean.jpg" for i in range(3)]
_OUT = Path("/home/ws/code/ship/docs/论文/figures/software")

log = logging.getLogger("render")
log.setLevel(logging.INFO)


def main() -> int:
    import cv2
    from PySide6.QtWidgets import QApplication

    # 顶层导入: 与 main.py 相同 —— cd ship_detector, core./ui./config 直接可见
    sys.path.insert(0, str(_REPO.parent))
    os.chdir(str(_REPO))
    from core.yolo_detector import YoloDetector
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    detector = YoloDetector()
    if not detector.load_model(str(_BEST)):
        log.error("模型加载失败: %s", _BEST)
        return 1
    log.info("模型加载成功: %s", _BEST.name)

    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()

    _OUT.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(_FRAMES):
        if not f.exists():
            log.warning("跳过缺失帧: %s", f)
            continue
        frame = cv2.imread(str(f))
        ships = detector.detect_frame(frame, conf=0.25)
        win.canvas.load_image(str(f))
        win.canvas.set_ships(ships)
        app.processEvents()
        png = _OUT / f"gui_frame{i}_{Path(f).name.replace('.jpg', '')}.png"
        win.grab().save(str(png))
        log.info("已渲染 %s  (目标数=%d)", png.name, len(ships))

    # 空主窗口截图(软件整体布局)
    win.canvas.ships = []
    win.canvas.set_ships([])
    app.processEvents()
    win.grab().save(str(_OUT / "gui_main_window.png"))
    log.info("已渲染 gui_main_window.png")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
