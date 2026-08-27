from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 模型权重路径 — 指向项目根目录下的预训练权重
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL_PATH = str(REPO_ROOT / "yolo11m.pt")

DEFAULT_CONFIDENCE = 0.45
DEFAULT_IMGSZ = 1280

# 9类船舶映射 — 单一数据源 (消除 yolo_detector.py 中的重复)
SHIP_CLASSES = {
    0: ("Ada", (0, 0, 255)),
    1: ("Akizuki", (0, 200, 0)),
    2: ("Alvaro De Bazan", (0, 255, 255)),
    3: ("Armourique", (255, 150, 0)),
    4: ("Independence", (255, 0, 255)),
    5: ("Jiangkai II", (255, 255, 0)),
    6: ("Oliver Hazard Perry", (128, 128, 255)),
    7: ("Sejong Daewang", (0, 128, 255)),
    8: ("Zumwalt", (128, 255, 0)),
}

SHIP_CLASS_NAMES = [v[0] for v in SHIP_CLASSES.values()]
SHIP_CLASS_COLORS = {k: v[1] for k, v in SHIP_CLASSES.items()}
