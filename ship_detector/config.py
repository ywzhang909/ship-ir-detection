from pathlib import Path
from core.data_models import SourceType

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT.parent / "yolo-training" / "runs" / "detect" / "train" / "weights"

CLASS_MAP = {
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

DEFAULT_CONFIDENCE = 0.45
DEFAULT_MODEL_PATH = str(DEFAULT_MODEL_DIR / "best.pt")
