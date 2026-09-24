"""全局常量、路径配置"""
from core.data_models import ProjectConfig
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 仓库根目录（ship_annotation/ 的上一级）
REPO_ROOT = PROJECT_ROOT.parent

# 训练运行目录：runs/detect/ship-detection/<run>/weights/best.pt
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "detect" / "ship-detection"

# 导出的配置实例
PROJECT_CONFIG = ProjectConfig(
    name="舰船识别项目",
    class_map={
        0: ("Ada", (255, 0, 0)),
        1: ("Akizuki", (0, 200, 0)),
        2: ("Alvaro De Bazan", (255, 255, 0)),
        3: ("Armourique", (0, 150, 255)),
        4: ("Independence", (255, 0, 255)),
        5: ("Jiangkai II", (0, 255, 255)),
        6: ("Oliver Hazard Perry", (128, 128, 255)),
        7: ("Sejong Daewang", (255, 128, 0)),
        8: ("Zumwalt", (128, 255, 0)),
    },
    confidence_threshold=0.45,
    auto_save=True,
)
