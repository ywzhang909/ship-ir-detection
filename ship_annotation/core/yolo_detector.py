import sys
import logging
from pathlib import Path
from typing import List, Optional

from .detector_base import DetectorBase
from .data_models import TargetObject, BoundingBox

logger = logging.getLogger(__name__)

# 仓库根目录（ship_annotation/core/ 向上三级）与 yolo-training 目录
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_YOLO_TRAINING_DIR = _REPO_ROOT / "yolo-training"
for _p in (str(_REPO_ROOT), str(_YOLO_TRAINING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用仓库根目录 detector.py 的无框架推理核心：
# discover_models / load_model / read_image_bgr / preprocess_for / predict_image
import detector as _core  # noqa: E402

# YOLO 船舶类别名称
SHIP_CLASS_NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
    "Independence", "Jiangkai II", "Oliver Hazard Perry",
    "Sejong Daewang", "Zumwalt",
]

# 类别颜色映射 (R, G, B) for Qt display
CLASS_COLORS = {
    0: (255, 0, 0),      # 红
    1: (0, 200, 0),      # 绿
    2: (255, 255, 0),    # 黄
    3: (0, 150, 255),    # 橙
    4: (255, 0, 255),    # 紫
    5: (0, 255, 255),    # 青
    6: (128, 128, 255),  # 粉
    7: (255, 128, 0),    # 深橙
    8: (128, 255, 0),    # 柠檬绿
}


class YoloDetector(DetectorBase):
    """YOLO 检测器实现。

    真实推理链路复用根目录 ``detector.py``：
    - ``available_models()`` 扫描 runs/detect/ship-detection/*/weights/best.pt，
      按 mAP50-95 降序排列，索引 0 即测试集最优模型；
    - ``load_best_model()`` 自动加载该最优模型；
    - ``detect()`` 推理前按模型训练时的输入分布做预处理
      （dual/wavelet/rpca/tophat/butterworth/raw），保证输入域一致。
    """

    def __init__(self):
        super().__init__()
        self._class_names = SHIP_CLASS_NAMES
        self._loaded = None          # detector.LoadedModel（含 prep 与 device）
        self._model_info = None      # 加载模型对应的 detector.ModelInfo
        self.last_latency_ms: float = 0.0

    @staticmethod
    def available_models() -> list:
        """发现全部可用模型，按 mAP50-95 降序（无指标者排最后）。"""
        return _core.discover_models()

    def load_model(self, model_path: str) -> bool:
        """加载 YOLO 模型并解析其训练输入分布"""
        try:
            self._loaded = _core.load_model(model_path)
            self._model_path = str(model_path)
            self._is_loaded = True
            # 以 checkpoint 内置的 names 为权威类别名（不同模型类别数不同，
            # 如 T5_yolo11l_fusion 为单类 "ship"，9 类舰名模型为另一数据集）
            names = getattr(self._loaded.model, "names", None)
            if isinstance(names, dict) and names:
                self._class_names = [
                    str(names[i]) for i in sorted(names.keys())
                ]
            logger.info(
                "模型加载成功: %s [prep=%s, device=%s]",
                model_path, self._loaded.prep, self._loaded.device,
            )
            return True
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            self._loaded = None
            self._is_loaded = False
            return False

    def load_best_model(self) -> bool:
        """自动加载测试集 mAP50-95 最高的模型"""
        models = self.available_models()
        if not models:
            logger.warning(
                "未找到任何模型权重: %s/*/weights/best.pt", _core.RUNS_DIR
            )
            return False
        best = models[0]
        if not self.load_model(str(best.path)):
            return False
        self._model_info = best
        logger.info("已自动选择最优模型: %s", best.label)
        return True

    @property
    def model_label(self) -> str:
        """当前加载模型的描述标签（名称 + 预处理类型 + 指标）"""
        if self._model_info is not None:
            return self._model_info.label
        if self._is_loaded and self._loaded is not None:
            name = Path(self._model_path or "").parent.parent.name or self._model_path
            return f"{name} [{self._loaded.prep}]"
        return "未加载"

    @property
    def prep(self) -> Optional[str]:
        """当前模型的训练输入分布类型"""
        return self._loaded.prep if self._loaded is not None else None

    def detect(self, image_path: str, conf: float = 0.25) -> List[TargetObject]:
        """对单张图片执行检测：灰度归一化 → 训练分布预处理 → YOLO 推理。

        返回的目标坐标为归一化 xywh（与画布约定一致）。
        """
        if not self._is_loaded or self._loaded is None:
            logger.warning("模型未加载，无法执行检测")
            return []

        try:
            bgr = _core.read_image_bgr(Path(image_path).read_bytes())
            detections, _annotated, latency_ms = _core.predict_image(
                self._loaded, bgr, conf=conf
            )
            self.last_latency_ms = latency_ms

            img_h, img_w = bgr.shape[:2]
            targets: List[TargetObject] = []
            for i, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det.xyxy
                targets.append(
                    TargetObject(
                        id=i,
                        class_id=det.cls_id,
                        class_name=det.cls_name,
                        bbox=BoundingBox(
                            x1 / img_w,
                            y1 / img_h,
                            (x2 - x1) / img_w,
                            (y2 - y1) / img_h,
                        ),
                        confidence=det.conf,
                        color=CLASS_COLORS.get(
                            det.cls_id % len(CLASS_COLORS), (128, 128, 128)
                        ),
                        is_manual=False,
                    )
                )

            logger.info(
                "检测完成: %s → %d 个目标 (%.0f ms)",
                image_path, len(targets), latency_ms,
            )
            return targets

        except Exception as e:
            logger.error("检测失败: %s", e)
            return []

    def get_class_names(self) -> List[str]:
        return self._class_names
