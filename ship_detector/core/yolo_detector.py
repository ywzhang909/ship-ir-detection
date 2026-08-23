import sys
import logging
from pathlib import Path
from typing import List, Optional

from .detector_base import DetectorBase
from .data_models import DetectedShip, BoundingBox

logger = logging.getLogger(__name__)

_YOLO_DIR = str(Path(__file__).resolve().parent.parent.parent / "yolo-training")
if _YOLO_DIR not in sys.path:
    sys.path.insert(0, _YOLO_DIR)

SHIP_CLASS_NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
    "Independence", "Jiangkai II", "Oliver Hazard Perry",
    "Sejong Daewang", "Zumwalt",
]

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 200, 0),
    2: (0, 255, 255),
    3: (255, 150, 0),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (128, 128, 255),
    7: (0, 128, 255),
    8: (128, 255, 0),
}


class YoloDetector(DetectorBase):

    def __init__(self):
        super().__init__()
        self._class_names = SHIP_CLASS_NAMES
        self._model = None

    def load_model(self, model_path: str) -> bool:
        try:
            from ultralytics import YOLO
            if not Path(model_path).exists():
                logger.warning("模型文件不存在: %s", model_path)
                return False
            self._model = YOLO(model_path)
            self._model_path = model_path
            self._is_loaded = True
            logger.info("模型加载成功: %s", model_path)
            return True
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            self._is_loaded = False
            return False

    def detect(self, image_path: str, conf: float = 0.25) -> List[DetectedShip]:
        if not self._is_loaded or self._model is None:
            logger.warning("模型未加载")
            return []

        try:
            import cv2
            results = self._model(image_path, conf=conf, verbose=False)
            ships = []
            track_id = 1

            img = cv2.imread(image_path)
            if img is None:
                return []
            img_h, img_w = img.shape[:2]

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        norm_x = x1 / img_w
                        norm_y = y1 / img_h
                        norm_w = (x2 - x1) / img_w
                        norm_h = (y2 - y1) / img_h

                        cls_name = (self._class_names[cls_id]
                                    if cls_id < len(self._class_names)
                                    else f"class_{cls_id}")
                        color = CLASS_COLORS.get(cls_id % len(CLASS_COLORS), (128, 128, 128))

                        ship = DetectedShip(
                            track_id=track_id,
                            class_id=cls_id,
                            class_name=cls_name,
                            bbox=BoundingBox(norm_x, norm_y, norm_w, norm_h),
                            confidence=conf_val,
                            color=color,
                        )
                        ships.append(ship)
                        track_id += 1

            logger.info("检测完成: %s → %d 艘舰船", image_path, len(ships))
            return ships

        except Exception as e:
            logger.error("检测失败: %s", e)
            return []

    def get_class_names(self) -> List[str]:
        return self._class_names
