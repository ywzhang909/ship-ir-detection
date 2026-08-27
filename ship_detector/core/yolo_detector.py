import sys
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .detector_base import DetectorBase
from .data_models import DetectedShip, BoundingBox

logger = logging.getLogger(__name__)

# 使用 config.py 中的统一映射
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _get_class_info():
    """延迟导入 config，避免循环依赖"""
    from config import SHIP_CLASS_NAMES, SHIP_CLASS_COLORS
    return SHIP_CLASS_NAMES, SHIP_CLASS_COLORS


class YoloDetector(DetectorBase):

    def __init__(self):
        super().__init__()
        self._model = None
        self._sahi_model = None
        self._use_sahi = False
        self._imgsz = 1280

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

    def set_sahi(self, enabled: bool):
        """启用/禁用 SAHI 切片推理"""
        self._use_sahi = enabled
        if enabled:
            self._init_sahi_model()

    def _init_sahi_model(self):
        """延迟初始化 SAHI 模型"""
        if self._sahi_model is not None:
            return
        try:
            import torch
            from sahi import AutoDetectionModel
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self._model_path,
                confidence_threshold=0.1,
                device=device,
            )
            logger.info("SAHI 模型初始化完成 (device=%s)", device)
        except ImportError:
            logger.warning("SAHI 未安装，回退到标准推理")
            self._use_sahi = False
        except Exception as e:
            logger.error("SAHI 初始化失败: %s", e)
            self._use_sahi = False

    # ========== 接口方法 ==========

    def detect(self, image_path: str, conf: float = 0.25) -> List[DetectedShip]:
        if not self._is_loaded or self._model is None:
            logger.warning("模型未加载")
            return []

        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("无法读取图片: %s", image_path)
                return []
            return self.detect_frame(img, conf)
        except Exception as e:
            logger.error("检测失败: %s", e)
            return []

    def detect_frame(self, frame: np.ndarray, conf: float = 0.25) -> List[DetectedShip]:
        """从 BGR ndarray 检测 — 视频帧/相机帧的入口"""
        if not self._is_loaded or self._model is None:
            return []

        try:
            class_names, class_colors = _get_class_info()

            if self._use_sahi and self._sahi_model is not None:
                return self._detect_sahi(frame, conf, class_names, class_colors)
            else:
                return self._detect_standard(frame, conf, class_names, class_colors)
        except Exception as e:
            logger.error("帧检测失败: %s", e)
            return []

    def _detect_standard(self, frame: np.ndarray, conf: float,
                         class_names: list, class_colors: dict) -> List[DetectedShip]:
        """标准 YOLO 推理"""
        results = self._model(frame, conf=conf, imgsz=self._imgsz, verbose=False)
        ships = []
        track_id = 1
        img_h, img_w = frame.shape[:2]

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                norm_x = x1 / img_w
                norm_y = y1 / img_h
                norm_w = (x2 - x1) / img_w
                norm_h = (y2 - y1) / img_h

                cls_name = (class_names[cls_id]
                            if cls_id < len(class_names)
                            else f"class_{cls_id}")
                color = class_colors.get(cls_id % len(class_colors), (128, 128, 128))

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

        logger.info("标准推理完成: %d 个目标", len(ships))
        return ships

    def _detect_sahi(self, frame: np.ndarray, conf: float,
                     class_names: list, class_colors: dict) -> List[DetectedShip]:
        """SAHI 切片推理 — 小目标检测"""
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            frame,
            self._sahi_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
        )

        ships = []
        track_id = 1
        img_h, img_w = frame.shape[:2]

        for pred in result.object_prediction_list:
            cls_id = int(pred.category.id)
            score = float(pred.score.value)
            if score < conf:
                continue

            x1, y1, x2, y2 = pred.bbox.to_voc_bbox()
            norm_x = x1 / img_w
            norm_y = y1 / img_h
            norm_w = (x2 - x1) / img_w
            norm_h = (y2 - y1) / img_h

            cls_name = (class_names[cls_id]
                        if cls_id < len(class_names)
                        else f"class_{cls_id}")
            color = class_colors.get(cls_id % len(class_colors), (128, 128, 128))

            ship = DetectedShip(
                track_id=track_id,
                class_id=cls_id,
                class_name=cls_name,
                bbox=BoundingBox(norm_x, norm_y, norm_w, norm_h),
                confidence=score,
                color=color,
            )
            ships.append(ship)
            track_id += 1

        logger.info("SAHI 推理完成: %d 个目标", len(ships))
        return ships

    def get_class_names(self) -> List[str]:
        from config import SHIP_CLASS_NAMES
        return SHIP_CLASS_NAMES
