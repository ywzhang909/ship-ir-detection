"""应用级逻辑（单例）"""
import logging
from pathlib import Path
from core.yolo_detector import YoloDetector
from core.frame_source import FrameSource
from config import PROJECT_CONFIG

logger = logging.getLogger(__name__)


class App:
    """应用单例，管理检测器和数据源"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.detector = YoloDetector()
        self.frame_source = FrameSource()
        self.config = PROJECT_CONFIG

        logger.info("应用初始化完成")

    def load_model(self, model_path: str) -> bool:
        """加载检测模型"""
        success = self.detector.load_model(model_path)
        if success:
            logger.info("模型已加载: %s", model_path)
        else:
            logger.warning("模型加载失败: %s", model_path)
        return success

    def detect_current(self) -> list:
        """对当前图片执行检测"""
        if self.frame_source.current_path is None:
            logger.warning("未选择图片")
            return []
        return self.detector.detect(
            self.frame_source.current_path,
            conf=self.config.confidence_threshold,
        )
