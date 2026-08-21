from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from .data_models import TargetObject


class DetectorBase(ABC):
    """检测器抽象基类"""

    def __init__(self):
        self._is_loaded = False
        self._model_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型，返回是否成功"""
        pass

    @abstractmethod
    def detect(self, image_path: str, conf: float = 0.25) -> List[TargetObject]:
        """对单张图片执行检测，返回目标列表"""
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """返回类别名称列表"""
        pass
