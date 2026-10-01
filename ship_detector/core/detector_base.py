from abc import ABC, abstractmethod
from typing import List, Optional
from .data_models import DetectedShip


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
        pass

    @abstractmethod
    def detect(self, image_path: str, conf: float = 0.25) -> List[DetectedShip]:
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        pass
