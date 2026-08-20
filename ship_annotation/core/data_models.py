from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from datetime import datetime


class SourceType(Enum):
    IMAGE = auto()
    CAMERA = auto()


class AnnotationTool(Enum):
    SELECT = auto()
    RECTANGLE = auto()
    ARROW = auto()
    DELETE = auto()


@dataclass
class BoundingBox:
    """归一化坐标 0.0~1.0，避免分辨率耦合"""
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = int(self.x * img_w)
        y1 = int(self.y * img_h)
        x2 = int((self.x + self.w) * img_w)
        y2 = int((self.y + self.h) * img_h)
        return x1, y1, x2, y2


@dataclass
class TargetObject:
    id: int
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = 0.0
    color: Tuple[int, int, int] = (255, 0, 0)
    is_manual: bool = False
    note: str = ""
    arrow_end: Optional[Tuple[float, float]] = None


@dataclass
class FrameData:
    frame_id: int
    timestamp: datetime
    source_type: SourceType
    image_path: Optional[str] = None
    pixmap: Optional[object] = None
    targets: List[TargetObject] = field(default_factory=list)


@dataclass
class ProjectConfig:
    name: str = "未命名项目"
    class_map: dict = field(default_factory=lambda: {
        0: ("船只", (0, 0, 255)),
        1: ("飞机", (0, 255, 0)),
        2: ("浮标", (255, 255, 0)),
    })
    confidence_threshold: float = 0.5
    auto_save: bool = True
