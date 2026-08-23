from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from datetime import datetime


class SourceType(Enum):
    IMAGE = auto()
    VIDEO = auto()
    CAMERA = auto()


@dataclass
class BoundingBox:
    """归一化坐标 0.0~1.0"""
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

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def area_norm(self) -> float:
        return self.w * self.h


@dataclass
class DetectedShip:
    """识别出的单个舰船目标"""
    track_id: int
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    color: Tuple[int, int, int] = (0, 0, 255)

    # 扩展信息（可选，由检测器填充）
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None
    length_m: Optional[float] = None
    ship_type: Optional[str] = None


@dataclass
class FrameResult:
    """单帧识别结果"""
    frame_id: int
    timestamp: datetime
    source_type: SourceType
    image_path: Optional[str] = None
    pixmap: Optional[object] = None
    ships: List[DetectedShip] = field(default_factory=list)
    total_ships: int = 0
