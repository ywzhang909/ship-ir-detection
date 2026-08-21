from typing import List, Optional, Tuple
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QPoint, QRect
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QPaintEvent,
)
from core.data_models import AnnotationTool, BoundingBox, TargetObject


class CentralCanvas(QLabel):
    """中央画布：图片渲染、标注绘制、鼠标交互（选择/框选）"""

    target_selected = Signal(object)
    targets_updated = Signal(list)
    status_message = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")

        self.current_tool = AnnotationTool.SELECT
        self.display_pixmap: QPixmap = QPixmap(640, 480)
        self.display_pixmap.fill(QColor("#1a1a1a"))
        self.targets: List[TargetObject] = []
        self.selected_target: Optional[TargetObject] = None
        self.default_class_id = 0

        self._mouse_pressed = False
        self._drag_start = QPoint()
        self._drag_current = QPoint()
        self._is_dragging_target = False

        self.setMouseTracking(True)

    # ========== 公共接口 ==========

    def set_current_tool(self, tool: AnnotationTool):
        self.current_tool = tool
        if tool != AnnotationTool.SELECT:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_default_class(self, class_id: int):
        self.default_class_id = class_id

    def load_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.status_message.emit(f"无法加载图片: {path}")
            return
        self.display_pixmap = pixmap
        self.targets = []
        self.selected_target = None
        self.update()
        self.status_message.emit(f"已加载: {path}")
        self.targets_updated.emit(self.targets)

    def connect_camera(self, url: str):
        self.status_message.emit(f"正在连接: {url}...")

    def update_frame(self, pixmap: QPixmap, detections: List[TargetObject]):
        self.display_pixmap = pixmap
        self.targets = detections
        self.update()
        self.targets_updated.emit(self.targets)

    def run_detection(self):
        self.status_message.emit("正在检测...")

    def update_target(self, target: TargetObject):
        for i, t in enumerate(self.targets):
            if t.id == target.id:
                self.targets[i] = target
                break
        self.update()

    def delete_target(self, target_id: int):
        self.targets = [t for t in self.targets if t.id != target_id]
        self.selected_target = None
        self.update()
        self.targets_updated.emit(self.targets)

    # ========== 绘制 ==========

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        img_rect = self._calc_fit_rect()
        painter.drawPixmap(img_rect, self.display_pixmap)

        painter.setRenderHint(QPainter.Antialiasing)
        for target in self.targets:
            self._draw_target(painter, target, img_rect)

        if self._mouse_pressed and self.current_tool == AnnotationTool.RECTANGLE:
            self._draw_draft_rect(painter, img_rect)

        painter.end()

    def _calc_fit_rect(self) -> QRect:
        """letterbox 缩放：保持宽高比居中"""
        if self.display_pixmap.isNull():
            return self.rect()
        pw, ph = self.display_pixmap.width(), self.display_pixmap.height()
        cw, ch = self.width(), self.height()
        if pw > 0 and ph > 0:
            scale = min(cw / pw, ch / ph)
        else:
            scale = 1
        w, h = int(pw * scale), int(ph * scale)
        x, y = (cw - w) // 2, (ch - h) // 2
        return QRect(x, y, w, h)

    def _to_norm(self, pos: QPoint, img_rect: QRect) -> Tuple[float, float]:
        """屏幕坐标 → 归一化坐标"""
        x = (pos.x() - img_rect.x()) / img_rect.width()
        y = (pos.y() - img_rect.y()) / img_rect.height()
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _to_pixel(self, bbox: BoundingBox, img_rect: QRect) -> QRect:
        """归一化 BoundingBox → 屏幕像素 QRect"""
        x1, y1, x2, y2 = bbox.to_pixels(img_rect.width(), img_rect.height())
        return QRect(img_rect.x() + x1, img_rect.y() + y1, x2 - x1, y2 - y1)

    def _draw_target(self, painter: QPainter, target: TargetObject, img_rect: QRect):
        rect = self._to_pixel(target.bbox, img_rect)
        color = QColor(*target.color)

        pen = QPen(color, 2)
        if target == self.selected_target:
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

        # 标签背景
        label = f"{target.class_name} {target.confidence:.2f}"
        font = QFont("Sans Serif", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw, th = fm.horizontalAdvance(label) + 8, fm.height() + 4
        painter.fillRect(rect.x(), rect.y() - th, tw, th, color)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.x() + 4, rect.y() - th + fm.ascent() + 2, label)

    def _draw_draft_rect(self, painter: QPainter, img_rect: QRect):
        """绘制拖拽中的临时矩形"""
        pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
        painter.setPen(pen)
        x = min(self._drag_start.x(), self._drag_current.x())
        y = min(self._drag_start.y(), self._drag_current.y())
        w = abs(self._drag_current.x() - self._drag_start.x())
        h = abs(self._drag_current.y() - self._drag_start.y())
        painter.drawRect(x, y, w, h)

    # ========== 鼠标交互 ==========

    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self._calc_fit_rect()
        if not img_rect.contains(event.pos()):
            return

        self._mouse_pressed = True
        self._drag_start = event.pos()
        self._drag_current = event.pos()

        if self.current_tool == AnnotationTool.SELECT:
            nx, ny = self._to_norm(event.pos(), img_rect)
            clicked = None
            for t in reversed(self.targets):
                x1, y1, x2, y2 = t.bbox.to_pixels(1, 1)
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    clicked = t
                    break
            self.selected_target = clicked
            self.target_selected.emit(clicked)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._mouse_pressed:
            return
        self._drag_current = event.pos()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._mouse_pressed:
            return
        self._mouse_pressed = False
        img_rect = self._calc_fit_rect()

        if self.current_tool == AnnotationTool.RECTANGLE:
            n1 = self._to_norm(self._drag_start, img_rect)
            n2 = self._to_norm(event.pos(), img_rect)
            x, y = min(n1[0], n2[0]), min(n1[1], n2[1])
            w, h = abs(n2[0] - n1[0]), abs(n2[1] - n1[1])
            if w < 0.01 or h < 0.01:
                self.update()
                return

            from config import PROJECT_CONFIG

            new_id = max([t.id for t in self.targets], default=0) + 1
            class_info = PROJECT_CONFIG.class_map.get(
                self.default_class_id, ("未知", (128, 128, 128))
            )

            new_target = TargetObject(
                id=new_id,
                class_id=self.default_class_id,
                class_name=class_info[0],
                bbox=BoundingBox(x, y, w, h),
                confidence=1.0,
                color=class_info[1],
                is_manual=True,
            )
            self.targets.append(new_target)
            self.targets_updated.emit(self.targets)
            self.status_message.emit(f"新建标注: {class_info[0]} #{new_id}")

        self.update()
