from typing import List, Optional, Tuple
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QPoint, QRect, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QPaintEvent, QImage
import numpy as np
import cv2

from core.data_models import DetectedShip, BoundingBox


class CentralCanvas(QLabel):
    ship_selected = Signal(object)
    ships_updated = Signal(int)
    status_message = Signal(str)
    fps_updated = Signal(float)
    resolution_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0d1117;")

        self.display_pixmap: QPixmap = QPixmap(640, 480)
        self.display_pixmap.fill(QColor("#0d1117"))
        self.ships: List[DetectedShip] = []
        self.selected_ship: Optional[DetectedShip] = None
        self.hovered_ship: Optional[DetectedShip] = None

        # 当前帧的 ndarray (BGR) — 用于检测
        self._current_frame: Optional[np.ndarray] = None
        self._current_image_path: Optional[str] = None

        # 视频/相机
        self._cap: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._is_playing = False
        self._frame_id = 0
        self._fps_counter = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._calc_fps)
        self._fps_timer.start(1000)

        self.setMouseTracking(True)

    @property
    def has_content(self) -> bool:
        return not self.display_pixmap.isNull()

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        return self._current_frame

    @property
    def current_image_path(self) -> Optional[str]:
        return self._current_image_path

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def image_width(self) -> int:
        return self.display_pixmap.width()

    @property
    def image_height(self) -> int:
        return self.display_pixmap.height()

    # ========== 图片加载 ==========

    def load_image(self, path: str):
        self._stop_capture()
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.status_message.emit(f"无法加载: {path}")
            return
        self.display_pixmap = pixmap
        self._current_image_path = path
        self._current_frame = cv2.imread(path)
        self.ships = []
        self.selected_ship = None
        self.hovered_ship = None
        self.ship_selected.emit(None)
        self.resolution_changed.emit(pixmap.width(), pixmap.height())
        self.update()
        self.status_message.emit(f"已加载图片: {path}")

    # ========== 视频/相机 ==========

    def load_video(self, path: str):
        self._stop_capture()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status_message.emit(f"无法打开视频: {path}")
            return
        self._cap = cap
        self._frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._timer.start(int(1000 / fps))
        self._is_playing = True
        self._read_next_frame()
        self.status_message.emit(f"播放视频: {path}")

    def connect_camera(self, url: str):
        self._stop_capture()
        # 尝试解析为整数 (本地摄像头)
        try:
            source = int(url)
        except ValueError:
            source = url  # RTSP URL

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status_message.emit(f"无法连接相机: {url}")
            return
        self._cap = cap
        self._frame_id = 0
        self._timer.start(33)  # ~30fps
        self._is_playing = True
        self._read_next_frame()
        self.status_message.emit(f"已连接相机: {url}")

    def pause(self):
        if self._is_playing:
            self._timer.stop()
            self._is_playing = False
            self.status_message.emit("已暂停")
        elif self._cap is not None and self._cap.isOpened():
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._timer.start(int(1000 / fps))
            self._is_playing = True
            self.status_message.emit("继续播放")

    def _stop_capture(self):
        self._timer.stop()
        self._is_playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_next_frame(self):
        if self._cap is None or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret:
            self._timer.stop()
            self._is_playing = False
            self.status_message.emit("视频播放结束")
            return
        self._frame_id += 1
        self._current_frame = frame
        self._set_frame_pixmap(frame)
        self._fps_counter += 1

    def _next_frame(self):
        """QTimer 回调 — 读取下一帧"""
        self._read_next_frame()

    def _calc_fps(self):
        """每秒计算一次 FPS"""
        if self._fps_counter > 0:
            self.fps_updated.emit(float(self._fps_counter))
            self._fps_counter = 0

    def _set_frame_pixmap(self, frame: np.ndarray):
        """将 BGR ndarray 转为 QPixmap 显示"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(qimg)
        self.resolution_changed.emit(w, h)
        self.update()

    def update_frame(self, pixmap: QPixmap, ships: List[DetectedShip] = None):
        """外部更新显示帧"""
        self.display_pixmap = pixmap
        if ships is not None:
            self.ships = ships
            self.ships_updated.emit(len(ships))
        self.update()

    def set_frame(self, frame: np.ndarray):
        """设置当前帧 ndarray (不触发检测)"""
        self._current_frame = frame
        self._set_frame_pixmap(frame)

    # ========== 检测 ==========

    def highlight_ship(self, track_id: int):
        for ship in self.ships:
            if ship.track_id == track_id:
                self.selected_ship = ship
                self.update()
                break

    def set_ships(self, ships: List[DetectedShip]):
        self.ships = ships
        self.selected_ship = None
        self.hovered_ship = None
        self.ships_updated.emit(len(ships))
        self.update()

    # ========== 绘制 ==========
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        img_rect = self._calc_fit_rect()
        painter.drawPixmap(img_rect, self.display_pixmap)

        painter.setRenderHint(QPainter.Antialiasing)
        for ship in self.ships:
            self._draw_ship_box(painter, ship, img_rect)

        # 播放状态指示
        if self._is_playing:
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            painter.setBrush(QColor(0, 200, 0, 100))
            painter.drawEllipse(img_rect.right() - 30, img_rect.y() + 10, 20, 20)

        painter.end()

    def _calc_fit_rect(self) -> QRect:
        if self.display_pixmap.isNull():
            return self.rect()
        pw, ph = self.display_pixmap.width(), self.display_pixmap.height()
        cw, ch = self.width(), self.height()
        scale = min(cw / pw, ch / ph) if pw > 0 and ph > 0 else 1
        w, h = int(pw * scale), int(ph * scale)
        x, y = (cw - w) // 2, (ch - h) // 2
        return QRect(x, y, w, h)

    def _to_norm(self, pos: QPoint, img_rect: QRect) -> Tuple[float, float]:
        x = (pos.x() - img_rect.x()) / img_rect.width()
        y = (pos.y() - img_rect.y()) / img_rect.height()
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _to_pixel_rect(self, bbox: BoundingBox, img_rect: QRect) -> QRect:
        x1, y1, x2, y2 = bbox.to_pixels(img_rect.width(), img_rect.height())
        return QRect(img_rect.x() + x1, img_rect.y() + y1, x2 - x1, y2 - y1)

    def _draw_ship_box(self, painter: QPainter, ship: DetectedShip, img_rect: QRect):
        rect = self._to_pixel_rect(ship.bbox, img_rect)
        color = QColor(*ship.color)

        is_selected = (ship == self.selected_ship)
        is_hovered = (ship == self.hovered_ship)

        pen_width = 3 if is_selected else (2 if is_hovered else 1)
        pen_style = Qt.DashLine if is_selected else Qt.SolidLine

        pen = QPen(color, pen_width, pen_style)
        painter.setPen(pen)

        if is_selected:
            painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 40))

        painter.drawRect(rect)

        label = f"{ship.class_name} #{ship.track_id}  {ship.confidence:.0%}"
        font = QFont("Sans Serif", 9, QFont.Bold if is_selected else QFont.Normal)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 4

        # 标签在框上方 (避免裁剪)
        label_y = max(img_rect.y(), rect.y() - th)
        painter.fillRect(rect.x(), label_y, tw, th, color)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.x() + 5, label_y + fm.ascent() + 2, label)

        if is_selected:
            painter.setBrush(color)
            painter.drawEllipse(rect.right() - 6, rect.bottom() - 6, 6, 6)

    # ========== 鼠标交互 ==========
    def mouseMoveEvent(self, event):
        img_rect = self._calc_fit_rect()
        if not img_rect.contains(event.position().toPoint()):
            if self.hovered_ship:
                self.hovered_ship = None
                self.setCursor(Qt.ArrowCursor)
                self.update()
            return

        nx, ny = self._to_norm(event.position().toPoint(), img_rect)
        found = None
        for ship in reversed(self.ships):
            x1, y1, x2, y2 = ship.bbox.to_pixels(1, 1)
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                found = ship
                break

        if found != self.hovered_ship:
            self.hovered_ship = found
            self.setCursor(Qt.PointingHandCursor if found else Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return

        img_rect = self._calc_fit_rect()
        if not img_rect.contains(event.position().toPoint()):
            self.selected_ship = None
            self.ship_selected.emit(None)
            self.update()
            return

        nx, ny = self._to_norm(event.position().toPoint(), img_rect)
        clicked = None
        for ship in reversed(self.ships):
            x1, y1, x2, y2 = ship.bbox.to_pixels(1, 1)
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                clicked = ship
                break

        self.selected_ship = clicked
        self.ship_selected.emit(clicked)

        if clicked:
            self.status_message.emit(
                f"选中: {clicked.class_name} #{clicked.track_id} "
                f"置信度{clicked.confidence:.1%}"
            )
        else:
            self.status_message.emit("取消选中")

        self.update()

    def closeEvent(self, event):
        """清理资源"""
        self._stop_capture()
        self._fps_timer.stop()
        super().closeEvent(event)
