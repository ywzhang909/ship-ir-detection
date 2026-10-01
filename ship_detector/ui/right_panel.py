from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel,
    QProgressBar, QFormLayout, QLineEdit,
    QListWidget, QListWidgetItem, QHBoxLayout, QFrame, QCheckBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from core.data_models import DetectedShip


class RightPanel(QWidget):
    highlight_ship = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        self._current_ship: Optional[DetectedShip] = None
        self._img_w = 1024
        self._img_h = 512
        self._ships: List[DetectedShip] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === 概览 ===
        self.summary = QLabel("共识别 0 艘舰船")
        self.summary.setStyleSheet("font-size: 13px; color: #666;")
        layout.addWidget(self.summary)

        # === 目标列表 ===
        list_group = QGroupBox("目标列表")
        lg = QVBoxLayout()
        self.ship_list = QListWidget()
        self.ship_list.itemClicked.connect(self._on_list_click)
        lg.addWidget(self.ship_list)
        list_group.setLayout(lg)
        layout.addWidget(list_group, stretch=1)

        # === 详情区 ===
        self.detail_group = QGroupBox("目标详情")
        dg = QFormLayout()
        dg.setSpacing(8)

        self.row_class = QHBoxLayout()
        self.color_block = QFrame()
        self.color_block.setFixedSize(16, 16)
        self.lbl_class = QLabel("-")
        self.row_class.addWidget(self.color_block)
        self.row_class.addWidget(self.lbl_class)
        self.row_class.addStretch()
        dg.addRow("类别:", self.row_class)

        self.lbl_id = QLabel("-")
        dg.addRow("跟踪 ID:", self.lbl_id)

        self.bar_conf = QProgressBar()
        self.bar_conf.setRange(0, 100)
        self.bar_conf.setTextVisible(True)
        dg.addRow("置信度:", self.bar_conf)

        self.lbl_norm = QLabel("-")
        dg.addRow("归一化坐标:", self.lbl_norm)

        self.lbl_pixel = QLabel("-")
        dg.addRow("像素坐标:", self.lbl_pixel)

        self.lbl_center = QLabel("-")
        dg.addRow("中心点:", self.lbl_center)

        self.lbl_area = QLabel("-")
        dg.addRow("面积占比:", self.lbl_area)

        # 备注 (持久化到 DetectedShip.note)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("添加备注...")
        self.edit_note.returnPressed.connect(self._save_note)
        dg.addRow("备注:", self.edit_note)

        # 已审核复选框
        self.chk_reviewed = QCheckBox("已审核")
        self.chk_reviewed.toggled.connect(self._toggle_reviewed)
        dg.addRow("", self.chk_reviewed)

        self.detail_group.setLayout(dg)
        layout.addWidget(self.detail_group, stretch=2)

        self._clear_details()
        layout.addStretch()

    def set_image_size(self, w: int, h: int):
        """设置当前图片尺寸，用于像素坐标计算"""
        self._img_w = w
        self._img_h = h

    def update_summary(self, count: int):
        self.summary.setText(f"共识别 {count} 艘舰船")

    def display_ship(self, ship: Optional[DetectedShip]):
        self._current_ship = ship
        if ship is None:
            self._clear_details()
            return

        color = QColor(*ship.color)
        self.color_block.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #333; border-radius: 3px;"
        )
        self.lbl_class.setText(ship.class_name)
        self.lbl_id.setText(str(ship.track_id))
        self.bar_conf.setValue(int(ship.confidence * 100))
        self.bar_conf.setFormat(f"{ship.confidence:.1%}")

        b = ship.bbox
        self.lbl_norm.setText(f"x:{b.x:.3f}  y:{b.y:.3f}  w:{b.w:.3f}  h:{b.h:.3f}")

        # 使用实际图片尺寸计算像素坐标
        px = b.to_pixels(self._img_w, self._img_h)
        self.lbl_pixel.setText(f"x:{px[0]}  y:{px[1]}  w:{px[2]-px[0]}  h:{px[3]-px[1]}")

        self.lbl_center.setText(f"({b.center[0]:.3f}, {b.center[1]:.3f})")
        self.lbl_area.setText(f"{b.area_norm:.2%}")

        # 备注和审核状态
        self.edit_note.blockSignals(True)
        self.edit_note.setText(ship.note)
        self.edit_note.blockSignals(False)
        self.chk_reviewed.blockSignals(True)
        self.chk_reviewed.setChecked(ship.reviewed)
        self.chk_reviewed.blockSignals(False)

    def _clear_details(self):
        self.color_block.setStyleSheet(
            "background-color: #ccc; border: 1px solid #333;"
        )
        self.lbl_class.setText("请点击识别框")
        self.lbl_id.setText("-")
        self.bar_conf.setValue(0)
        self.bar_conf.setFormat("")
        self.lbl_norm.setText("-")
        self.lbl_pixel.setText("-")
        self.lbl_center.setText("-")
        self.lbl_area.setText("-")
        self.edit_note.blockSignals(True)
        self.edit_note.clear()
        self.edit_note.blockSignals(False)
        self.chk_reviewed.blockSignals(True)
        self.chk_reviewed.setChecked(False)
        self.chk_reviewed.blockSignals(False)

    def _save_note(self):
        """保存备注到 DetectedShip"""
        if self._current_ship is not None:
            self._current_ship.note = self.edit_note.text()

    def _toggle_reviewed(self, checked: bool):
        """标记审核状态"""
        if self._current_ship is not None:
            self._current_ship.reviewed = checked

    def update_ship_list(self, ships: List[DetectedShip]):
        self._ships = ships
        self.ship_list.clear()
        for ship in ships:
            prefix = "V " if ship.reviewed else ""
            item = QListWidgetItem(f"{prefix}#{ship.track_id} {ship.class_name}")
            item.setData(Qt.UserRole, ship.track_id)
            color = QColor(*ship.color)
            item.setForeground(color)
            self.ship_list.addItem(item)

    def _on_list_click(self, item: QListWidgetItem):
        track_id = item.data(Qt.UserRole)
        self.highlight_ship.emit(track_id)
