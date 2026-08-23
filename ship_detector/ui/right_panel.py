from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel,
    QProgressBar, QFormLayout, QLineEdit,
    QListWidget, QListWidgetItem, QHBoxLayout, QFrame,
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

        self.lbl_speed = QLabel("-")
        dg.addRow("航速(节):", self.lbl_speed)

        self.lbl_heading = QLabel("-")
        dg.addRow("航向(°):", self.lbl_heading)

        self.lbl_length = QLabel("-")
        dg.addRow("估算船长(m):", self.lbl_length)

        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("添加备注...")
        dg.addRow("备注:", self.edit_note)

        self.detail_group.setLayout(dg)
        layout.addWidget(self.detail_group, stretch=2)

        self._clear_details()
        layout.addStretch()

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
        self.lbl_pixel.setText("见画布")
        self.lbl_center.setText(f"({b.center[0]:.3f}, {b.center[1]:.3f})")
        self.lbl_area.setText(f"{b.area_norm:.2%}")

        self.lbl_speed.setText(f"{ship.speed_knots:.1f}" if ship.speed_knots else "—")
        self.lbl_heading.setText(f"{ship.heading_deg:.1f}" if ship.heading_deg else "—")
        self.lbl_length.setText(f"{ship.length_m:.1f}" if ship.length_m else "—")

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
        self.lbl_speed.setText("—")
        self.lbl_heading.setText("—")
        self.lbl_length.setText("—")

    def update_ship_list(self, ships: List[DetectedShip]):
        self.ship_list.clear()
        for ship in ships:
            item = QListWidgetItem(f"#{ship.track_id} {ship.class_name}")
            item.setData(Qt.UserRole, ship.track_id)
            color = QColor(*ship.color)
            item.setForeground(color)
            self.ship_list.addItem(item)

    def _on_list_click(self, item: QListWidgetItem):
        track_id = item.data(Qt.UserRole)
        self.highlight_ship.emit(track_id)
