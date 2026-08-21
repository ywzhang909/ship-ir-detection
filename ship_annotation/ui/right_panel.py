from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QListWidget, QListWidgetItem, QComboBox, QLabel,
    QDoubleSpinBox, QLineEdit, QPushButton, QFormLayout,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from core.data_models import TargetObject, BoundingBox


class RightPanel(QWidget):
    """右侧面板：默认类别选择、目标列表、属性编辑器"""

    target_modified = Signal(object)
    target_deleted = Signal(int)
    class_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self._current_target: Optional[TargetObject] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 类别快速选择 ===
        self.class_group = QGroupBox("默认类别")
        class_layout = QVBoxLayout()
        self.combo_default_class = QComboBox()
        self._load_classes(self.combo_default_class)
        class_layout.addWidget(self.combo_default_class)
        self.class_group.setLayout(class_layout)
        layout.addWidget(self.class_group)

        # === 目标列表 ===
        self.list_group = QGroupBox("目标列表")
        list_layout = QVBoxLayout()
        self.target_list = QListWidget()
        list_layout.addWidget(self.target_list)
        self.list_group.setLayout(list_layout)
        layout.addWidget(self.list_group, stretch=1)

        # === 属性编辑 ===
        self.prop_group = QGroupBox("属性")
        prop_layout = QFormLayout()
        self.lbl_id = QLabel("-")
        self.combo_class = QComboBox()
        self._load_classes(self.combo_class)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0, 1)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        self.edit_note = QLineEdit()
        self.btn_delete = QPushButton("🗑 删除目标")
        self.btn_delete.setStyleSheet("color: red;")

        prop_layout.addRow("ID:", self.lbl_id)
        prop_layout.addRow("类别:", self.combo_class)
        prop_layout.addRow("置信度:", self.spin_conf)
        prop_layout.addRow("备注:", self.edit_note)
        prop_layout.addRow(self.btn_delete)
        self.prop_group.setLayout(prop_layout)
        layout.addWidget(self.prop_group)

        self._connect_signals()
        self._set_editor_enabled(False)

    def _load_classes(self, combo: QComboBox):
        from config import PROJECT_CONFIG

        combo.clear()
        for cid, (name, color) in PROJECT_CONFIG.class_map.items():
            combo.addItem(name, cid)

    def _connect_signals(self):
        self.combo_default_class.currentIndexChanged.connect(
            lambda: self.class_changed.emit(self.combo_default_class.currentData())
        )
        self.target_list.itemClicked.connect(self._on_list_clicked)
        self.combo_class.currentIndexChanged.connect(self._on_property_changed)
        self.spin_conf.valueChanged.connect(self._on_property_changed)
        self.edit_note.textChanged.connect(self._on_property_changed)
        self.btn_delete.clicked.connect(self._on_delete)

    # ========== 公共接口 ==========

    def update_target_list(self, targets: List[TargetObject]):
        self.target_list.clear()
        for t in targets:
            item = QListWidgetItem(f"[{t.class_name}] #{t.id} ({t.confidence:.2f})")
            item.setData(Qt.UserRole, t.id)
            color = QColor(*t.color)
            item.setForeground(color)
            self.target_list.addItem(item)

    def set_target(self, target: Optional[TargetObject]):
        self._current_target = target
        if target is None:
            self._set_editor_enabled(False)
            return

        self._set_editor_enabled(True)
        self.lbl_id.setText(str(target.id))
        idx = self.combo_class.findData(target.class_id)
        if idx >= 0:
            self.combo_class.setCurrentIndex(idx)
        self.spin_conf.setValue(target.confidence)
        self.edit_note.setText(target.note)

        for i in range(self.target_list.count()):
            item = self.target_list.item(i)
            if item.data(Qt.UserRole) == target.id:
                self.target_list.setCurrentItem(item)
                break

    # ========== 私有方法 ==========

    def _set_editor_enabled(self, enabled: bool):
        self.combo_class.setEnabled(enabled)
        self.spin_conf.setEnabled(enabled)
        self.edit_note.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)
        if not enabled:
            self.lbl_id.setText("-")

    def _on_list_clicked(self, item: QListWidgetItem):
        target_id = item.data(Qt.UserRole)
        # 通过主窗口中转到画布选中

    def _on_property_changed(self):
        if self._current_target is None:
            return
        updated = TargetObject(
            id=self._current_target.id,
            class_id=self.combo_class.currentData(),
            class_name=self.combo_class.currentText(),
            bbox=self._current_target.bbox,
            confidence=self.spin_conf.value(),
            color=self._current_target.color,
            is_manual=self._current_target.is_manual,
            note=self.edit_note.text(),
        )
        self.target_modified.emit(updated)

    def _on_delete(self):
        if self._current_target:
            self.target_deleted.emit(self._current_target.id)
            self._set_editor_enabled(False)
