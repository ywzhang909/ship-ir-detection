from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPlainTextEdit, QLabel
from PySide6.QtCore import Qt


class BottomPanel(QWidget):
    """底部面板：系统日志输出 + 状态信息（FPS / 分辨率 / 帧号）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(180)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("系统日志...")
        self.log_edit.setMaximumBlockCount(500)

        self.lbl_fps = QLabel("FPS: -")
        self.lbl_resolution = QLabel("分辨率: -")
        self.lbl_frame = QLabel("帧号: -")

        right = QVBoxLayout()
        right.addWidget(self.lbl_fps)
        right.addWidget(self.lbl_resolution)
        right.addWidget(self.lbl_frame)
        right.addStretch()

        layout.addWidget(self.log_edit, stretch=3)
        layout.addLayout(right, stretch=1)

    def append_log(self, message: str, level: str = "INFO"):
        from PySide6.QtGui import QTextDocument
        color_map = {"INFO": "black", "WARN": "orange", "ERROR": "red"}
        color = color_map.get(level, "black")
        html = f'<span style="color:{color}">[{level}] {message}</span>'
        self.log_edit.appendHtml(html)

    def update_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def set_resolution(self, w: int, h: int):
        self.lbl_resolution.setText(f"分辨率: {w}x{h}")
