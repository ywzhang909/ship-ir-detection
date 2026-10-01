from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPlainTextEdit, QLabel, QPushButton
from PySide6.QtCore import Signal


class BottomPanel(QWidget):
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(160)
        layout = QHBoxLayout(self)

        # 左侧: 日志 + 清空按钮
        left = QVBoxLayout()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("系统日志...")
        left.addWidget(self.log)

        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("清空日志")
        self.btn_clear.setFixedWidth(80)
        self.btn_clear.clicked.connect(self.clear_log)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_clear)
        left.addLayout(btn_row)

        layout.addLayout(left, stretch=3)

        # 右侧: 信息标签
        info = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_res = QLabel("分辨率: -")
        self.lbl_frame = QLabel("帧号: -")
        info.addWidget(self.lbl_fps)
        info.addWidget(self.lbl_res)
        info.addWidget(self.lbl_frame)
        info.addStretch()

        layout.addLayout(info, stretch=1)

    def append_log(self, msg: str, level: str = "INFO"):
        color = {"INFO": "#333", "WARN": "#ff9800", "ERROR": "#f44336"}.get(level, "#333")
        self.log.appendHtml(f'<span style="color:{color}">[{level}] {msg}</span>')

    def update_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def set_resolution(self, w: int, h: int):
        self.lbl_res.setText(f"分辨率: {w}x{h}")

    def set_frame(self, frame_id: int):
        self.lbl_frame.setText(f"帧号: {frame_id}")

    def clear_log(self):
        self.log.clear()
        self.clear_requested.emit()
