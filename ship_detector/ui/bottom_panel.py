from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPlainTextEdit, QLabel


class BottomPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(160)
        layout = QHBoxLayout(self)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("系统日志...")

        info = QVBoxLayout()
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_res = QLabel("分辨率: -")
        self.lbl_frame = QLabel("帧号: -")
        info.addWidget(self.lbl_fps)
        info.addWidget(self.lbl_res)
        info.addWidget(self.lbl_frame)
        info.addStretch()

        layout.addWidget(self.log, stretch=3)
        layout.addLayout(info, stretch=1)

    def append_log(self, msg: str, level: str = "INFO"):
        color = {"INFO": "#333", "WARN": "#ff9800", "ERROR": "#f44336"}.get(level, "#333")
        self.log.appendHtml(f'<span style="color:{color}">[{level}] {msg}</span>')

    def update_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def set_resolution(self, w: int, h: int):
        self.lbl_res.setText(f"分辨率: {w}×{h}")

    def set_frame(self, frame_id: int):
        self.lbl_frame.setText(f"帧号: {frame_id}")

    def clear_log(self):
        self.log.clear()
