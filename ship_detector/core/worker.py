from PySide6.QtCore import QThread, Signal


class DetectionWorker(QThread):
    """后台推理线程 — 避免 UI 冻结"""
    finished = Signal(list)       # List[DetectedShip]
    error = Signal(str)
    progress = Signal(str)        # 状态文本

    def __init__(self, detector, frame, conf: float, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.frame = frame
        self.conf = conf

    def run(self):
        try:
            self.progress.emit("正在推理...")
            ships = self.detector.detect_frame(self.frame, conf=self.conf)
            self.finished.emit(ships)
        except Exception as e:
            self.error.emit(str(e))
