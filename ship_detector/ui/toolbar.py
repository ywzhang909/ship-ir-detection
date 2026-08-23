from PySide6.QtWidgets import QToolBar
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction


class Toolbar(QToolBar):
    run_detection = Signal()
    pause_video = Signal()
    save_result = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.act_detect = QAction("▶ 开始检测", self)
        self.act_detect.triggered.connect(self.run_detection)
        self.addAction(self.act_detect)

        self.act_pause = QAction("⏸ 暂停", self)
        self.act_pause.triggered.connect(self.pause_video)
        self.addAction(self.act_pause)

        self.addSeparator()

        self.act_save = QAction("💾 导出结果", self)
        self.act_save.triggered.connect(self.save_result)
        self.addAction(self.act_save)
