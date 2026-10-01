from PySide6.QtWidgets import QToolBar
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QKeySequence


class Toolbar(QToolBar):
    open_file = Signal()
    run_detection = Signal()
    pause_video = Signal()
    save_result = Signal()
    continuous_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.act_open = QAction("打开", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self.open_file)
        self.addAction(self.act_open)

        self.addSeparator()

        self.act_detect = QAction("开始检测", self)
        self.act_detect.setShortcut(QKeySequence("F5"))
        self.act_detect.triggered.connect(self.run_detection)
        self.addAction(self.act_detect)

        self.act_continuous = QAction("连续检测", self)
        self.act_continuous.setCheckable(True)
        self.act_continuous.setToolTip("播放时逐帧实时识别，画面持续更新检测框")
        self.act_continuous.toggled.connect(self.continuous_toggled)
        self.addAction(self.act_continuous)

        self.act_pause = QAction("暂停", self)
        self.act_pause.setShortcut(QKeySequence("Space"))
        self.act_pause.triggered.connect(self.pause_video)
        self.act_pause.setEnabled(False)
        self.addAction(self.act_pause)

        self.addSeparator()

        self.act_save = QAction("导出结果", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self.save_result)
        self.addAction(self.act_save)

    def set_continuous(self, enabled: bool):
        """同步按钮勾选态；屏蔽信号避免与 MainWindow 形成回环。"""
        self.act_continuous.blockSignals(True)
        self.act_continuous.setChecked(bool(enabled))
        self.act_continuous.blockSignals(False)

    def set_detecting(self, detecting: bool):
        """切换检测/暂停状态"""
        if detecting:
            self.act_detect.setText("检测中...")
            self.act_detect.setEnabled(False)
            self.act_pause.setEnabled(True)
        else:
            self.act_detect.setText("开始检测")
            self.act_detect.setEnabled(True)
            self.act_pause.setEnabled(False)
