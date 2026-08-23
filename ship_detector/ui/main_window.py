from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QStatusBar
from PySide6.QtCore import Qt
from ui.toolbar import Toolbar
from ui.left_panel import LeftPanel
from ui.central_canvas import CentralCanvas
from ui.right_panel import RightPanel
from ui.bottom_panel import BottomPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舰船识别系统")
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 工具栏
        self.toolbar = Toolbar(self)
        self.addToolBar(self.toolbar)

        # 中部三栏
        self.splitter = QSplitter(Qt.Horizontal)
        self.left_panel = LeftPanel()
        self.canvas = CentralCanvas()
        self.right_panel = RightPanel()

        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([220, 960, 220])

        layout.addWidget(self.splitter, stretch=5)

        # 底部
        self.bottom_panel = BottomPanel()
        layout.addWidget(self.bottom_panel, stretch=1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self._connect_signals()
        self._init_log()

    def _connect_signals(self):
        # 左 -> 画布
        self.left_panel.image_selected.connect(self.canvas.load_image)
        self.left_panel.video_selected.connect(self.canvas.load_video)
        self.left_panel.camera_connected.connect(self.canvas.connect_camera)

        # 工具栏 -> 画布
        self.toolbar.run_detection.connect(self.canvas.run_detection)
        self.toolbar.pause_video.connect(self.canvas.pause)
        self.toolbar.save_result.connect(self._on_save)

        # 画布 -> 右侧面板
        self.canvas.ship_selected.connect(self.right_panel.display_ship)
        self.canvas.ships_updated.connect(self.left_panel.update_ship_count)
        self.canvas.ships_updated.connect(self.right_panel.update_summary)
        self.canvas.status_message.connect(self.status_bar.showMessage)
        self.canvas.fps_updated.connect(self.bottom_panel.update_fps)
        self.canvas.resolution_changed.connect(self.bottom_panel.set_resolution)

        # 画布 ships_updated 同步更新目标列表
        self.canvas.ships_updated.connect(
            lambda c: self.right_panel.update_ship_list(self.canvas.ships)
        )

        # 右侧面板 -> 画布
        self.right_panel.highlight_ship.connect(self.canvas.highlight_ship)

    def _init_log(self):
        self.bottom_panel.append_log("系统启动", "INFO")
        self.bottom_panel.append_log("欢迎使用舰船识别系统", "INFO")

    def _on_save(self):
        self.bottom_panel.append_log("导出结果功能开发中...", "INFO")
        self.status_bar.showMessage("导出功能开发中")
