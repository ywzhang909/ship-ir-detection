from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt
from ui.toolbar import Toolbar
from ui.left_panel import LeftPanel
from ui.central_canvas import CentralCanvas
from ui.right_panel import RightPanel
from ui.bottom_panel import BottomPanel
from core.data_models import AnnotationTool, TargetObject
from app import App


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("舰船识别标注系统 v0.1")
        self.setGeometry(100, 100, 1400, 900)

        self._app = App()

        # 中央容器
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)

        # 顶部工具栏
        self.toolbar = Toolbar(self)
        self.addToolBar(self.toolbar)

        # 中部水平分割器
        self.h_splitter = QSplitter(Qt.Horizontal)
        self.left_panel = LeftPanel()
        self.canvas = CentralCanvas()
        self.right_panel = RightPanel()

        self.h_splitter.addWidget(self.left_panel)
        self.h_splitter.addWidget(self.canvas)
        self.h_splitter.addWidget(self.right_panel)
        self.h_splitter.setSizes([220, 960, 220])

        # 底部面板
        self.bottom_panel = BottomPanel()

        self.main_layout.addWidget(self.h_splitter, stretch=5)
        self.main_layout.addWidget(self.bottom_panel, stretch=1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self._connect_signals()
        self._init_logging()

    def _connect_signals(self):
        """所有信号槽连接集中在此"""
        # 工具栏 -> 画布
        self.toolbar.tool_changed.connect(self.canvas.set_current_tool)
        self.toolbar.run_detection.connect(self._on_run_detection)
        self.toolbar.save_project.connect(self._on_save)

        # 左侧面板 -> 画布
        self.left_panel.image_selected.connect(self.canvas.load_image)
        self.left_panel.camera_connected.connect(self.canvas.connect_camera)

        # 画布 -> 右侧面板
        self.canvas.target_selected.connect(self.right_panel.set_target)
        self.canvas.targets_updated.connect(self.right_panel.update_target_list)
        self.canvas.status_message.connect(self.status_bar.showMessage)
        self.canvas.fps_updated.connect(self.bottom_panel.update_fps)

        # 右侧面板 -> 画布
        self.right_panel.target_modified.connect(self.canvas.update_target)
        self.right_panel.target_deleted.connect(self.canvas.delete_target)
        self.right_panel.class_changed.connect(self.canvas.set_default_class)

    def _init_logging(self):
        """初始化日志到下方日志面板"""
        self.bottom_panel.append_log("系统启动", "INFO")
        self.bottom_panel.append_log("欢迎使用舰船识别标注系统 v0.1", "INFO")

        # 检查检测器
        self.left_panel.set_detector_status(self._app.detector.is_loaded)
        self.left_panel.set_source_status(False, "未连接")

    def _on_run_detection(self):
        """执行检测"""
        if self.canvas.display_pixmap.isNull():
            self.bottom_panel.append_log("请先加载图片", "WARN")
            return

        if not self._app.detector.is_loaded:
            self.bottom_panel.append_log("检测器未加载，请先加载模型", "WARN")
            self.status_bar.showMessage("检测器未加载")
            return

        self.bottom_panel.append_log("开始检测...", "INFO")
        self.status_bar.showMessage("正在检测...")

        # 执行检测
        current_path = self._app.frame_source.current_path
        if current_path:
            targets = self._app.detector.detect(
                current_path,
                conf=self._app.config.confidence_threshold,
            )
            self.canvas.targets = targets
            self.canvas.update()
            self.canvas.targets_updated.emit(targets)
            self.bottom_panel.append_log(f"检测完成，找到 {len(targets)} 个目标", "INFO")
            self.status_bar.showMessage(f"检测完成: {len(targets)} 个目标")
        else:
            self.bottom_panel.append_log("无当前图片路径", "WARN")

    def _on_save(self):
        """保存项目"""
        self.bottom_panel.append_log("保存功能开发中...", "INFO")
        self.status_bar.showMessage("保存功能开发中")
