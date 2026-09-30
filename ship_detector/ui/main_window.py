import logging
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QStatusBar
from PySide6.QtCore import Qt
from ui.toolbar import Toolbar
from ui.left_panel import LeftPanel
from ui.central_canvas import CentralCanvas
from ui.right_panel import RightPanel
from ui.bottom_panel import BottomPanel
from core.yolo_detector import YoloDetector
from core.worker import DetectionWorker, VideoDetectWorker
from config import DEFAULT_MODEL_PATH, DEFAULT_CONFIDENCE

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舰船识别系统")
        self.setGeometry(100, 100, 1400, 900)

        # 检测器
        self.detector = YoloDetector()
        self._worker: DetectionWorker = None
        self._detecting = False

        # 连续检测（播放中逐帧识别）
        self._continuous = False
        self._detection_generation = 0
        self.detect_worker = VideoDetectWorker(self.detector, DEFAULT_CONFIDENCE)

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
        self._load_model()

    def _load_model(self):
        """启动时加载检测模型"""
        self.bottom_panel.append_log(f"加载模型: {DEFAULT_MODEL_PATH}", "INFO")
        ok = self.detector.load_model(DEFAULT_MODEL_PATH)
        self.left_panel.set_detector_status(ok)
        if ok:
            self.bottom_panel.append_log("检测器就绪", "INFO")
            self.status_bar.showMessage("检测器已加载")
        else:
            self.bottom_panel.append_log("检测器加载失败 — 请检查模型文件", "ERROR")
            self.status_bar.showMessage("检测器加载失败")

    def _connect_signals(self):
        # 工具栏 -> 功能
        self.toolbar.open_file.connect(self._on_toolbar_open)
        self.toolbar.run_detection.connect(self._run_detection)
        self.toolbar.pause_video.connect(self.canvas.pause)
        self.toolbar.save_result.connect(self._on_save)

        # 左 -> 画布
        self.left_panel.image_selected.connect(self.canvas.load_image)
        self.left_panel.video_selected.connect(self.canvas.load_video)
        self.left_panel.camera_connected.connect(self.canvas.connect_camera)

        # 置信度变化
        self.left_panel.confidence_changed.connect(self._on_confidence_changed)

        # SAHI 切换
        self.left_panel.sahi_toggled.connect(self._on_sahi_toggled)

        # 画布 -> 右侧面板
        self.canvas.ship_selected.connect(self.right_panel.display_ship)
        self.canvas.ships_updated.connect(self.left_panel.update_ship_count)
        self.canvas.ships_updated.connect(self.right_panel.update_summary)
        self.canvas.status_message.connect(self.status_bar.showMessage)
        self.canvas.resolution_changed.connect(self._on_resolution_changed)
        self.canvas.fps_updated.connect(self.bottom_panel.update_fps)

        # 画布 ships_updated 同步更新目标列表
        self.canvas.ships_updated.connect(
            lambda c: self.right_panel.update_ship_list(self.canvas.ships)
        )

        # 右侧面板 -> 画布 (列表点击 -> 选中 + 详情)
        self.right_panel.highlight_ship.connect(self._on_list_highlight)

        # 数据源状态
        self.left_panel.image_selected.connect(
            lambda p: self.left_panel.set_source_status(f"图片: {p}", True)
        )
        self.left_panel.video_selected.connect(
            lambda p: self.left_panel.set_source_status(f"视频: {p}", True)
        )
        self.left_panel.camera_connected.connect(
            lambda u: self.left_panel.set_source_status(f"相机: {u}", True)
        )

        # 连续检测：画布出帧 -> 直接调用 worker.submit（worker 无线程事件循环，
        # 队列连接永远不会被派发）；worker 结果 -> 队列连接回 UI 线程。
        self.canvas.frame_captured.connect(self._on_frame_captured)
        self.detect_worker.result_ready.connect(self._on_continuous_result)
        self.toolbar.continuous_toggled.connect(self.set_continuous_detection)

        # 切换数据源时递增代次，丢弃上一段视频迟到回来的结果
        for sig in (self.left_panel.image_selected,
                    self.left_panel.video_selected,
                    self.left_panel.camera_connected):
            sig.connect(lambda *_: self.set_detection_generation(
                self.detect_worker.advance_generation()))

    def _on_toolbar_open(self):
        """工具栏打开按钮 — 根据当前源类型打开对应对话框"""
        if self.left_panel.rb_image.isChecked():
            self.left_panel._open_image()
        elif self.left_panel.rb_video.isChecked():
            self.left_panel._open_video()
        else:
            self.left_panel.btn_connect.click()

    def _on_sahi_toggled(self, enabled: bool):
        self.detector.set_sahi(enabled)
        mode = "SAHI 切片" if enabled else "标准"
        self.bottom_panel.append_log(f"推理模式: {mode}", "INFO")
        self.status_bar.showMessage(f"切换到 {mode} 推理")

    def _on_resolution_changed(self, w: int, h: int):
        self.bottom_panel.set_resolution(w, h)
        self.right_panel.set_image_size(w, h)

    def _on_list_highlight(self, track_id: int):
        """列表点击 -> 画布高亮 + 详情面板更新"""
        self.canvas.highlight_ship(track_id)
        for ship in self.canvas.ships:
            if ship.track_id == track_id:
                self.right_panel.display_ship(ship)
                break

    def set_continuous_detection(self, enabled: bool) -> None:
        """开关「播放中连续识别」。

        只停止投递帧、不销毁 worker 线程: QThread 结束后无法重启, 频繁开关
        会把自己锁死; 线程空转等待下一帧的代价可以忽略。
        """
        enabled = bool(enabled)
        self._continuous = enabled
        self.toolbar.set_continuous(enabled)

        # 检测器可能被外部替换过, 投递前统一同步引用
        self.detect_worker.detector = self.detector

        if enabled:
            if not self.detect_worker.isRunning():
                self.detect_worker.start()
            self.set_detection_generation(self.detect_worker.advance_generation())
            self.bottom_panel.append_log("连续检测: 开", "INFO")
            self.status_bar.showMessage("连续检测已开启 — 播放视频即逐帧识别")
        else:
            self.detect_worker.clear_pending()
            self.bottom_panel.append_log("连续检测: 关", "INFO")

    def set_detection_generation(self, generation: int) -> None:
        self._detection_generation = int(generation)

    def _on_frame_captured(self, frame, frame_id: int) -> None:
        """画布解码出一帧 — 直接调用 submit（worker 没有 Qt 事件循环）。"""
        if not self._continuous:
            return
        self.detect_worker.detector = self.detector
        self.detect_worker.submit(
            frame, frame_id, self.left_panel.get_confidence(),
            self._detection_generation,
        )

    def _on_continuous_result(self, generation: int, ships: list) -> None:
        """worker 结果回到 UI 线程; 丢弃来自上一段视频的迟到结果。"""
        if generation != self._detection_generation:
            return
        self.canvas.set_ships(ships)

    def _run_detection(self):
        """执行检测 — 在后台线程运行"""
        if self._continuous:
            self.bottom_panel.append_log("连续检测进行中，已忽略单帧检测", "WARN")
            return
        if self._detecting:
            return

        if not self.detector._is_loaded:
            self.status_bar.showMessage("检测器未加载")
            self.bottom_panel.append_log("检测器未加载，无法检测", "WARN")
            return

        frame = self.canvas.current_frame
        if frame is None:
            self.status_bar.showMessage("请先加载图片")
            self.bottom_panel.append_log("无检测目标，请先加载图片", "WARN")
            return

        conf = self.left_panel.get_confidence()
        self._detecting = True
        self.toolbar.set_detecting(True)

        self._worker = DetectionWorker(self.detector, frame, conf)
        self._worker.finished.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    def _on_detection_done(self, ships: list):
        self._detecting = False
        self.toolbar.set_detecting(False)
        self.canvas.set_ships(ships)
        self.bottom_panel.append_log(f"检测完成: {len(ships)} 个目标", "INFO")
        self.status_bar.showMessage(f"检测完成: {len(ships)} 个目标")

        for s in ships:
            self.bottom_panel.append_log(
                f"  #{s.track_id} {s.class_name} conf={s.confidence:.1%}", "INFO"
            )

    def _on_detection_error(self, err: str):
        self._detecting = False
        self.toolbar.set_detecting(False)
        self.bottom_panel.append_log(f"检测失败: {err}", "ERROR")
        self.status_bar.showMessage("检测失败")

    def _on_confidence_changed(self, val: float):
        self.bottom_panel.append_log(f"置信度阈值: {val:.2f}", "INFO")

    def _on_save(self):
        """导出检测结果"""
        ships = self.canvas.ships
        if not ships:
            self.bottom_panel.append_log("无检测结果可导出", "WARN")
            return

        from PySide6.QtWidgets import QFileDialog
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "detection_result.json", "JSON (*.json)"
        )
        if not path:
            return

        data = []
        for s in ships:
            data.append({
                "track_id": s.track_id,
                "class_id": s.class_id,
                "class_name": s.class_name,
                "confidence": s.confidence,
                "bbox": {"x": s.bbox.x, "y": s.bbox.y, "w": s.bbox.w, "h": s.bbox.h},
                "note": s.note,
                "reviewed": s.reviewed,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.bottom_panel.append_log(f"已导出: {path} ({len(ships)} 个目标)", "INFO")
        self.status_bar.showMessage(f"已导出 {path}")

    def _init_log(self):
        self.bottom_panel.append_log("系统启动", "INFO")
        self.bottom_panel.append_log("欢迎使用舰船识别系统", "INFO")

    def closeEvent(self, event):
        """退出时清理"""
        self.detect_worker.shutdown()
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self.canvas.close()
        super().closeEvent(event)
