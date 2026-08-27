import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton,
    QPushButton, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QStackedWidget, QSlider, QHBoxLayout,
    QFileDialog,
)
from PySide6.QtCore import Signal, Qt


class LeftPanel(QWidget):
    image_selected = Signal(str)
    video_selected = Signal(str)
    camera_connected = Signal(str)
    confidence_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # === 源选择 ===
        group = QGroupBox("数据源")
        g_layout = QVBoxLayout()
        self.rb_image = QRadioButton("🖼 本地图片")
        self.rb_video = QRadioButton("🎬 视频文件")
        self.rb_camera = QRadioButton("📷 相机/流")
        self.rb_image.setChecked(True)
        g_layout.addWidget(self.rb_image)
        g_layout.addWidget(self.rb_video)
        g_layout.addWidget(self.rb_camera)
        group.setLayout(g_layout)
        layout.addWidget(group)

        # === 动态内容 ===
        self.stack = QStackedWidget()

        # 图片页
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        btn_row = QHBoxLayout()
        self.btn_open_img = QPushButton("打开图片")
        self.btn_open_folder = QPushButton("打开文件夹")
        btn_row.addWidget(self.btn_open_img)
        btn_row.addWidget(self.btn_open_folder)
        l1.addLayout(btn_row)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        l1.addWidget(self.file_tree)
        # 预览缩略图
        self.preview_label = QLabel("预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(120)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        l1.addWidget(self.preview_label)

        # 视频页
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        self.btn_open_vid = QPushButton("打开视频")
        self.video_label = QLabel("未选择视频")
        self.video_label.setWordWrap(True)
        l2.addWidget(self.btn_open_vid)
        l2.addWidget(self.video_label)
        l2.addStretch()

        # 相机页
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("0 或 rtsp://...")
        self.btn_connect = QPushButton("🔌 连接")
        self.lbl_cam = QLabel("● 未连接")
        self.lbl_cam.setStyleSheet("color: gray;")
        l3.addWidget(QLabel("设备号 / RTSP地址:"))
        l3.addWidget(self.edit_url)
        l3.addWidget(self.btn_connect)
        l3.addWidget(self.lbl_cam)
        l3.addStretch()

        self.stack.addWidget(p1)
        self.stack.addWidget(p2)
        self.stack.addWidget(p3)
        layout.addWidget(self.stack)

        # === 检测设置 ===
        settings = QGroupBox("检测设置")
        s_layout = QVBoxLayout()

        # 置信度滑块
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(45)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.lbl_conf = QLabel("0.45")
        self.lbl_conf.setMinimumWidth(35)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.lbl_conf)
        s_layout.addLayout(conf_row)

        # SAHI 开关
        self.chk_sahi = QPushButton("SAHI: 关")
        self.chk_sahi.setCheckable(True)
        self.chk_sahi.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: white; }")
        s_layout.addWidget(self.chk_sahi)

        settings.setLayout(s_layout)
        layout.addWidget(settings)

        # === 识别统计 ===
        stat = QGroupBox("识别统计")
        st_layout = QVBoxLayout()
        self.lbl_count = QLabel("当前画面舰船数: 0")
        self.lbl_count.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        st_layout.addWidget(self.lbl_count)
        stat.setLayout(st_layout)
        layout.addWidget(stat)

        # === 系统状态 ===
        sys_group = QGroupBox("系统状态")
        sys_layout = QVBoxLayout()
        self.lbl_detector = QLabel("检测器: 未加载")
        self.lbl_source = QLabel("数据源: 未连接")
        sys_layout.addWidget(self.lbl_detector)
        sys_layout.addWidget(self.lbl_source)
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)

        layout.addStretch()
        self._connect_signals()

    def _connect_signals(self):
        self.rb_image.toggled.connect(lambda c: c and self.stack.setCurrentIndex(0))
        self.rb_video.toggled.connect(lambda c: c and self.stack.setCurrentIndex(1))
        self.rb_camera.toggled.connect(lambda c: c and self.stack.setCurrentIndex(2))

        self.btn_open_img.clicked.connect(self._open_image)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_open_vid.clicked.connect(self._open_video)
        self.file_tree.itemClicked.connect(self._on_tree_click)
        self.btn_connect.clicked.connect(
            lambda: self.camera_connected.emit(self.edit_url.text())
        )

        # 置信度
        self.conf_slider.valueChanged.connect(
            lambda v: (self.lbl_conf.setText(f"{v/100:.2f}"),
                       self.confidence_changed.emit(v / 100))
        )

        # SAHI
        self.chk_sahi.clicked.connect(self._toggle_sahi)

    def _on_tree_click(self, item):
        path = item.data(0, Qt.UserRole)
        if path:
            self.image_selected.emit(path)
            self._show_preview(path)

    def _show_preview(self, path: str):
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if path:
            self._add_to_tree(path)
            self.image_selected.emit(path)
            self._show_preview(path)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )

        if not files:
            return

        self.file_tree.clear()
        for fp in files:
            self._add_to_tree(fp)

        # 自动选中第一张
        if files:
            self.image_selected.emit(files[0])
            self._show_preview(files[0])

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.video_label.setText(path)
            self.video_selected.emit(path)

    def _add_to_tree(self, path: str):
        item = QTreeWidgetItem([os.path.basename(path)])
        item.setData(0, Qt.UserRole, path)
        self.file_tree.addTopLevelItem(item)

    def get_confidence(self) -> float:
        return self.conf_slider.value() / 100.0

    def _toggle_sahi(self):
        checked = self.chk_sahi.isChecked()
        self.chk_sahi.setText(f"SAHI: {'开' if checked else '关'}")

    def update_ship_count(self, count: int):
        self.lbl_count.setText(f"当前画面舰船数: {count}")

    def set_detector_status(self, ready: bool):
        color = "green" if ready else "gray"
        text = "已就绪" if ready else "未加载"
        self.lbl_detector.setText(
            f'检测器: <span style="color:{color}">{text}</span>'
        )

    def set_source_status(self, text: str, ok: bool = True):
        color = "green" if ok else "gray"
        self.lbl_source.setText(
            f'数据源: <span style="color:{color}">{text}</span>'
        )
