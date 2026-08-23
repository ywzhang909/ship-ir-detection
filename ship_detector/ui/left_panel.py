from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton,
    QPushButton, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QStackedWidget,
)
from PySide6.QtCore import Signal, Qt


class LeftPanel(QWidget):
    image_selected = Signal(str)
    video_selected = Signal(str)
    camera_connected = Signal(str)

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
        self.btn_open_img = QPushButton("打开图片")
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        l1.addWidget(self.btn_open_img)
        l1.addWidget(self.file_tree)

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

        # === 识别统计 ===
        stat = QGroupBox("识别统计")
        s_layout = QVBoxLayout()
        self.lbl_count = QLabel("当前画面舰船数: 0")
        self.lbl_count.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        s_layout.addWidget(self.lbl_count)
        stat.setLayout(s_layout)
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
        self.btn_open_vid.clicked.connect(self._open_video)
        self.file_tree.itemClicked.connect(
            lambda item: self.image_selected.emit(item.data(0, Qt.UserRole))
        )
        self.btn_connect.clicked.connect(
            lambda: self.camera_connected.emit(self.edit_url.text())
        )

    def _open_image(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if path:
            self._add_to_tree(path)
            self.image_selected.emit(path)

    def _open_video(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.video_label.setText(path)
            self.video_selected.emit(path)

    def _add_to_tree(self, path: str):
        import os
        item = QTreeWidgetItem([os.path.basename(path)])
        item.setData(0, Qt.UserRole, path)
        self.file_tree.addTopLevelItem(item)

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
