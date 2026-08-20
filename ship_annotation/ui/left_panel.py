import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QPushButton, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QStackedWidget, QFileDialog,
)
from PySide6.QtCore import Signal, Qt


class LeftPanel(QWidget):
    """左侧面板：数据源选择、文件树、相机控制、设备状态"""

    image_selected = Signal(str)
    camera_connected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 源选择 ===
        self.source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout()
        self.rb_image = QRadioButton("本地图片")
        self.rb_camera = QRadioButton("相机/视频流")
        self.rb_image.setChecked(True)
        source_layout.addWidget(self.rb_image)
        source_layout.addWidget(self.rb_camera)
        self.source_group.setLayout(source_layout)
        layout.addWidget(self.source_group)

        # === 动态内容区 ===
        self.stack = QStackedWidget()

        # -- 图片页 --
        self.page_image = QWidget()
        img_layout = QVBoxLayout(self.page_image)
        self.btn_open = QPushButton("📁 打开文件夹")
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setColumnCount(1)
        img_layout.addWidget(self.btn_open)
        img_layout.addWidget(self.file_tree)

        # -- 相机页 --
        self.page_camera = QWidget()
        cam_layout = QVBoxLayout(self.page_camera)
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("rtsp://... 或 0")
        self.btn_connect = QPushButton("🔌 连接")
        self.lbl_cam_status = QLabel("● 未连接")
        self.lbl_cam_status.setStyleSheet("color: gray;")
        cam_layout.addWidget(QLabel("地址:"))
        cam_layout.addWidget(self.edit_url)
        cam_layout.addWidget(self.btn_connect)
        cam_layout.addWidget(self.lbl_cam_status)
        cam_layout.addStretch()

        self.stack.addWidget(self.page_image)
        self.stack.addWidget(self.page_camera)
        layout.addWidget(self.stack)

        # === 设备状态 ===
        self.status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        self.lbl_detector = QLabel("🤖 检测器: 未加载")
        self.lbl_source = QLabel("📷 数据源: 未连接")
        status_layout.addWidget(self.lbl_detector)
        status_layout.addWidget(self.lbl_source)
        self.status_group.setLayout(status_layout)
        layout.addWidget(self.status_group)

        layout.addStretch()
        self._connect_signals()

    def _connect_signals(self):
        self.rb_image.toggled.connect(lambda: self.stack.setCurrentIndex(0))
        self.rb_camera.toggled.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_open.clicked.connect(self._on_open_folder)
        self.file_tree.itemClicked.connect(
            lambda item: self.image_selected.emit(item.data(0, Qt.UserRole))
        )
        self.btn_connect.clicked.connect(
            lambda: self.camera_connected.emit(self.edit_url.text())
        )

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.file_tree.clear()
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                item = QTreeWidgetItem([f])
                item.setData(0, Qt.UserRole, os.path.join(folder, f))
                self.file_tree.addTopLevelItem(item)

    def set_detector_status(self, loaded: bool):
        color = "green" if loaded else "gray"
        text = "已就绪" if loaded else "未加载"
        self.lbl_detector.setText(
            f"🤖 检测器: <span style='color:{color}'>{text}</span>"
        )

    def set_source_status(self, connected: bool, text: str = ""):
        color = "green" if connected else "gray"
        self.lbl_source.setText(
            f"📷 数据源: <span style='color:{color}'>{text}</span>"
        )
