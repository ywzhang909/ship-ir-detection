# 舰船识别标注系统 — 界面开发手册（MVP版）

> **目标**：本手册面向 AI 代码生成，要求结构清晰、命名规范、依赖单一，确保任何遵循此手册的 AI 都能稳定输出可运行的 PySide6 代码。

---

## 一、技术栈与约束

| 项目 | 选择 | 理由 |
|------|------|------|
| GUI 框架 | **PySide6** | Qt6 官方绑定，LGPL 许可，组件丰富 |
| 视频/图像渲染 | **QLabel + QPainter** | 2D 绘制足够，无需 OpenGL 复杂度 |
| 图表/监控 | **PySide6.QtCharts** | 原生集成，无需额外依赖 |
| 检测后端 | 抽象接口 + YOLO 示例 | 解耦 UI 与算法 |
| 配置存储 | JSON | 零依赖，人可读 |
| 代码风格 | PEP8，类名 CamelCase，变量/函数 snake_case |

**禁止引入**：DearPyGui、PyQtGraph、OpenCV 的 `cv2.imshow`（可用 cv2 做图像处理，但显示必须用 Qt）。

---

## 二、项目目录结构

```
ship_annotation/
├── main.py                  # 入口，创建 QApplication
├── config.py                # 全局常量、路径配置
├── app.py                   # 应用级逻辑（单例）
├── core/
│   ├── __init__.py
│   ├── data_models.py       # 所有数据模型（@dataclass）
│   ├── detector_base.py     # 检测器抽象基类
│   ├── yolo_detector.py     # YOLO 检测器实现（示例）
│   └── frame_source.py      # 图像源抽象（图片/相机）
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # QMainWindow，组装所有面板
│   ├── toolbar.py           # 顶部工具栏
│   ├── left_panel.py        # 左侧面板（源选择+文件树）
│   ├── central_canvas.py    # 中央画布（核心交互区）
│   ├── right_panel.py       # 右侧面板（目标列表+属性）
│   └── bottom_panel.py      # 底部面板（日志+状态）
├── resources/
│   ├── icons/               # 图标文件（.png/.svg）
│   └── styles/
│       └── main.qss         # 全局样式表
└── tests/
    └── test_models.py
```

---

## 三、数据模型（`core/data_models.py`）

所有数据模型使用 `@dataclass`，**不可**使用字典传递结构化数据。

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from datetime import datetime


class SourceType(Enum):
    IMAGE = auto()      # 单张/文件夹图片
    CAMERA = auto()     # 相机/视频流


class AnnotationTool(Enum):
    SELECT = auto()     # 选择/拖拽
    RECTANGLE = auto()  # 矩形标注
    ARROW = auto()      # 方向箭头
    DELETE = auto()     # 删除模式


@dataclass
class BoundingBox:
    """归一化坐标 0.0~1.0，避免分辨率耦合"""
    x: float          # 左上角 x
    y: float          # 左上角 y
    w: float          # 宽度
    h: float          # 高度
    
    def to_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """转为像素坐标 (x1, y1, x2, y2)"""
        x1 = int(self.x * img_w)
        y1 = int(self.y * img_h)
        x2 = int((self.x + self.w) * img_w)
        y2 = int((self.y + self.h) * img_h)
        return x1, y1, x2, y2


@dataclass
class TargetObject:
    """画面中的一个检测/标注目标"""
    id: int
    class_id: int
    class_name: str                    # 如 "船只", "飞机"
    bbox: BoundingBox
    confidence: float = 0.0            # 检测置信度 0~1
    color: Tuple[int, int, int] = (255, 0, 0)  # BGR 显示颜色
    is_manual: bool = False            # True=人工标注, False=AI检测
    note: str = ""                     # 用户备注
    arrow_end: Optional[Tuple[float, float]] = None  # 方向箭头终点（归一化）


@dataclass
class FrameData:
    """一帧画面的完整数据"""
    frame_id: int
    timestamp: datetime
    source_type: SourceType
    image_path: Optional[str] = None   # 图片模式路径
    pixmap: Optional[object] = None  # QPixmap（运行时不序列化）
    targets: List[TargetObject] = field(default_factory=list)


@dataclass
class ProjectConfig:
    """项目配置"""
    name: str = "未命名项目"
    class_map: dict = field(default_factory=lambda: {
        0: ("船只", (0, 0, 255)),      # 红
        1: ("飞机", (0, 255, 0)),      # 绿
        2: ("浮标", (255, 255, 0)),    # 黄
    })
    confidence_threshold: float = 0.5
    auto_save: bool = True
```

---

## 四、UI 模块详细规范

### 4.1 主窗口（`ui/main_window.py`）

```python
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStatusBar
)
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    """
    主窗口，使用 QSplitter 实现面板可拖拽调整。
    布局：
        顶部：Toolbar
        中部：水平三栏（左|中|右），用 QSplitter
        底部：BottomPanel
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舰船识别标注系统 v0.1")
        self.setGeometry(100, 100, 1400, 900)
        
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
        self.h_splitter.setSizes([220, 960, 220])  # 初始比例
        
        # 底部面板
        self.bottom_panel = BottomPanel()
        
        self.main_layout.addWidget(self.h_splitter, stretch=5)
        self.main_layout.addWidget(self.bottom_panel, stretch=1)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        self._connect_signals()
    
    def _connect_signals(self):
        """所有信号槽连接集中在此"""
        # 工具栏 -> 画布
        self.toolbar.tool_changed.connect(self.canvas.set_current_tool)
        self.toolbar.run_detection.connect(self.canvas.run_detection)
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
```

### 4.2 左侧面板（`ui/left_panel.py`）

**UI 内容**：
- **源选择区**（QGroupBox）：两个 QRadioButton（本地图片 / 相机流）
- **图片模式**：QPushButton("打开文件夹") + QTreeWidget（文件树）
- **相机模式**：QLineEdit（RTSP 地址）+ QPushButton("连接") + QLabel（状态灯）
- **设备状态**：QWidget，含两个 QLabel（状态图标 + 文字）

**交互**：
- 切换 RadioButton 时，动态显示/隐藏对应控件
- 点击文件树项，发射 `image_selected(str: path)`
- 点击连接，发射 `camera_connected(str: url)`

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QPushButton, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QStackedWidget
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QColor

class LeftPanel(QWidget):
    image_selected = Signal(str)      # 图片路径
    camera_connected = Signal(str)    # RTSP 地址
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self._setup_ui()
    
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        
        # === 源选择 ===
        self.source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout()
        self.rb_image = QRadioButton("本地图片")
        self.rb_camera = QRadioButton("相机/视频流")
        self.rb_image.setChecked(True)
        source_layout.addWidget(self.rb_image)
        source_layout.addWidget(self.rb_camera)
        self.source_group.setLayout(source_layout)
        self.layout.addWidget(self.source_group)
        
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
        self.layout.addWidget(self.stack)
        
        # === 设备状态 ===
        self.status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        self.lbl_detector = QLabel("🤖 检测器: 未加载")
        self.lbl_source = QLabel("📷 数据源: 未连接")
        status_layout.addWidget(self.lbl_detector)
        status_layout.addWidget(self.lbl_source)
        self.status_group.setLayout(status_layout)
        self.layout.addWidget(self.status_group)
        
        self.layout.addStretch()
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
        """弹出 QFileDialog，填充文件树"""
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.file_tree.clear()
        import os
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                item = QTreeWidgetItem([f])
                item.setData(0, Qt.UserRole, os.path.join(folder, f))
                self.file_tree.addTopLevelItem(item)
    
    def set_detector_status(self, loaded: bool):
        color = "green" if loaded else "gray"
        text = "已就绪" if loaded else "未加载"
        self.lbl_detector.setText(f"🤖 检测器: <span style='color:{color}'>{text}</span>")
    
    def set_source_status(self, connected: bool, text: str = ""):
        color = "green" if connected else "gray"
        self.lbl_source.setText(f"📷 数据源: <span style='color:{color}'>{text}</span>")
```

### 4.3 中央画布（`ui/central_canvas.py`）

**核心设计**：
- 继承 `QLabel`，重写 `paintEvent`
- 维护一个 `self.display_pixmap: QPixmap`（原始画面）
- 维护 `self.targets: List[TargetObject]`（当前帧目标）
- 鼠标事件处理：框选、拖拽、选中

**坐标系统**：
- 所有内部坐标使用**归一化**（0.0~1.0）
- 绘制时根据 `self.width()`、`self.height()` 转为像素
- 保持宽高比，黑边填充（letterbox）

```python
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QPoint, QRect
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QPaintEvent

class CentralCanvas(QLabel):
    # 信号定义
    target_selected = Signal(object)      # TargetObject
    targets_updated = Signal(list)        # List[TargetObject]
    status_message = Signal(str)
    fps_updated = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        
        # 状态
        self.current_tool = AnnotationTool.SELECT
        self.display_pixmap: QPixmap = QPixmap(640, 480)
        self.display_pixmap.fill(QColor("#1a1a1a"))
        self.targets: List[TargetObject] = []
        self.selected_target: Optional[TargetObject] = None
        self.default_class_id = 0
        
        # 鼠标交互状态
        self._mouse_pressed = False
        self._drag_start = QPoint()
        self._drag_current = QPoint()
        self._is_dragging_target = False
        
        self.setMouseTracking(True)
    
    def set_current_tool(self, tool: AnnotationTool):
        self.current_tool = tool
        self.setCursor(Qt.CrossCursor if tool != AnnotationTool.SELECT else Qt.ArrowCursor)
    
    def set_default_class(self, class_id: int):
        self.default_class_id = class_id
    
    def load_image(self, path: str):
        """加载本地图片"""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.status_message.emit(f"无法加载图片: {path}")
            return
        self.display_pixmap = pixmap
        self.targets = []  # 新图片清空标注
        self.update()
        self.status_message.emit(f"已加载: {path}")
        self.targets_updated.emit(self.targets)
    
    def connect_camera(self, url: str):
        """连接相机（占位，实际用 QTimer + cv2/线程）"""
        self.status_message.emit(f"正在连接: {url}...")
        # TODO: 启动视频线程，帧到达时调用 self.update_frame()
    
    def update_frame(self, pixmap: QPixmap, detections: List[TargetObject]):
        """视频线程回调，更新画面"""
        self.display_pixmap = pixmap
        self.targets = detections
        self.update()
        self.targets_updated.emit(self.targets)
    
    def run_detection(self):
        """触发检测（调用 core/detector）"""
        self.status_message.emit("正在检测...")
        # TODO: 调用检测器，结果填充 self.targets
        self.update()
        self.targets_updated.emit(self.targets)
    
    def update_target(self, target: TargetObject):
        """右侧面板修改后回写"""
        for i, t in enumerate(self.targets):
            if t.id == target.id:
                self.targets[i] = target
                break
        self.update()
    
    def delete_target(self, target_id: int):
        self.targets = [t for t in self.targets if t.id != target_id]
        self.selected_target = None
        self.update()
        self.targets_updated.emit(self.targets)
    
    # ========== 绘制 ==========
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        
        # 1. 计算 letterbox 区域
        img_rect = self._calc_fit_rect()
        painter.drawPixmap(img_rect, self.display_pixmap)
        
        # 2. 绘制检测框
        painter.setRenderHint(QPainter.Antialiasing)
        for target in self.targets:
            self._draw_target(painter, target, img_rect)
        
        # 3. 绘制交互中框
        if self._mouse_pressed and self.current_tool == AnnotationTool.RECTANGLE:
            self._draw_draft_rect(painter, img_rect)
        
        painter.end()
    
    def _calc_fit_rect(self) -> QRect:
        """计算图片在控件中保持比例的显示区域"""
        if self.display_pixmap.isNull():
            return self.rect()
        pw, ph = self.display_pixmap.width(), self.display_pixmap.height()
        cw, ch = self.width(), self.height()
        scale = min(cw / pw, ch / ph) if pw > 0 and ph > 0 else 1
        w, h = int(pw * scale), int(ph * scale)
        x, y = (cw - w) // 2, (ch - h) // 2
        return QRect(x, y, w, h)
    
    def _to_norm(self, pos: QPoint, img_rect: QRect) -> Tuple[float, float]:
        """像素坐标转归一化坐标"""
        x = (pos.x() - img_rect.x()) / img_rect.width()
        y = (pos.y() - img_rect.y()) / img_rect.height()
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
    
    def _to_pixel(self, bbox: BoundingBox, img_rect: QRect) -> QRect:
        x1, y1, x2, y2 = bbox.to_pixels(img_rect.width(), img_rect.height())
        return QRect(img_rect.x() + x1, img_rect.y() + y1, x2 - x1, y2 - y1)
    
    def _draw_target(self, painter: QPainter, target: TargetObject, img_rect: QRect):
        rect = self._to_pixel(target.bbox, img_rect)
        color = QColor(*target.color)
        
        # 框
        pen = QPen(color, 2)
        if target == self.selected_target:
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)
        
        # 标签背景
        label = f"{target.class_name} {target.confidence:.2f}"
        font = QFont("Microsoft YaHei", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw, th = fm.horizontalAdvance(label) + 8, fm.height() + 4
        painter.fillRect(rect.x(), rect.y() - th, tw, th, color)
        
        # 标签文字
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.x() + 4, rect.y() - th + fm.ascent() + 2, label)
    
    def _draw_draft_rect(self, painter: QPainter, img_rect: QRect):
        pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
        painter.setPen(pen)
        x = min(self._drag_start.x(), self._drag_current.x())
        y = min(self._drag_start.y(), self._drag_current.y())
        w = abs(self._drag_current.x() - self._drag_start.x())
        h = abs(self._drag_current.y() - self._drag_start.y())
        painter.drawRect(x, y, w, h)
    
    # ========== 鼠标交互 ==========
    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self._calc_fit_rect()
        if not img_rect.contains(event.pos()):
            return
        
        self._mouse_pressed = True
        self._drag_start = event.pos()
        self._drag_current = event.pos()
        
        if self.current_tool == AnnotationTool.SELECT:
            # 点选检测
            nx, ny = self._to_norm(event.pos(), img_rect)
            clicked = None
            for t in reversed(self.targets):  # 后绘制的在上层
                x1, y1, x2, y2 = t.bbox.to_pixels(1, 1)
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    clicked = t
                    break
            self.selected_target = clicked
            self.target_selected.emit(clicked)
            self.update()
        
        elif self.current_tool == AnnotationTool.RECTANGLE:
            pass  # 拖拽中绘制草稿框
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._mouse_pressed:
            return
        self._drag_current = event.pos()
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._mouse_pressed:
            return
        self._mouse_pressed = False
        img_rect = self._calc_fit_rect()
        
        if self.current_tool == AnnotationTool.RECTANGLE:
            # 创建新标注
            n1 = self._to_norm(self._drag_start, img_rect)
            n2 = self._to_norm(event.pos(), img_rect)
            x, y = min(n1[0], n2[0]), min(n1[1], n2[1])
            w, h = abs(n2[0] - n1[0]), abs(n2[1] - n1[1])
            if w < 0.01 or h < 0.01:
                self.update()
                return
            
            from core.data_models import BoundingBox, TargetObject
            new_id = max([t.id for t in self.targets], default=0) + 1
            # 从 config 获取类别信息
            from config import PROJECT_CONFIG
            class_info = PROJECT_CONFIG.class_map.get(self.default_class_id, ("未知", (128,128,128)))
            
            new_target = TargetObject(
                id=new_id,
                class_id=self.default_class_id,
                class_name=class_info[0],
                bbox=BoundingBox(x, y, w, h),
                confidence=1.0,
                color=class_info[1],
                is_manual=True
            )
            self.targets.append(new_target)
            self.targets_updated.emit(self.targets)
            self.status_message.emit(f"新建标注: {class_info[0]} #{new_id}")
        
        self.update()
```

### 4.4 右侧面板（`ui/right_panel.py`）

**UI 内容**：
- **类别选择**：QComboBox（从 ProjectConfig 加载）
- **目标列表**：QListWidget，显示所有目标
- **属性编辑**：QGroupBox，含：
  - ID（只读 QLabel）
  - 类别（QComboBox）
  - 置信度（QDoubleSpinBox）
  - 备注（QLineEdit）
  - 删除按钮（QPushButton）

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QListWidget, QListWidgetItem, QComboBox, QLabel,
    QDoubleSpinBox, QLineEdit, QPushButton, QFormLayout
)
from PySide6.QtCore import Signal, Qt

class RightPanel(QWidget):
    target_modified = Signal(object)   # TargetObject
    target_deleted = Signal(int)       # target_id
    class_changed = Signal(int)        # class_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self._current_target: Optional[TargetObject] = None
        self._setup_ui()
    
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        
        # === 类别快速选择 ===
        self.class_group = QGroupBox("默认类别")
        class_layout = QVBoxLayout()
        self.combo_default_class = QComboBox()
        self._load_classes(self.combo_default_class)
        class_layout.addWidget(self.combo_default_class)
        self.class_group.setLayout(class_layout)
        self.layout.addWidget(self.class_group)
        
        # === 目标列表 ===
        self.list_group = QGroupBox("目标列表")
        list_layout = QVBoxLayout()
        self.target_list = QListWidget()
        list_layout.addWidget(self.target_list)
        self.list_group.setLayout(list_layout)
        self.layout.addWidget(self.list_group, stretch=1)
        
        # === 属性编辑 ===
        self.prop_group = QGroupBox("属性")
        prop_layout = QFormLayout()
        self.lbl_id = QLabel("-")
        self.combo_class = QComboBox()
        self._load_classes(self.combo_class)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0, 1)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        self.edit_note = QLineEdit()
        self.btn_delete = QPushButton("🗑 删除目标")
        self.btn_delete.setStyleSheet("color: red;")
        
        prop_layout.addRow("ID:", self.lbl_id)
        prop_layout.addRow("类别:", self.combo_class)
        prop_layout.addRow("置信度:", self.spin_conf)
        prop_layout.addRow("备注:", self.edit_note)
        prop_layout.addRow(self.btn_delete)
        self.prop_group.setLayout(prop_layout)
        self.layout.addWidget(self.prop_group)
        
        self._connect_signals()
        self._set_editor_enabled(False)
    
    def _load_classes(self, combo: QComboBox):
        from config import PROJECT_CONFIG
        combo.clear()
        for cid, (name, color) in PROJECT_CONFIG.class_map.items():
            combo.addItem(name, cid)
    
    def _connect_signals(self):
        self.combo_default_class.currentIndexChanged.connect(
            lambda: self.class_changed.emit(self.combo_default_class.currentData())
        )
        self.target_list.itemClicked.connect(self._on_list_clicked)
        self.combo_class.currentIndexChanged.connect(self._on_property_changed)
        self.spin_conf.valueChanged.connect(self._on_property_changed)
        self.edit_note.textChanged.connect(self._on_property_changed)
        self.btn_delete.clicked.connect(self._on_delete)
    
    def update_target_list(self, targets: List[TargetObject]):
        self.target_list.clear()
        for t in targets:
            item = QListWidgetItem(f"[{t.class_name}] #{t.id} ({t.confidence:.2f})")
            item.setData(Qt.UserRole, t.id)
            # 左侧色条
            color = QColor(*t.color)
            item.setForeground(color)
            self.target_list.addItem(item)
    
    def set_target(self, target: Optional[TargetObject]):
        self._current_target = target
        if target is None:
            self._set_editor_enabled(False)
            return
        
        self._set_editor_enabled(True)
        self.lbl_id.setText(str(target.id))
        self.combo_class.setCurrentIndex(
            self.combo_class.findData(target.class_id)
        )
        self.spin_conf.setValue(target.confidence)
        self.edit_note.setText(target.note)
        
        # 列表高亮
        for i in range(self.target_list.count()):
            item = self.target_list.item(i)
            if item.data(Qt.UserRole) == target.id:
                self.target_list.setCurrentItem(item)
                break
    
    def _set_editor_enabled(self, enabled: bool):
        self.combo_class.setEnabled(enabled)
        self.spin_conf.setEnabled(enabled)
        self.edit_note.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)
        if not enabled:
            self.lbl_id.setText("-")
    
    def _on_list_clicked(self, item: QListWidgetItem):
        target_id = item.data(Qt.UserRole)
        # 发射信号让画布选中，画布再回调 set_target
        # 这里简化：直接让画布处理，或者通过主窗口中转
    
    def _on_property_changed(self):
        if self._current_target is None:
            return
        # 构造修改后的对象
        from core.data_models import TargetObject, BoundingBox
        updated = TargetObject(
            id=self._current_target.id,
            class_id=self.combo_class.currentData(),
            class_name=self.combo_class.currentText(),
            bbox=self._current_target.bbox,
            confidence=self.spin_conf.value(),
            color=self._current_target.color,  # TODO: 根据新类别更新颜色
            is_manual=self._current_target.is_manual,
            note=self.edit_note.text()
        )
        self.target_modified.emit(updated)
    
    def _on_delete(self):
        if self._current_target:
            self.target_deleted.emit(self._current_target.id)
            self._set_editor_enabled(False)
```

### 4.5 底部面板（`ui/bottom_panel.py`）

**UI 内容**：
- **日志输出**：QTextEdit（只读，最大行数 500）
- **状态信息**：QLabel（FPS、分辨率、帧号）

```python
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import Qt

class BottomPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(180)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        
        # 日志
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("系统日志...")
        self.log_edit.setMaximumBlockCount(500)  # 自动截断
        
        # 状态
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_resolution = QLabel("分辨率: -")
        self.lbl_frame = QLabel("帧号: -")
        
        right = QVBoxLayout()
        right.addWidget(self.lbl_fps)
        right.addWidget(self.lbl_resolution)
        right.addWidget(self.lbl_frame)
        right.addStretch()
        
        layout.addWidget(self.log_edit, stretch=3)
        layout.addLayout(right, stretch=1)
    
    def append_log(self, message: str, level: str = "INFO"):
        color = {"INFO": "black", "WARN": "orange", "ERROR": "red"}.get(level, "black")
        self.log_edit.append(f'<span style="color:{color}">[{level}] {message}</span>')
    
    def update_fps(self, fps: float):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
    
    def set_resolution(self, w: int, h: int):
        self.lbl_resolution.setText(f"分辨率: {w}x{h}")
```

### 4.6 工具栏（`ui/toolbar.py`）

```python
from PySide6.QtWidgets import QToolBar, QButtonGroup
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction

class Toolbar(QToolBar):
    tool_changed = Signal(object)      # AnnotationTool
    run_detection = Signal()
    save_project = Signal()
    
    def __init__(self, parent=None):
        super().__init__("工具", parent)
        self._setup_ui()
    
    def _setup_ui(self):
        # 工具按钮组
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        tools = [
            (AnnotationTool.SELECT, "选择", "cursor"),
            (AnnotationTool.RECTANGLE, "矩形", "rect"),
            (AnnotationTool.ARROW, "箭头", "arrow"),
            (AnnotationTool.DELETE, "删除", "delete"),
        ]
        
        for tool, name, icon in tools:
            btn = QAction(name, self)
            btn.setCheckable(True)
            btn.triggered.connect(lambda checked, t=tool: self.tool_changed.emit(t))
            self.addAction(btn)
            if tool == AnnotationTool.SELECT:
                btn.setChecked(True)
        
        self.addSeparator()
        
        # 功能按钮
        self.act_detect = QAction("▶ 开始检测", self)
        self.act_detect.triggered.connect(self.run_detection)
        self.addAction(self.act_detect)
        
        self.act_save = QAction("💾 保存", self)
        self.act_save.triggered.connect(self.save_project)
        self.addAction(self.act_save)
```

---

## 五、配置与入口

### `config.py`

```python
from core.data_models import ProjectConfig

PROJECT_CONFIG = ProjectConfig(
    name="舰船识别项目",
    class_map={
        0: ("船只", (0, 0, 255)),      # BGR 红
        1: ("飞机", (0, 255, 0)),      # BGR 绿
        2: ("浮标", (0, 255, 255)),    # BGR 黄
        3: ("其他", (128, 128, 128)),  # 灰
    },
    confidence_threshold=0.45,
    auto_save=True,
)
```

### `main.py`

```python
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow

def main():
    # 启用高分屏支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 跨平台一致
    
    # 加载样式表（可选）
    try:
        with open("resources/styles/main.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

---

## 六、信号槽总览图

```
Toolbar          LeftPanel          CentralCanvas        RightPanel        BottomPanel
   │                 │                    │                  │                │
   ├─tool_changed───>│                    │                  │                │
   ├─run_detection───>│                    │                  │                │
   │                 ├─image_selected────>│                  │                │
   │                 ├─camera_connected───>│                  │                │
   │                 │                    ├─target_selected──>│                │
   │                 │                    ├─targets_updated───>│                │
   │                 │                    ├─status_message────────────────────>│
   │                 │                    ├─fps_updated───────────────────────>│
   │                 │                    │<─target_modified──┤                │
   │                 │                    │<─target_deleted──┤                │
   │                 │                    │<─class_changed───┤                │
```

---

## 七、AI 生成代码的约束清单

为确保 AI 输出稳定，生成时必须检查：

- [ ] 所有自定义信号使用 `Signal(...)` 显式声明类型
- [ ] 所有数据模型使用 `@dataclass`，禁止用 `dict` 传递
- [ ] 画布坐标必须经过 `_calc_fit_rect()` 转换，禁止直接拿 `self.width()`
- [ ] 颜色统一使用 BGR 元组 `(b, g, r)`，与 OpenCV 保持一致
- [ ] 所有 UI 文本使用中文，代码注释使用中文
- [ ] 每个 QWidget 子类必须有 `parent=None` 参数
- [ ] 信号槽连接集中在 `_connect_signals()` 方法中
- [ ] 资源路径使用相对路径，通过 `config.py` 统一管理

---

## 八、扩展路线图

| 阶段 | 功能 | 涉及文件 |
|------|------|----------|
| MVP | 图片加载、矩形标注、类别选择、保存 JSON | 全部基础文件 |
| v0.2 | 视频播放、时间轴（QSlider）、帧跳转 | `central_canvas.py`, `bottom_panel.py` |
| v0.3 | 检测器接入（YOLO）、自动标注 | `core/yolo_detector.py` |
| v0.4 | 多边形标注、方向箭头交互 | `central_canvas.py` |
| v0.5 | 项目文件（.shipproj）、导出 COCO/YOLO | `core/project_manager.py` |
