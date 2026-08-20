from PySide6.QtWidgets import QToolBar, QButtonGroup, QToolButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction
from core.data_models import AnnotationTool


class Toolbar(QToolBar):
    """工具栏：工具切换、检测触发、项目保存"""

    tool_changed = Signal(object)
    run_detection = Signal()
    save_project = Signal()

    def __init__(self, parent=None):
        super().__init__("工具", parent)
        self._setup_ui()

    def _setup_ui(self):
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        tools = [
            (AnnotationTool.SELECT, "选择"),
            (AnnotationTool.RECTANGLE, "矩形"),
            (AnnotationTool.ARROW, "箭头"),
            (AnnotationTool.DELETE, "删除"),
        ]

        for i, (tool, name) in enumerate(tools):
            btn = QToolButton(self)
            btn.setText(name)
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            btn.clicked.connect(lambda checked, t=tool: self.tool_changed.emit(t))
            self.addWidget(btn)
            self.btn_group.addButton(btn, i)
            if tool == AnnotationTool.SELECT:
                btn.setChecked(True)

        self.addSeparator()

        self.act_detect = QAction("▶ 开始检测", self)
        self.act_detect.triggered.connect(self.run_detection)
        self.addAction(self.act_detect)

        self.act_save = QAction("💾 保存", self)
        self.act_save.triggered.connect(self.save_project)
        self.addAction(self.act_save)

