"""自动截图测试 - 启动应用、加载测试图片、截图"""
import sys
import os
import numpy as np
from pathlib import Path

# 确保可以导入 ship_annotation 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont

from ui.main_window import MainWindow
from core.data_models import BoundingBox, TargetObject


def create_test_image(path: str, width: int = 1024, height: int = 512):
    """创建一张合成的红外风格测试图片，包含模拟船只目标"""
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor(20, 30, 50))  # 深蓝背景（模拟海面）
    
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制波浪纹理
    pen = QPen(QColor(30, 45, 65), 1)
    painter.setPen(pen)
    for y in range(0, height, 15):
        for x in range(0, width, 3):
            offset = int(3 * np.sin(x * 0.02 + y * 0.1))
            painter.drawPoint(x, y + offset)
    
    # 绘制模拟船只（亮色矩形 + 细节）
    ships = [
        (200, 200, 80, 25, " Ada"),
        (500, 180, 100, 30, "Akizuki"),
        (750, 250, 60, 20, "Frigate"),
    ]
    
    for sx, sy, sw, sh, label in ships:
        # 船体
        painter.fillRect(sx, sy, sw, sh, QColor(180, 190, 200))
        # 上层建筑
        painter.fillRect(sx + sw//4, sy - sh//2, sw//2, sh//2, QColor(160, 170, 180))
        # 标签
        font = QFont("Sans Serif", 8)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(sx, sy - sh - 5, label)
    
    # 添加一些噪声点模拟红外杂波
    for _ in range(200):
        nx = np.random.randint(0, width)
        ny = np.random.randint(0, height)
        brightness = np.random.randint(40, 80)
        img.setPixelColor(nx, ny, QColor(brightness, brightness + 10, brightness + 20))
    
    painter.end()
    img.save(path)
    print(f"测试图片已创建: {path}")
    return path


def add_mock_targets(window):
    """手动添加模拟检测目标到画布"""
    targets = [
        TargetObject(
            id=1, class_id=0, class_name="Ada",
            bbox=BoundingBox(0.17, 0.35, 0.08, 0.06),
            confidence=0.92, color=(255, 0, 0), is_manual=False
        ),
        TargetObject(
            id=2, class_id=1, class_name="Akizuki",
            bbox=BoundingBox(0.47, 0.30, 0.10, 0.07),
            confidence=0.87, color=(0, 200, 0), is_manual=False
        ),
        TargetObject(
            id=3, class_id=2, class_name="Alvaro De Bazan",
            bbox=BoundingBox(0.72, 0.42, 0.06, 0.05),
            confidence=0.73, color=(255, 255, 0), is_manual=False
        ),
    ]
    window.canvas.targets = targets
    window.canvas.update()
    window.right_panel.update_target_list(targets)
    window.bottom_panel.append_log("已加载模拟检测结果: 3 个目标", "INFO")
    window.bottom_panel.append_log("[INFO] Ada: conf=0.92, pos=[174,179,256,210]", "INFO")
    window.bottom_panel.append_log("[INFO] Akizuki: conf=0.87, pos=[481,154,583,190]", "INFO")
    window.bottom_panel.append_log("[INFO] Alvaro De Bazan: conf=0.73, pos=[737,215,799,241]", "INFO")
    window.status_bar.showMessage("检测完成: 3 个目标")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 加载样式表
    qss_path = Path(__file__).resolve().parent.parent / "resources" / "styles" / "main.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    
    # 创建主窗口
    window = MainWindow()
    window.setWindowTitle("舰船识别标注系统 v0.1 - 演示")
    window.resize(1400, 900)
    window.show()
    
    # 创建并加载测试图片
    test_dir = Path(__file__).resolve().parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    test_img = str(test_dir / "test_ship_infrared.png")
    create_test_image(test_img)
    
    # 加载图片到画布
    window.canvas.load_image(test_img)
    window.left_panel.set_source_status(True, f"已连接 ({Path(test_img).name})")
    window.left_panel.set_detector_status(True)
    window.bottom_panel.set_resolution(1024, 512)
    window.bottom_panel.append_log(f"已加载图片: {test_img}", "INFO")
    
    # 添加模拟检测结果
    add_mock_targets(window)
    
    # 截图
    def take_screenshot():
        screenshot_path = Path(__file__).resolve().parent.parent / "resources" / "screenshot.png"
        screen = app.primaryScreen()
        if screen:
            pixmap = screen.grabWindow(window.winId())
            pixmap.save(str(screenshot_path))
            print(f"截图已保存: {screenshot_path}")
        else:
            # 后备方案：直接抓取窗口
            pixmap = window.grab()
            pixmap.save(str(screenshot_path))
            print(f"截图已保存 (窗口抓取): {screenshot_path}")
        app.quit()
    
    QTimer.singleShot(2000, take_screenshot)
    app.exec()


if __name__ == "__main__":
    main()
