"""完整功能演示：加载图片 + 模拟检测 + 截图"""
import sys, os, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont, QBrush

from ui.main_window import MainWindow
from core.data_models import BoundingBox, TargetObject, AnnotationTool


def create_realistic_ir_image(path: str, width: int = 1024, height: int = 512):
    """创建逼真的红外海面船只图像"""
    img = QImage(width, height, QImage.Format_RGB32)
    
    # 深色海面背景（模拟红外冷背景）
    for y in range(height):
        for x in range(width):
            # 基础海面亮度（底部更亮模拟远海）
            base = int(15 + 20 * (y / height))
            # 水平波纹
            wave = int(4 * np.sin(x * 0.015 + y * 0.05))
            # 随机噪声
            noise = np.random.randint(-3, 4)
            val = max(0, min(255, base + wave + noise))
            img.setPixelColor(x, y, QColor(val, val + 5, val + 10))
    
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制5个不同大小的船只（模拟不同距离）
    ships = [
        # (x, y, w, h, 类别, 亮度)
        (120, 280, 120, 35, "Ada", 200),
        (380, 200, 80, 25, "Akizuki", 180),
        (550, 320, 60, 18, "Alvaro De Bazan", 160),
        (720, 250, 100, 30, "Independence", 190),
        (880, 290, 45, 15, "Jiangkai II", 150),
    ]
    
    for sx, sy, sw, sh, name, brightness in ships:
        # 船体（亮色，模拟热目标）
        painter.fillRect(sx, sy, sw, sh, QColor(brightness, brightness - 10, brightness - 20))
        
        # 上层建筑
        bldg_w = sw // 3
        bldg_h = sh // 2
        painter.fillRect(sx + sw // 4, sy - bldg_h, bldg_w, bldg_h, 
                        QColor(brightness - 20, brightness - 30, brightness - 40))
        
        # 烟囱（更亮的点）
        painter.fillRect(sx + sw // 3, sy - bldg_h - 5, 6, 8, 
                        QColor(min(255, brightness + 30), brightness, brightness - 10))
        
        # 标签
        font = QFont("Sans Serif", 8)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 100))
        painter.drawText(sx, sy - bldg_h - 12, name)
    
    # 添加海面杂波点（模拟红外噪声）
    for _ in range(500):
        nx = np.random.randint(0, width)
        ny = np.random.randint(int(height * 0.6), height)
        brightness = np.random.randint(30, 60)
        painter.setPen(QPen(QColor(brightness, brightness + 5, brightness + 10), 1))
        painter.drawPoint(nx, ny)
    
    # 添加远处的微小目标（测试小目标检测）
    for _ in range(8):
        mx = np.random.randint(50, width - 50)
        my = np.random.randint(50, int(height * 0.5))
        painter.fillRect(mx, my, 4, 3, QColor(120, 115, 110))
    
    painter.end()
    img.save(path)
    print(f"红外图像已创建: {path}")


def create_detection_results():
    """创建模拟检测结果（9类船舶）"""
    targets = [
        TargetObject(
            id=1, class_id=0, class_name="Ada",
            bbox=BoundingBox(0.117, 0.52, 0.117, 0.068),
            confidence=0.94, color=(255, 0, 0), is_manual=False
        ),
        TargetObject(
            id=2, class_id=1, class_name="Akizuki",
            bbox=BoundingBox(0.371, 0.37, 0.078, 0.049),
            confidence=0.89, color=(0, 200, 0), is_manual=False
        ),
        TargetObject(
            id=3, class_id=2, class_name="Alvaro De Bazan",
            bbox=BoundingBox(0.537, 0.605, 0.059, 0.035),
            confidence=0.76, color=(255, 255, 0), is_manual=False
        ),
        TargetObject(
            id=4, class_id=3, class_name="Armourique",
            bbox=BoundingBox(0.703, 0.469, 0.098, 0.059),
            confidence=0.82, color=(0, 150, 255), is_manual=False
        ),
        TargetObject(
            id=5, class_id=4, class_name="Independence",
            bbox=BoundingBox(0.859, 0.547, 0.044, 0.029),
            confidence=0.68, color=(255, 0, 255), is_manual=False
        ),
    ]
    return targets


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
    window.setWindowTitle("舰船识别标注系统 v0.1 — 红外船舶检测演示")
    window.resize(1400, 900)
    window.show()
    
    # 创建测试图像
    test_dir = Path(__file__).resolve().parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    test_img = str(test_dir / "demo_ship_infrared.png")
    create_realistic_ir_image(test_img)
    
    # 加载图片到画布
    window.canvas.load_image(test_img)
    window.left_panel.set_source_status(True, f"已加载 ({Path(test_img).name})")
    window.left_panel.set_detector_status(True)
    window.bottom_panel.set_resolution(1024, 512)
    window.bottom_panel.append_log(f"已加载图片: {Path(test_img).name}", "INFO")
    
    # 添加模拟检测结果
    targets = create_detection_results()
    window.canvas.targets = targets
    window.canvas.update()
    window.right_panel.update_target_list(targets)
    window.right_panel.set_target(targets[0])  # 选中第一个目标
    
    # 更新日志
    window.bottom_panel.append_log("检测完成，找到 5 个目标", "INFO")
    for t in targets:
        window.bottom_panel.append_log(
            f"[INFO] {t.class_name}: conf={t.confidence:.2f}, "
            f"pos=[{t.bbox.x:.3f},{t.bbox.y:.3f},{t.bbox.w:.3f},{t.bbox.h:.3f}]",
            "INFO"
        )
    window.status_bar.showMessage("检测完成: 5 个目标")
    
    # 截图
    def take_screenshot():
        screenshot_path = Path(__file__).resolve().parent.parent / "resources" / "screenshot.png"
        screen = app.primaryScreen()
        if screen:
            pixmap = screen.grabWindow(window.winId())
            pixmap.save(str(screenshot_path))
            print(f"截图已保存: {screenshot_path}")
            print(f"文件大小: {screenshot_path.stat().st_size} bytes")
        app.quit()
    
    QTimer.singleShot(2500, take_screenshot)
    app.exec()


if __name__ == "__main__":
    main()
