# 舰船识别系统 — Ship IR Detection

红外波段海面舰艇目标检测与识别系统，基于 YOLO11 + PySide6 可视化界面。

## 系统截图

![舰船识别标注系统](ship_annotation/resources/screenshot.png)

## 环境要求

- Python 3.13+
- CUDA 12.8 (RTX 4090/5080)
- uv 包管理器

## 快速开始

```bash
# 安装依赖
uv sync

# 启动可视化界面
cd ship_annotation
uv run python main.py
```

## 项目结构

```
ship/
├── pyproject.toml              # 工作区配置 (uv workspace)
├── yolo-training/              # YOLO 训练与推理
│   ├── train.py                # 训练入口
│   ├── predict.py              # 推理脚本 (标准 + SAHI)
│   └── configs/                # 数据集与训练配置
├── ship_annotation/            # PySide6 可视化界面
│   ├── main.py                 # 入口
│   ├── config.py               # 全局配置
│   ├── app.py                  # 应用单例
│   ├── core/                   # 核心模块
│   │   ├── data_models.py      # 数据模型
│   │   ├── detector_base.py    # 检测器基类
│   │   ├── yolo_detector.py    # YOLO 检测器
│   │   └── frame_source.py     # 图像源管理
│   ├── ui/                     # 界面模块
│   │   ├── main_window.py      # 主窗口
│   │   ├── toolbar.py          # 工具栏
│   │   ├── left_panel.py       # 左侧面板
│   │   ├── central_canvas.py   # 中央画布
│   │   ├── right_panel.py      # 右侧面板
│   │   └── bottom_panel.py     # 底部日志
│   ├── resources/              # 资源文件
│   │   └── styles/main.qss     # 全局样式
│   └── tests/                  # 单元测试
└── docs/                       # 研究论文
```

## 功能特性

### 可视化界面 (PySide6)

- **图片加载**: 支持文件夹浏览，批量加载图片
- **YOLO 检测**: 集成 YOLO11 模型，支持标准/SAHI 推理
- **目标标注**: 矩形框选择、拖拽、编辑
- **属性编辑**: 类别切换、置信度调整、备注添加
- **工具栏**: 选择/矩形/箭头/删除工具
- **日志面板**: 实时显示检测结果与系统状态

### YOLO 推理

- **标准推理**: `uv run python predict.py --mode standard --source image.jpg`
- **SAHI 推理**: `uv run python predict.py --mode sahi --source ./images/`
- **预处理**: 支持 12 种预处理方法 (tophat_clahe, retinex 等)

## 训练模型

```bash
# 快速测试 (1 epoch)
cd yolo-training
uv run python train.py --model yolo11n.pt --epochs 1 --batch 4

# 完整训练
uv run python train.py
```

## 船舶类别 (9 类)

| ID | 类别 | 描述 |
|----|------|------|
| 0 | Ada | 护卫舰 |
| 1 | Akizuki | 驱逐舰 |
| 2 | Alvaro De Bazan | 护卫舰 |
| 3 | Armourique | 护卫舰 |
| 4 | Independence | 濒海战斗舰 |
| 5 | Jiangkai II | 护卫舰 (中国) |
| 6 | Oliver Hazard Perry | 护卫舰 |
| 7 | Sejong Daewang | 驱逐舰 (韩国) |
| 8 | Zumwalt | 驱逐舰 |

## 测试

```bash
# 数据模型测试
uv run pytest ship_annotation/tests/test_models.py -v

# 标签格式测试
cd yolo-training
uv run pytest tests/test_labels.py -v
```

## 技术栈

- **深度学习**: PyTorch 2.11 + CUDA 12.8, Ultralytics YOLO11, SAHI
- **GUI 框架**: PySide6 (Qt6)
- **图像处理**: OpenCV, Pillow
- **包管理**: uv (workspace)

## License

Internal use only.
