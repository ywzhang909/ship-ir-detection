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
├── app.py                      # Streamlit 批量推理工具界面层
├── detector.py                 # 无框架依赖的检测核心（注册表/加载/预处理/推理/绘图/指标）
├── tests/                      # Streamlit 工具 pytest + AppTest 测试
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

### Streamlit 批量推理工具

基于 YOLO11 的红外图像船舶检测可视化工具：上传一张或多张图片，选择模型（默认自动加载效果最好的模型），在 GPU 上运行检测，逐图查看标注结果与统计指标。

```powershell
# 启动应用
uv run streamlit run app.py
```

浏览器访问 <http://localhost:8501>。

**功能特性：**

- **多图上传** — 支持 `.jpg / .jpeg / .png / .bmp`，一次可上传多张批量推理
- **模型选择** — 自动扫描 `runs/detect/ship-detection/*/weights/best.pt`，按测试集 mAP50-95 降序排列，**默认选中效果最好的模型**
- **按模型自动预处理** — 从每个训练运行的 `args.yaml` 读取数据配置，自动路由到对应的输入预处理流水线
- **GPU 推理** — 自动检测 CUDA，侧边栏显示 GPU 型号；不可用时降级 CPU 并给出警告
- **结果展示** — 每张图的标注图（可下载 PNG）、逐图指标表、汇总指标与类别分布柱状图

**使用指南：**

1. **选择模型** — 左侧边栏下拉框列出所有已训练模型及其测试集指标。默认即为效果最好的 `T5_yolo11l_fusion`。
2. **调整参数**（可选）— 置信度阈值滑块（默认 0.25）、输入尺寸（默认 640）。
3. **上传图片** — 拖拽或点击上传区域，支持多选。
4. **点击 "Run detection"** — 页面显示汇总指标、类别分布柱状图与逐图标注结果。
5. 单张图片解码失败不会影响批内其他图片，错误会在对应卡片中单独提示。

**模型与预处理：**

不同实验使用了不同的训练输入分布。工具从每个运行目录下的 `args.yaml` 的 `data:` 字段自动判断，并在推理前施加相同预处理，保证输入分布与训练一致：

| 预处理类型 `[tag]` | 匹配关键字 | 流水线 |
|---|---|---|
| `dual` | `dual_fusion` | 双流融合：AWB → tophat+CLAHE ⊕ AWB → Butterworth BPF+CLAHE |
| `wavelet` | `wavelet_dual_fusion` | 双流融合（频域流替换为小波 DWT） |
| `rpca` | `rpca_dual_fusion` | 双流融合（频域流替换为 RPCA 前景提取） |
| `tophat` | `tophat_clahe` | 单流：AWB → top-hat(k=31) → CLAHE → unsharp |
| `butterworth` | `butterworth_clahe` | 单流：AWB → Butterworth BPF(8/80) → CLAHE → unsharp |
| `raw` | 其他 | 原图直接送入网络 |

所有输入图像统一按参考脚本约定做灰度归一化（BGR→灰度→BGR），与单波段红外训练域保持一致。

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
# Streamlit 工具测试 (23 个用例：单元测试 + AppTest 集成测试 + GPU 冒烟测试)
uv run pytest tests/ -v

# 数据模型测试
uv run pytest ship_annotation/tests/test_models.py -v

# 标签格式测试
cd yolo-training
uv run pytest tests/test_labels.py -v
```

## 技术栈

- **深度学习**: PyTorch 2.11 + CUDA 12.8, Ultralytics YOLO11, SAHI
- **GUI 框架**: PySide6 (Qt6), Streamlit
- **图像处理**: OpenCV, Pillow
- **包管理**: uv (workspace)

## 已知说明

- 所有模型均在 NSLSR / 合成渲染域上训练；跨域图像（如 `datasets/customn` 实拍数据）可能检出偏少甚至为零，属于正常的域差现象，并非工具故障。
- 若未找到任何权重文件，Streamlit 应用会提示预期的目录结构并停止。

## License

Internal use only.
