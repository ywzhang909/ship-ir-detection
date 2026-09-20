# 图生成基建（generate_figures/）

论文图表统一生成基建。所有 figure agent 必须复用本目录模块，保证图风一致、中文无豆腐块、与正文引用严格对齐。

## 目录结构

```
generate_figures/
├── common_style.py   # 统一样式模块（CJK 字体 / 6 色调色板 / save_fig）
├── manifest.json     # Fig1–Fig10 图清单（锁定契约）
└── README.md         # 本说明
```

## 使用方式

### 1. 统一样式（common_style.py）

```python
import common_style
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, color=common_style.PALETTE[0], label="Ours")
ax.set_title("舰艇检测 mAP 对比")          # 中文直接写
common_style.save_fig(fig, "figures/fig8_ablation.png")   # DPI=200
```

- **CJK 字体**：模块导入即自动注册 Noto Sans CJK SC（`font_manager.addfont` + `rcParams`），并修复中文负号方块问题（`axes.unicode_minus=False`）。
- **调色板**：`common_style.PALETTE` 统一 6 色（`#4C72B0` 蓝 / `#DD8452` 橙 / `#55A868` 绿 / `#C44E52` 红 / `#8172B3` 紫 / `#937860` 棕）。
- **save_fig(fig, path, dpi=200, tight=True)**：自动建目录、`tight_layout`、`bbox_inches="tight"`；**若渲染出现 CJK glyph missing 警告则 raise**，禁止静默产出豆腐块图。

### 2. 图清单（manifest.json）

`manifest.json` 锁定 Fig1–Fig10 的 `filename / title / description / status`。规则：

- 文件名必须与正文 `![图N](figures/xxx.png)` 引用一致；
- `status` 取值：`planned`（规划）→ `generated`（已生成）→ `final`（定稿）；
- 生成图片后请把对应条目 `status` 改为 `generated`/`final`。

## TDD 校验 harness

```bash
# 从仓库根目录运行
uv run pytest docs/论文/tests/test_paper.py -v
```

9 组断言中与本基建相关：

| 组 | 断言 | 状态 |
|----|------|------|
| 8 | CJK 字体 smoke：`common_style.save_fig` 中文渲染无 glyph 警告 | 基建完成后即绿 |
| 9 | `manifest.json` 有效：锁定 Fig1–Fig10、字段齐全、filename 唯一 | 基建完成后即绿 |
| 4 | 图完整性：正文引用的 PNG 存在 / 可打开 / >10KB，`figures/` 无孤儿图 | 依赖正文 + 图片本体 |

## 工作流

1. 写正文（`docs/论文/论文.md`）→ 组 1–7 转绿；
2. 按 manifest 生成 Fig1–Fig10 到 `docs/论文/figures/` → 组 4 转绿；
3. 全程保持组 8–9 绿（基建不可破坏）。

## 环境

- Python venv：`/home/ws/code/ship/.venv/bin/python`（cv2 / numpy / matplotlib / pywavelets 已装）
- 运行：`~/.local/bin/uv run pytest ...`
- 字体：Noto Sans CJK SC（`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`）