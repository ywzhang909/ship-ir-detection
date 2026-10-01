# -*- coding: utf-8 -*-
"""论文图表统一样式模块（CJK 字体 / 调色板 / save_fig）。

所有 figure agent 必须通过本模块生成图片，保证：
    1. 中文渲染无豆腐块（Noto Sans CJK SC，font_manager.addfont + rcParams）；
    2. 图风一致（统一 6 色调色板、DPI=200）；
    3. 中文缺字即失败（save_fig 捕获 glyph missing 警告并 raise）。

用法:
    import common_style
    fig, ax = plt.subplots()
    ax.plot(x, y, color=common_style.PALETTE[0])
    common_style.save_fig(fig, "figures/fig1_pipeline.png")

依赖: matplotlib（venv 已装）。无第三方新增依赖。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 无显示环境时强制 Agg 后端
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# 调色板（统一 6 色，论文级低饱和配色）
# ---------------------------------------------------------------------------
PALETTE = [
    "#4C72B0",  # 蓝
    "#DD8452",  # 橙
    "#55A868",  # 绿
    "#C44E52",  # 红
    "#8172B3",  # 紫
    "#937860",  # 棕
]

# ---------------------------------------------------------------------------
# CJK 字体配置
# ---------------------------------------------------------------------------
# Noto Sans CJK SC 常见安装路径（Linux / macOS / Windows）
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
]

CJK_FONT_NAME = "Noto Sans CJK SC"


def _find_cjk_font_path() -> str:
    """优先复用 matplotlib 已注册的 Noto Sans CJK SC，否则搜索已知路径。"""
    for f in fm.fontManager.ttflist:
        if f.name == CJK_FONT_NAME:
            return f.fname
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        f"未找到 CJK 字体 {CJK_FONT_NAME}。请安装 Noto Sans CJK SC，"
        f"或把字体路径加入 common_style._FONT_CANDIDATES。"
    )


def setup_cjk_font() -> str:
    """注册 CJK 字体并写入 rcParams；返回字体名。"""
    font_path = _find_cjk_font_path()
    fm.fontManager.addfont(font_path)  # .ttc 亦可注册（取首个 face）
    plt.rcParams["font.sans-serif"] = [CJK_FONT_NAME] + plt.rcParams.get(
        "font.sans-serif", []
    )
    plt.rcParams["font.family"] = "sans-serif"
    # 修复中文负号显示为方块的问题
    plt.rcParams["axes.unicode_minus"] = False
    return CJK_FONT_NAME


# 模块导入即完成字体配置（幂等）
setup_cjk_font()

# ---------------------------------------------------------------------------
# save_fig：统一保存 + 中文缺字检测
# ---------------------------------------------------------------------------
_GLYPH_WARNING_PATTERNS = ("missing from font", "Glyph")


def save_fig(fig, path, dpi: int = 200, tight: bool = True) -> Path:
    """保存图片；若渲染时出现 CJK glyph missing 警告则 raise。

    参数:
        fig:   matplotlib Figure
        path: 输出路径（父目录自动创建）
        dpi:  分辨率，默认 200
        tight:保存前执行 tight_layout + bbox_inches='tight'

    返回:
        实际写入的 Path

    异常:
        RuntimeError: 渲染出现 glyph missing 警告（中文豆腐块），
                      说明字体配置失效，禁止静默产出坏图。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if tight:
            try:
                fig.tight_layout()
            except Exception:  # noqa: BLE001 — tight_layout 失败不阻塞保存
                pass
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    glyph_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and any(p in str(w.message) for p in _GLYPH_WARNING_PATTERNS)
    ]
    if glyph_warnings:
        raise RuntimeError(
            f"CJK glyph missing 警告（中文豆腐块）: {path}\n"
            f"  {glyph_warnings[0].message}\n"
            f"请检查 common_style.setup_cjk_font() 字体配置。"
        )
    return path