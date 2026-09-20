# -*- coding: utf-8 -*-
"""论文 TDD 校验 harness（主契约）。

运行方式（从仓库根目录）:
    uv run pytest docs/论文/tests/test_paper.py -v

9 组断言:
    1. 结构       —— 必需标题层级存在且有序
    2. 方法覆盖   —— 每个方法/策略关键词必须出现
    3. 四要素块   —— 每个 `### 5.x` 方法块恰好含 4 个 `####` 子块
    4. 图完整性   —— 引用的 PNG 存在/可打开/>10KB，figures/ 无孤儿图
    5. 数字抽查   —— ~25 个地标数字精确出现（禁止四舍五入）
    6. 无占位符   —— 不含 TODO/TBD/待补/XXX/Lorem/×××
    7. 关键词/摘要/图嵌入 —— 关键词 3–5 个、摘要 150–400 字、图嵌入 ≥ 8
    8. CJK 字体 smoke —— common_style 中文渲染无 glyph 警告
    9. manifest 有效 —— manifest.json 锁定 Fig1–Fig10

TDD 状态:
    - 组 1–7 依赖论文正文（默认 docs/论文/论文.md，可用 PAPER_PATH 覆盖）。
      正文未写时有意失败（RED）；正文写完后必须全绿。
    - 组 8–9 依赖图表基建（common_style.py / manifest.json），基建完成后即绿。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
PAPER_DIR = Path(__file__).resolve().parents[1]          # docs/论文
REPO_ROOT = Path(__file__).resolve().parents[3]          # ship/
GENERATE_DIR = PAPER_DIR / "generate_figures"
FIGURES_DIR = PAPER_DIR / "figures"
MANIFEST_PATH = GENERATE_DIR / "manifest.json"

# 允许 figure agent 用环境变量覆盖正文路径
PAPER_PATH = Path(os.environ.get("PAPER_PATH", PAPER_DIR / "论文.md"))

# 让 tests 能 import generate_figures.common_style
if str(GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATE_DIR))

# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------
# 组 1：必需标题层级（有序）
REQUIRED_STRUCTURE = [
    ("摘要", "摘要"),
    ("关键词", "关键词"),
    ("## 1 引言", "1 引言"),
    ("## 2 相关工作", "2 相关工作"),
    ("## 3 方法", "3 方法"),
    ("## 4 实验设置", "4 实验设置"),
    ("## 5 实验结果与作用分析", "5 实验结果与作用分析"),
    ("## 6 讨论", "6 讨论"),
    ("## 7 结论", "7 结论"),
    ("参考文献", "参考文献"),
]

# 组 2：方法覆盖（每组命中任一别名即可）
METHOD_KEYWORDS = [
    ("CLAHE", ["CLAHE"]),
    ("Retinex 多尺度 (MSR)", ["MSR", "Retinex"]),
    ("CRRP/CP-PB 数据增强", ["CRRP", "CP-PB", "Copy-Paste", "复制粘贴"]),
    ("同态滤波 (Homomorphic)", ["同态滤波", "Homomorphic"]),
    ("Top-hat / Bottom-hat", ["Top-hat", "Top hat", "顶帽", "Bottom-hat", "底帽"]),
    ("Butterworth", ["Butterworth", "巴特沃斯"]),
    ("DWT 小波", ["DWT", "小波"]),
    ("DoG", ["DoG", "高斯差分"]),
    ("各向异性扩散", ["各向异性扩散", "anisotropic", "Anisotropic"]),
    ("中值/双边滤波", ["中值", "双边", "median", "Median", "bilateral", "Bilateral"]),
    ("AWB 白平衡", ["AWB", "白平衡"]),
    ("空频融合 (SpatialFrequencyFusion)", ["SpatialFrequencyFusion", "空频融合", "空间频率融合"]),
    ("损失函数 (Shape-IoU/WIoU/NWD)", ["Shape-IoU", "WIoU", "NWD"]),
    ("训练策略 (AdamW)", ["AdamW"]),
    ("训练策略 (Cosine)", ["Cosine", "余弦"]),
    ("训练策略 (Mosaic)", ["Mosaic", "马赛克"]),
    ("训练策略 (早停)", ["早停", "early stop", "EarlyStopping", "early-stopping"]),
    ("训练策略 (AMP)", ["AMP", "混合精度"]),
    ("训练策略 (augment-jitter)", ["augment-jitter"]),
    ("训练策略 (preprocess-jitter)", ["preprocess-jitter"]),
    ("训练策略 (fusion warmup + lr-scale)", ["warmup", "warm-up", "lr-scale", "学习率缩放"]),
]

# 组 3：每个 `### 5.x` 方法块必须恰好含这 4 个 `####` 子块
REQUIRED_SUBBLOCKS = {"实现与实验", "结果", "结论", "作用"}

# 组 5：地标数字（来自 data_tables.md，禁止四舍五入）
LANDMARK_NUMBERS = [
    "98.13", "97.87", "37.67", "89.75", "52.30", "93.97", "92.66",
    "68.50", "96.47", "96.63", "20,030,803", "3,361", "0.0008", "12.8",
    "803", "243", "118", "5–30", "512×640", "150", "20", "16", "640",
    "39", "5:0.5:1", "0.01", "0.5", "7.5",
]

# 组 6：占位符
PLACEHOLDER_RE = re.compile(r"TODO|TBD|待补|XXX|Lorem|×××", re.IGNORECASE)

# 组 7：关键词/摘要/图嵌入
KEYWORD_MIN, KEYWORD_MAX = 3, 5
ABSTRACT_MIN_CHARS, ABSTRACT_MAX_CHARS = 150, 400
MIN_FIGURE_EMBEDS = 8

# 组 9：manifest 契约
MANIFEST_FIG_IDS = {f"Fig{i}" for i in range(1, 11)}
MANIFEST_ALLOWED_STATUS = {"planned", "generated", "final"}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _get_paper_text() -> str:
    """读取论文正文；不存在时给出可操作的失败信息。"""
    if not PAPER_PATH.exists():
        pytest.fail(
            f"论文正文不存在: {PAPER_PATH}\n"
            f"（组 1–7 为 TDD 契约断言，正文未写时有意失败。"
            f"可用环境变量 PAPER_PATH 指向实际正文文件。）"
        )
    return PAPER_PATH.read_text(encoding="utf-8")


def _extract_headings(text: str) -> list[tuple[int, str]]:
    """返回 [(level, title), ...]，level 为 # 数量。"""
    out = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.strip())
        if m:
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def _find_heading_positions(text: str, label: str) -> list[int]:
    """返回正文中匹配 label 的标题行号（1-based）。

    label 可为 `摘要`（兼容 `# 摘要`/`## 摘要`/`**摘要**`）或 `## 1 引言`
    （提取纯标题部分后按任意层级匹配）。
    """
    title = re.sub(r"^#+\s*", "", label).strip()
    positions = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if re.match(rf"^#{{1,6}}\s*{re.escape(title)}", stripped):
            positions.append(i)
        elif re.match(rf"^\*\*{re.escape(title)}\*\*", stripped):
            positions.append(i)
    return positions


def _normalize_number(text: str) -> str:
    """数字规范化：容忍千分位逗号、en/em 破折号、乘号 × 的排版变体（不改变数值）。"""
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "x")
    )


def _count_cjk(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


# ---------------------------------------------------------------------------
# 组 1：结构
# ---------------------------------------------------------------------------
def test_01_structure_headings_ordered():
    text = _get_paper_text()
    headings = _extract_headings(text)
    heading_lines = [ln for ln, _ in headings]

    missing = []
    positions: list[int] = []
    for label, _display in REQUIRED_STRUCTURE:
        pos = _find_heading_positions(text, label)
        if not pos:
            missing.append(label)
            continue
        positions.append(pos[0])

    assert not missing, f"缺少必需标题: {missing}"
    assert positions == sorted(positions), (
        f"标题顺序错误: {list(zip([d for _, d in REQUIRED_STRUCTURE], positions))}"
    )
    # 摘要/关键词必须出现在 ## 1 引言 之前
    intro_pos = _find_heading_positions(text, "1 引言")[0]
    assert positions[0] < intro_pos and positions[1] < intro_pos, (
        "摘要/关键词必须位于 ## 1 引言 之前"
    )
    # 参考文献必须位于 ## 7 结论 之后
    concl_pos = _find_heading_positions(text, "7 结论")[0]
    assert positions[-1] > concl_pos, "参考文献必须位于 ## 7 结论 之后"


# ---------------------------------------------------------------------------
# 组 2：方法覆盖
# ---------------------------------------------------------------------------
def test_02_method_coverage():
    text = _get_paper_text()
    missing = []
    for display, aliases in METHOD_KEYWORDS:
        if not any(a in text for a in aliases):
            missing.append(display)
    assert not missing, f"正文缺少以下方法/策略关键词: {missing}"


# ---------------------------------------------------------------------------
# 组 3：四要素块
# ---------------------------------------------------------------------------
def test_03_four_element_blocks():
    text = _get_paper_text()
    lines = text.splitlines()

    # 定位所有 `### 5.x` 方法块
    block_starts = []
    for i, line in enumerate(lines):
        if re.match(r"^###\s+5\.\d+", line.strip()):
            block_starts.append(i)

    assert block_starts, "正文中未找到任何 `### 5.x` 方法块"

    problems = []
    for idx, start in enumerate(block_starts):
        end = block_starts[idx + 1] if idx + 1 < len(block_starts) else len(lines)
        block_lines = lines[start:end]
        subblocks = [
            re.match(r"^####\s+(.+?)\s*$", ln.strip()).group(1)
            for ln in block_lines
            if re.match(r"^####\s+", ln.strip())
        ]
        title = re.match(r"^###\s+(.+?)\s*$", block_lines[0].strip()).group(1)
        if len(subblocks) != 4 or set(subblocks) != REQUIRED_SUBBLOCKS:
            problems.append(f"{title}: 子块={subblocks}（应为 {sorted(REQUIRED_SUBBLOCKS)}）")

    assert not problems, f"四要素块不完整:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# 组 4：图完整性
# ---------------------------------------------------------------------------
def test_04_figure_integrity():
    text = _get_paper_text()
    refs = re.findall(r"!\[图(\d+)\]\(figures/([^)]+\.png)\)", text)
    assert refs, "正文中未找到任何 `![图N](figures/xxx.png)` 引用"

    from PIL import Image

    bad = []
    for num, fname in refs:
        fpath = FIGURES_DIR / fname
        if not fpath.exists():
            bad.append(f"图{num} 文件不存在: {fname}")
            continue
        try:
            with Image.open(fpath) as im:
                im.verify()
        except Exception as exc:  # noqa: BLE001
            bad.append(f"图{num} 无法用 PIL 打开: {fname} ({exc})")
            continue
        if fpath.stat().st_size <= 10 * 1024:
            bad.append(f"图{num} 小于 10KB: {fname} ({fpath.stat().st_size} bytes)")

    # 无孤儿图：figures/ 下每个文件都必须被正文引用
    if FIGURES_DIR.exists():
        referenced = {fname for _, fname in refs}
        orphans = sorted(
            p.name for p in FIGURES_DIR.iterdir()
            if p.is_file() and p.name not in referenced
        )
        if orphans:
            bad.append(f"figures/ 下存在未被正文引用的孤儿图: {orphans}")

    assert not bad, "图完整性检查失败:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# 组 5：数字抽查
# ---------------------------------------------------------------------------
def test_05_landmark_numbers():
    text = _get_paper_text()
    norm_text = _normalize_number(text)
    missing = [
        n for n in LANDMARK_NUMBERS
        if _normalize_number(n) not in norm_text
    ]
    assert not missing, (
        f"正文缺少以下地标数字（禁止四舍五入，须精确出现）: {missing}"
    )


# ---------------------------------------------------------------------------
# 组 6：无占位符
# ---------------------------------------------------------------------------
def test_06_no_placeholders():
    text = _get_paper_text()
    hits = sorted(set(PLACEHOLDER_RE.findall(text)))
    assert not hits, f"正文包含占位符: {hits}"


# ---------------------------------------------------------------------------
# 组 7：关键词 / 摘要 / 图嵌入
# ---------------------------------------------------------------------------
def test_07_keywords_abstract_figure_count():
    text = _get_paper_text()

    # 关键词：3–5 个（支持 `关键词：A、B、C` 同行，或 `## 关键词` 标题 + 下一行内容）
    lines = text.splitlines()
    kw_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^#{0,6}\s*\**关键词", ln.strip())),
        None,
    )
    assert kw_idx is not None, "正文缺少关键词行"
    kw_content = re.sub(r"^#{0,6}\s*\**关键词\**\s*[:：]?\s*", "", lines[kw_idx].strip())
    if not kw_content:
        for ln in lines[kw_idx + 1:]:
            if ln.strip():
                kw_content = ln.strip()
                break
    keywords = [k for k in re.split(r"[、,，;；]", kw_content) if k.strip()]
    assert KEYWORD_MIN <= len(keywords) <= KEYWORD_MAX, (
        f"关键词数量应为 {KEYWORD_MIN}–{KEYWORD_MAX} 个，实际 {len(keywords)}: {keywords}"
    )

    # 摘要：150–400 字（CJK 字符数）
    lines = text.splitlines()
    abs_start = next(
        (i for i, ln in enumerate(lines) if re.match(r"^#{0,6}\s*\**摘要", ln.strip())),
        None,
    )
    assert abs_start is not None, "正文缺少摘要"
    abs_end = next(
        (i for i in range(abs_start + 1, len(lines))
         if re.match(r"^#{1,6}\s+", lines[i].strip())),
        len(lines),
    )
    abstract = "\n".join(lines[abs_start + 1:abs_end])
    n_cjk = _count_cjk(abstract)
    assert ABSTRACT_MIN_CHARS <= n_cjk <= ABSTRACT_MAX_CHARS, (
        f"摘要 CJK 字数应为 {ABSTRACT_MIN_CHARS}–{ABSTRACT_MAX_CHARS}，实际 {n_cjk}"
    )

    # 图嵌入数 ≥ 8
    n_figs = len(re.findall(r"!\[图\d+\]", text))
    assert n_figs >= MIN_FIGURE_EMBEDS, (
        f"图嵌入数应 ≥ {MIN_FIGURE_EMBEDS}，实际 {n_figs}"
    )


# ---------------------------------------------------------------------------
# 组 8：CJK 字体 smoke（基建 GREEN）
# ---------------------------------------------------------------------------
def test_08_common_style_cjk_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import common_style

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title("舰艇红外检测 中文渲染测试")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("mAP@0.5")
    ax.plot([0, 1, 2], [0.5, 0.8, 0.9], color=common_style.PALETTE[0])

    out = common_style.save_fig(fig, tmp_path / "smoke.png")
    plt.close(fig)

    assert out.exists(), "save_fig 未产出文件"
    assert out.stat().st_size > 0, "save_fig 产出空文件"
    # save_fig 内部捕获 glyph missing 警告并 raise，走到这里即证明中文渲染无豆腐块


# ---------------------------------------------------------------------------
# 组 9：manifest 有效（基建 GREEN）
# ---------------------------------------------------------------------------
def test_09_manifest_valid():
    assert MANIFEST_PATH.exists(), f"manifest 不存在: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    figs = manifest.get("figures")
    assert isinstance(figs, dict), "manifest 缺少 figures 对象"
    assert set(figs.keys()) == MANIFEST_FIG_IDS, (
        f"manifest 必须锁定 Fig1–Fig10，实际: {sorted(figs.keys())}"
    )

    filenames = []
    for fid in sorted(figs, key=lambda k: int(k[3:])):
        entry = figs[fid]
        assert isinstance(entry, dict), f"{fid} 条目必须是对象"
        assert entry.get("filename", "").endswith(".png"), f"{fid} 缺少 .png filename"
        assert entry.get("title"), f"{fid} 缺少 title"
        assert entry.get("description"), f"{fid} 缺少 description"
        assert entry.get("status") in MANIFEST_ALLOWED_STATUS, (
            f"{fid} status 非法: {entry.get('status')}"
        )
        filenames.append(entry["filename"])

    assert len(filenames) == len(set(filenames)), "manifest 中存在重复 filename"