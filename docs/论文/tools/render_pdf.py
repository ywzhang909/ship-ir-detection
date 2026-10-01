#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""论文.md → PDF 渲染脚本（TDD 约束内，纯基建，不触碰正文/测试）。

用法:
    uv run python docs/论文/tools/render_pdf.py [论文.md 路径] [输出.pdf 路径]

依赖: uv run --with weasyprint,markdown python docs/论文/tools/render_pdf.py ...
       （或使用已就绪的 /tmp/pdfenv 虚拟环境）
依赖系统: weasyprint (libpango 等) + Noto CJK 字体（已探测：/usr/share/fonts/opentype/noto/）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_PATH = Path(__file__).resolve().parents[2] / "论文.md"
OUT_PATH = Path(__file__).resolve().parents[1] / "论文.pdf"
CJK_FONT = (
    '"Noto Serif CJK SC", "Noto Sans CJK SC", "Noto Sans CJK SC Regular", sans-serif'
)

HEADER_FOOTER_HTML = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@page {{
  size: A4;
  margin: 2.2cm 1.8cm 2.2cm 1.8cm;
  @bottom-center {{
    content: counter(page) " / " counter(pages);
    font-family: {font};
    font-size: 9pt; color: #888;
  }}
}}
html, body {{
  font-family: {font};
  font-size: 10.5pt; line-height: 1.75; color: #1a1a1a; text-align: justify;
}}
h1 {{ font-size: 17pt; text-align: center; margin: 0.2em 0 0.6em; line-height: 1.4; }}
h2 {{ font-size: 14pt; border-bottom: 1.5pt solid #2c5f8a; padding-bottom: 3pt;
      margin-top: 1.2em; color: #17466b; break-after: avoid; }}
h3 {{ font-size: 12pt; color: #1c4a6e; margin-top: 1em; break-after: avoid; }}
h4 {{ font-size: 11pt; color: #333; margin-top: 0.8em; break-after: avoid; }}
p {{ margin: 0.35em 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 0.6em 0; }}
th, td {{ border: 0.75pt solid #9fb4c6; padding: 3pt 5pt; text-align: center; }}
th {{ background: #eaf1f7; color: #1c4a6e; font-weight: 700; }}
tr:nth-child(even) td {{ background: #f7fafc; }}
img {{ max-width: 100%; height: auto; margin: 0.4em 0; }}
figure {{ margin: 0.7em 0; text-align: center; }}
figcaption {{ font-size: 9.5pt; color: #555; margin-top: 2pt; text-align: center; }}
code {{ font-family: "Noto Sans Mono CJK SC", monospace; font-size: 9pt;
       background: #f2f4f6; padding: 1pt 3pt; border-radius: 2pt; }}
pre {{ background: #f6f8fa; border: 0.75pt solid #dde4ea; padding: 6pt 8pt;
      font-size: 8.5pt; overflow-wrap: break-word; white-space: pre-wrap; }}
blockquote {{ border-left: 3pt solid #2c5f8a; margin: 0.5em 0; padding-left: 0.8em;
             color: #444; background: #f4f8fb; }}
a {{ color: #17466b; text-decoration: none; }} ul,ol {{ margin: 0.4em 0 0.4em 1.4em; }}
li {{ margin: 0.2em 0; }}
</style>
</head><body>
"""


def md_to_html(md_text: str, font: str = CJK_FONT) -> str:
    from markdown import markdown  # 延迟导入：仅渲染路径需要

    # 图引用保持 markdown 默认输出（HTML <p><img ...></p> 包裹为空行隔开的段落）
    body = markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    return HEADER_FOOTER_HTML.format(font=font) + body + "\n</body></html>\n"


def render(md_path: Path, pdf_path: Path) -> None:
    from weasyprint import HTML  # 延迟导入：仅渲染路径需要

    md_text = md_path.read_text(encoding="utf-8")
    html = md_to_html(md_text)
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    print(f"PDF 已生成: {pdf_path} ({pdf_path.stat().st_size} bytes)")


if __name__ == "__main__":
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER_PATH
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    render(md_path, pdf_path)
