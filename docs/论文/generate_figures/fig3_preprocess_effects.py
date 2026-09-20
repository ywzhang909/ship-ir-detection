# -*- coding: utf-8 -*-
"""Fig3 预处理效果对比 — 真实渲染。

对真实 demo 红外图（ship_detector/tests/demo_ir.png，1024×512）运行
yolo-training/preprocessing_module.py 的真实预处理代码，生成 3×4 对比面板：

    CLAHE / AWB / 空域tophat+CLAHE / 频域Butterworth+CLAHE /
    MSR / 同态 / DWT / DoG / 各向异性 / 中值 / 双边 / 双流3通道

所有面板均调用真实函数/管线，参数取自模块内记录参数（_PRESET_CONFIGS 预设
与 PreprocessingConfig 默认值）。禁止手绘/伪造。

输出: docs/论文/figures/fig3_preprocess_effects.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # ship/
YOLO_DIR = REPO_ROOT / "yolo-training"
FIG_DIR = REPO_ROOT / "docs" / "论文" / "figures"
GEN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(YOLO_DIR))       # preprocessing_module
sys.path.insert(0, str(GEN_DIR))        # common_style

import common_style                      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import preprocessing_module as pm        # noqa: E402

# ---------------------------------------------------------------------------
# 源图：真实 demo 红外图
# ---------------------------------------------------------------------------
SRC_IMG = REPO_ROOT / "ship_detector" / "tests" / "demo_ir.png"
bgr = cv2.imread(str(SRC_IMG))
assert bgr is not None, f"无法读取源图: {SRC_IMG}"
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# ---------------------------------------------------------------------------
# 面板定义：全部使用真实代码 + 记录参数
# ---------------------------------------------------------------------------
# 记录参数来源（模块内 _PRESET_CONFIGS / PreprocessingConfig 默认值）：
#   tophat_clahe 预设: tophat_kernel=31, clahe_clip_limit=3.0, tile=(8,8), unsharp=0.5
#   awb_butterworth_clahe 预设: butterworth low=8, high=80, order=2, clip=3.0, unsharp=0.5
#   retinex_gamma 预设: retinex_scales=[15,80,250]
#   awb_wavelet_clahe 预设: db4, level=1, LL*0.3, LH/HL*1.5, HH*0.3
#   dog_retinex 预设: gauss_diff_sigma1=1.0, sigma2=3.0
#   median_clahe 预设: median_kernel=5
#   awb 预设: wb_corner_fraction=0.06, wb_gray_target=128.0, gain_clamp=(0.5,3.0)
#   PreprocessingConfig 默认: homomorphic cutoff=30/high=2.0/low=0.5,
#                             anisotropic iter=5/k=50, bilateral d=9/σc=75/σs=75
#   awb_dual_fusion: PreprocessingPipelineDual() 默认双流

def _pipe(cfg: pm.PreprocessingConfig):
    return pm.PreprocessingPipeline(cfg)(gray)


PANELS = [
    {
        "name": "CLAHE",
        "title": "CLAHE 对比度增强",
        "params": "clip=3.0, tile=8×8",
        "source": "tophat_clahe 预设增强阶段",
        "color": False,
        "render": lambda: pm.enhance_clahe(gray, clip_limit=3.0, tile_grid=(8, 8)),
    },
    {
        "name": "AWB",
        "title": "AWB 自动白平衡",
        "params": "corner=0.06, target=128",
        "source": "awb_tophat_clahe 预设色彩阶段",
        "color": True,
        "render": lambda: pm.correct_white_balance_sea(
            bgr, corner_fraction=0.06, gray_target=128.0, gain_clamp=(0.5, 3.0)
        ),
    },
    {
        "name": "空域tophat+CLAHE",
        "title": "空域 Top-hat + CLAHE",
        "params": "tophat k=31, clip=3.0, unsharp=0.5",
        "source": "tophat_clahe 预设",
        "color": False,
        "render": lambda: _pipe(pm.PreprocessingConfig(
            sea_clutter_method=pm.SeaClutterMethod.TOPHAT,
            tophat_kernel=31,
            enhancement_method=pm.EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            edge_enhance=True,
            unsharp_strength=0.5,
        )),
    },
    {
        "name": "频域Butterworth+CLAHE",
        "title": "频域 Butterworth + CLAHE",
        "params": "low=8, high=80, clip=3.0, unsharp=0.5",
        "source": "awb_butterworth_clahe 预设(去AWB)",
        "color": False,
        "render": lambda: _pipe(pm.PreprocessingConfig(
            sea_clutter_method=pm.SeaClutterMethod.BUTTERWORTH_BPF,
            butterworth_cutoff_low=8.0,
            butterworth_cutoff_high=80.0,
            butterworth_order=2,
            enhancement_method=pm.EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            edge_enhance=True,
            unsharp_strength=0.5,
        )),
    },
    {
        "name": "MSR",
        "title": "MSR 多尺度 Retinex",
        "params": "scales=[15,80,250]",
        "source": "retinex_gamma 预设增强阶段",
        "color": False,
        "render": lambda: pm.enhance_retinex_msr(gray, scales=[15, 80, 250], gain=1.0, offset=0.0),
    },
    {
        "name": "同态",
        "title": "同态滤波",
        "params": "cutoff=30, high=2.0, low=0.5",
        "source": "PreprocessingConfig 默认",
        "color": False,
        "render": lambda: pm.enhance_homomorphic(gray, cutoff=30.0, high=2.0, low=0.5),
    },
    {
        "name": "DWT",
        "title": "DWT 小波域抑制",
        "params": "db4, LL×0.3, LH/HL×1.5, HH×0.3",
        "source": "awb_wavelet_clahe 预设抑制阶段",
        "color": False,
        "render": lambda: pm.suppress_wavelet(
            gray, wavelet_name="db4", level=1,
            scale_ll=0.3, scale_lh=1.5, scale_hl=1.5, scale_hh=0.3,
        ),
    },
    {
        "name": "DoG",
        "title": "DoG 高斯差分",
        "params": "σ1=1.0, σ2=3.0",
        "source": "dog_retinex 预设抑制阶段",
        "color": False,
        "render": lambda: pm.suppress_gauss_diff(gray, sigma1=1.0, sigma2=3.0),
    },
    {
        "name": "各向异性",
        "title": "各向异性扩散",
        "params": "iter=5, k=50",
        "source": "PreprocessingConfig 默认",
        "color": False,
        "render": lambda: pm.suppress_anisotropic(gray, iterations=5, k=50.0),
    },
    {
        "name": "中值",
        "title": "中值滤波",
        "params": "kernel=5",
        "source": "median_clahe 预设抑制阶段",
        "color": False,
        "render": lambda: pm.suppress_median(gray, kernel=5),
    },
    {
        "name": "双边",
        "title": "双边滤波",
        "params": "d=9, σc=75, σs=75",
        "source": "PreprocessingConfig 默认",
        "color": False,
        "render": lambda: pm.suppress_bilateral(gray, d=9, sigma_color=75.0, sigma_space=75.0),
    },
    {
        "name": "双流3通道",
        "title": "双流融合 3 通道",
        "params": "[spatial, freq, orig]",
        "source": "awb_dual_fusion 预设",
        "color": True,
        "render": lambda: pm.PreprocessingPipelineDual()(bgr),
    },
]

# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
results = []
for p in PANELS:
    out = p["render"]()
    assert out is not None and out.size > 0, f"面板 {p['name']} 渲染失败"
    results.append(out)

# ---------------------------------------------------------------------------
# 3×4 面板排版
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(20, 9.5))
fig.suptitle(
    f"红外海面舰艇图像预处理效果对比（源图：demo_ir.png，{W}×{H}）",
    fontsize=16, y=0.995,
)

for idx, (p, out) in enumerate(zip(PANELS, results)):
    ax = axes[idx // 4][idx % 4]
    if p["color"]:
        # 3 通道结果：BGR → RGB 显示
        disp = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        ax.imshow(disp)
    else:
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{p['title']}\n{p['params']}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#4C72B0")
        spine.set_linewidth(1.2)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = FIG_DIR / "fig3_preprocess_effects.png"
common_style.save_fig(fig, out_path, dpi=200)
print(f"已保存: {out_path}")

# ---------------------------------------------------------------------------
# 参数一致性校验（对照模块内记录参数）
# ---------------------------------------------------------------------------
print("\n===== 面板参数一致性校验 =====")
checks = []

def check(label, used, recorded):
    ok = used == recorded
    checks.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {label}: 使用={used}  记录={recorded}")

# 1. CLAHE
check("CLAHE clip_limit", 3.0, pm._PRESET_CONFIGS["tophat_clahe"].clahe_clip_limit)
check("CLAHE tile_grid", (8, 8), pm._PRESET_CONFIGS["tophat_clahe"].clahe_tile_grid)
# 2. AWB
check("AWB corner_fraction", 0.06, pm._PRESET_CONFIGS["awb_tophat_clahe"].wb_corner_fraction)
check("AWB gray_target", 128.0, pm._PRESET_CONFIGS["awb_tophat_clahe"].wb_gray_target)
check("AWB gain_clamp", (0.5, 3.0), pm._PRESET_CONFIGS["awb_tophat_clahe"].wb_gain_clamp)
# 3. tophat+CLAHE
check("tophat kernel", 31, pm._PRESET_CONFIGS["tophat_clahe"].tophat_kernel)
check("tophat_clahe clip", 3.0, pm._PRESET_CONFIGS["tophat_clahe"].clahe_clip_limit)
check("tophat_clahe unsharp", 0.5, pm._PRESET_CONFIGS["tophat_clahe"].unsharp_strength)
# 4. Butterworth+CLAHE
check("butterworth low", 8.0, pm._PRESET_CONFIGS["awb_butterworth_clahe"].butterworth_cutoff_low)
check("butterworth high", 80.0, pm._PRESET_CONFIGS["awb_butterworth_clahe"].butterworth_cutoff_high)
check("butterworth order", 2, pm._PRESET_CONFIGS["awb_butterworth_clahe"].butterworth_order)
check("butterworth_clahe clip", 3.0, pm._PRESET_CONFIGS["awb_butterworth_clahe"].clahe_clip_limit)
# 5. MSR
check("MSR scales", [15, 80, 250], pm._PRESET_CONFIGS["retinex_gamma"].retinex_scales)
# 6. 同态
cfg_def = pm.PreprocessingConfig()
check("homomorphic cutoff", 30.0, cfg_def.homomorphic_cutoff)
check("homomorphic high", 2.0, cfg_def.homomorphic_high)
check("homomorphic low", 0.5, cfg_def.homomorphic_low)
# 7. DWT
check("wavelet name", "db4", pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_name)
check("wavelet level", 1, pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_level)
check("wavelet scale_ll", 0.3, pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_scale_ll)
check("wavelet scale_lh", 1.5, pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_scale_lh)
check("wavelet scale_hl", 1.5, pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_scale_hl)
check("wavelet scale_hh", 0.3, pm._PRESET_CONFIGS["awb_wavelet_clahe"].wavelet_scale_hh)
# 8. DoG
check("DoG sigma1", 1.0, pm._PRESET_CONFIGS["dog_retinex"].gauss_diff_sigma1)
check("DoG sigma2", 3.0, pm._PRESET_CONFIGS["dog_retinex"].gauss_diff_sigma2)
# 9. 各向异性
check("anisotropic iter", 5, cfg_def.anisotropic_iterations)
check("anisotropic k", 50.0, cfg_def.anisotropic_k)
# 10. 中值
check("median kernel", 5, pm._PRESET_CONFIGS["median_clahe"].median_kernel)
# 11. 双边
check("bilateral d", 9, cfg_def.bilateral_d)
check("bilateral sigma_color", 75.0, cfg_def.bilateral_sigma_color)
check("bilateral sigma_space", 75.0, cfg_def.bilateral_sigma_space)
# 12. 双流3通道
dual = pm.PreprocessingPipelineDual()
check("dual spatial tophat k", dual.spatial_pipeline.config.tophat_kernel, 31)
check("dual spatial clip", dual.spatial_pipeline.config.clahe_clip_limit, 3.0)
check("dual freq method", dual.freq_pipeline.config.sea_clutter_method,
      pm.SeaClutterMethod.BUTTERWORTH_BPF)
check("dual freq low", dual.freq_pipeline.config.butterworth_cutoff_low, 8.0)
check("dual freq high", dual.freq_pipeline.config.butterworth_cutoff_high, 80.0)

print(f"\n校验结果: {sum(checks)}/{len(checks)} 项一致")
assert all(checks), "存在参数不一致项！"
print("全部面板参数与源记录参数一致。")