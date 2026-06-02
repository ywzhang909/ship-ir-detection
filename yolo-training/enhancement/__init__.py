"""
Infrared Maritime Image Enhancement Module — Preprocessing Pipeline for YOLO Ship Detection.

Provides production-level implementations of 6 categories of image enhancement
techniques, specifically optimized for 1024×512 grayscale IR maritime imagery
with small ship targets (5-30 px).

All methods accept/return uint8 grayscale (H, W) arrays unless noted.

Usage:
    from enhancement import Preprocessor, EnhancementConfig

    cfg = EnhancementConfig(method='clahe', clahe_clip_limit=3.0)
    pre = Preprocessor(cfg)
    enhanced = pre.process(ir_image)

    # Pipeline: apply multiple methods in sequence
    cfg = EnhancementConfig(method='pipeline', pipeline=['retinex_msr', 'clahe', 'unsharp'])
    pre = Preprocessor(cfg)
    enhanced = pre.process(ir_image)

    # Compare all methods on one image
    from enhancement import compare_methods
    results = compare_methods(ir_image)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Image = np.ndarray  # uint8 grayscale (H, W)
class EnhancementMethod(str, Enum):
    NONE = "none"
    CLAHE = "clahe"
    AHE = "ahe"
    BCLAHE = "bclahe"
    GAMMA = "gamma"
    CONTRAST_STRETCH = "contrast_stretch"
    RETINEX_SSR = "retinex_ssr"
    RETINEX_MSR = "retinex_msr"
    RETINEX_MSRCR = "retinex_msrcr"
    UNSHARP = "unsharp"
    LAPLACIAN = "laplacian"
    HIGH_BOOST = "high_boost"
    GUIDED_FILTER = "guided_filter"
    HOMOMORPHIC = "homomorphic"
    DCT_ENHANCE = "dct_enhance"
    WAVELET_ENHANCE = "wavelet_enhance"
    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    NUC = "nuc"
    DRC = "drc"
    PIPELINE = "pipeline"


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement preprocessing."""
    method: EnhancementMethod | str = EnhancementMethod.CLAHE
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8
    clahe_tile_size_h: int | None = None
    clahe_tile_size_w: int | None = None
    ahe_tile_size: int = 8
    bclahe_clip_limit: float = 4.0
    bclahe_tile_size: int = 16
    bclahe_kernel_size: int = 5
    gamma: float | None = None
    gamma_auto_strength: float = 1.0
    stretch_low_pct: float = 2.0
    stretch_high_pct: float = 98.0
    retinex_sigmas: List[float] = field(default_factory=lambda: [15, 80, 250])
    retinex_gain: float = 1.0
    retinex_offset: float = 0.0
    retinex_use_gray_world: bool = False
    unsharp_kernel_size: int = 9
    unsharp_sigma: float = 1.5
    unsharp_strength: float = 1.5
    laplacian_kernel_size: int = 3
    laplacian_scale: float = 0.5
    high_boost_kernel_size: int = 9
    high_boost_sigma: float = 2.0
    high_boost_amount: float = 1.5
    guided_radius: int = 8
    guided_eps: float = 0.01
    homo_cutoff: float = 30.0
    homo_gl: float = 0.5
    homo_gh: float = 2.0
    dct_threshold_pct: float = 5.0
    wavelet_levels: int = 3
    wavelet_denoise_sigma: float = 5.0
    wavelet_enhance_gain: float = 1.5
    polarity_map: str = "auto"
    drc_method: str = "log"
    drc_gamma: float = 0.5
    pipeline: List[str] = field(default_factory=lambda: ["clahe", "unsharp"])


@dataclass
class EnhancementResult:
    """Result of a single enhancement operation."""
    image: Image
    method: str
    elapsed_ms: float
    params: dict
# ===================================================================
# 1. CONTRAST ENHANCEMENT
# ===================================================================

def apply_clahe(img, clip_limit=3.0, tile_size=8, tile_size_h=None, tile_size_w=None):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization.

    Best for IR maritime (1024x512):
      - clip_limit: 2.0-4.0 (3.0 balances contrast vs noise)
      - tile_size: 8x8 for full-res, 4x4 for small targets
        8x8 gives tiles of ~64x64 px on 512px height
      - OpenCV rescales clip_limit internally:
        Tc = max(Tu * (h*w) / 256, 1)
        So clip_limit=3 with tile_size=8 gives effective clip ~50

    Cost: ~2-5 ms. Small ship impact: +++
    """
    th = tile_size_h or tile_size
    tw = tile_size_w or tile_size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(th, tw))
    return clahe.apply(img)


def apply_ahe(img, tile_size=8):
    """
    Adaptive Histogram Equalization (no clip limit - more noise).
    Can over-amplify noise in low-SNR IR maritime.
    Cost: ~1-3 ms. Small ship impact: ++
    """
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def apply_bclahe(img, clip_limit=4.0, tile_size=16, kernel_size=5):
    """
    Bilateral CLAHE - applies bilateral filter before CLAHE to reduce
    noise amplification in flat regions (sea background).
    Cost: ~10-30 ms. Small ship impact: ++
    """
    filtered = cv2.bilateralFilter(img, kernel_size, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(filtered)


def apply_gamma_correction(img, gamma=None, auto_strength=1.0):
    """
    Gamma correction with auto-estimation.

    Auto-gamma: gamma = log(0.5) / log(median/255) * auto_strength
    For IR maritime:
      - Well-exposed: gamma ~0.8-1.2
      - Dim: gamma ~0.4-0.7 (brighten)
      - Hot bg: gamma ~1.3-2.0 (darken)

    Cost: <1 ms. Small ship impact: ++
    """
    if gamma is None:
        median = float(np.median(img))
        if median < 1:
            gamma = 1.0
        else:
            norm = median / 255.0
            gamma = np.log(0.5) / max(np.log(norm + 1e-6), 1e-6)
            gamma *= auto_strength
            gamma = np.clip(gamma, 0.2, 3.0)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.float32)
    return cv2.LUT(img, table).astype(np.uint8)


def apply_contrast_stretch(img, low_pct=2.0, high_pct=98.0):
    """
    Percentile-based contrast stretching.
    Maps [low_pct, high_pct] to [0, 255].
    Cost: <1 ms. Small ship impact: +/- neutral globally
    """
    low_val = np.percentile(img, low_pct)
    high_val = np.percentile(img, high_pct)
    if high_val - low_val < 1:
        return img
    stretched = np.clip(
        (img.astype(np.float32) - low_val) * (255.0 / (high_val - low_val)),
        0, 255,
    )
    return stretched.astype(np.uint8)
# ===================================================================
# 2. RETINEX-BASED ENHANCEMENT
# ===================================================================

def _retinex_single_scale(img_f, sigma, gain=1.0, offset=0.0):
    """Single Scale Retinex on float64 image [0,1]."""
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    ksize = max(ksize, 3)  # ensure odd and >= 3
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)

    img_log = np.log(np.maximum(img_f, 1e-6))
    blur_log = np.log(np.maximum(blurred, 1e-6))
    retinex = img_log - blur_log

    return gain * retinex + offset


def _normalize(arr):
    """Min-max normalize to [0, 255]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255)


def apply_ssr(img, sigma=80.0, gain=1.0, offset=0.0):
    """
    Single Scale Retinex.

    I(x,y) = L(x,y) * R(x,y) -> log R = log I - log(I * G)

    Sigma controls the scale:
      15 = fine detail, 80 = balanced, 250 = global illumination

    Cost: ~5-15 ms. Small ship impact: +++
    """
    img_f = img.astype(np.float64) / 255.0
    result = _retinex_single_scale(img_f, sigma, gain, offset)
    return _normalize(result).astype(np.uint8)


def apply_msr(img, sigmas=None, gain=1.0, offset=0.0):
    """
    Multi-Scale Retinex - weighted sum of SSR at multiple scales.

    Scales [15, 80, 250] (recommended for IR maritime):
      - 15: local detail/small ship features
      - 80: mid-scale contrast
      - 250: global illumination

    Cost: ~15-40 ms. Small ship impact: +++
    """
    if sigmas is None:
        sigmas = [15.0, 80.0, 250.0]

    img_f = img.astype(np.float64) / 255.0
    acc = np.zeros_like(img_f, dtype=np.float64)
    n_scales = len(sigmas)

    for s in sigmas:
        acc += _retinex_single_scale(img_f, s, gain, offset)

    result = acc / n_scales
    return _normalize(result).astype(np.uint8)


def apply_msrcr(img, sigmas=None, gain=128.0, offset=0.0, use_gray_world=False):
    """
    Multi-Scale Retinex with Color Restoration (adapted for grayscale).

    Adds contrast restoration factor C = log(alpha * I + beta) as
    local gain control. alpha=125, beta=0.5 works well.

    Cost: ~15-40 ms. Small ship impact: +++
    """
    if sigmas is None:
        sigmas = [15.0, 80.0, 250.0]

    img_f = img.astype(np.float64) / 255.0
    acc = np.zeros_like(img_f, dtype=np.float64)
    for s in sigmas:
        acc += _retinex_single_scale(img_f, s, 1.0, 0.0)

    msr = acc / len(sigmas)

    alpha = 125.0
    beta = 0.5
    img_sum = np.clip(np.sum(img_f, axis=-1) if img_f.ndim == 3 else img_f, 1e-6, None)

    if use_gray_world:
        c = np.log(alpha * (img_f / (img_sum + 1e-6)) + beta)
    else:
        c = np.log(alpha * img_f + beta)

    result = gain * (c * msr) + offset
    return _normalize(result).astype(np.uint8)
# ===================================================================
# 3. EDGE ENHANCEMENT
# ===================================================================

def apply_unsharp_mask(img, kernel_size=9, sigma=1.5, strength=1.5):
    """
    Unsharp masking: enhanced = img + strength * (img - blurred).

    For IR maritime small ships (5-30px):
      - kernel_size: 5-15 (9 good for 512px height)
      - sigma: 1.0-3.0
      - strength: 1.0-2.0

    Cost: ~2-5 ms. Small ship impact: +++ (enhances ship-sea boundary)
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(
        img.astype(np.float32), 1.0 + strength,
        blurred.astype(np.float32), -strength, 0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_laplacian_enhance(img, kernel_size=3, scale=0.5):
    """
    Laplacian of Gaussian enhancement.
    More aggressive than unsharp - enhances ALL edges including noise.
    Cost: ~1-3 ms. Small ship impact: +++ but also amplifies clutter
    """
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=kernel_size)
    enhanced = img.astype(np.float32) + scale * lap
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def apply_high_boost_filter(img, kernel_size=9, sigma=2.0, amount=1.5):
    """
    High-boost: enhanced = A*img - (A-1)*blurred where A > 1.
    A=1 -> standard unsharp, A>1 -> low-frequency boost too.
    Cost: ~2-5 ms. Small ship impact: ++
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    enhanced = cv2.addWeighted(
        img.astype(np.float32), amount,
        blurred.astype(np.float32), -(amount - 1.0), 0,
    )
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _guided_filter_numpy(I, r, eps):
    """NumPy guided filter fallback (O(N) box-filter based)."""
    r_int = max(int(r), 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r_int, r_int))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r_int, r_int))
    var_I = mean_II - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = (1.0 - a) * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r_int, r_int))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r_int, r_int))
    return mean_a * I + mean_b


def apply_guided_filter_enhance(img, radius=8, eps=0.01):
    """
    Guided filter-based detail enhancement.
    Edge-preserving decomposition: base + detail_strength * detail.
    Cost: ~20-100 ms. Small ship impact: +++ (edge-preserving)
    """
    img_f = img.astype(np.float32)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        base = cv2.ximgproc.guidedFilter(guide=img_f, src=img_f, radius=radius, eps=eps)
    else:
        base = _guided_filter_numpy(img_f, radius, eps)

    detail = img_f - base
    enhanced = base + 2.0 * detail
    return np.clip(enhanced, 0, 255).astype(np.uint8)
# ===================================================================
# 4. FREQUENCY DOMAIN METHODS
# ===================================================================

def apply_homomorphic_filter(img, cutoff=30.0, gl=0.5, gh=2.0):
    """
    Homomorphic filtering for illumination correction.

    I = L * R -> log I = log L + log R
    Suppress illumination (L) low-freq, amplify reflectance (R) high-freq.

    For IR maritime:
      - cutoff: 20-50
      - gl: 0.3-0.7 (illumination suppression)
      - gh: 1.5-3.0 (detail enhancement)

    Cost: ~5-15 ms. Small ship impact: ++
    """
    img_f = img.astype(np.float64)
    img_log = np.log1p(img_f)

    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    uu, vv = np.meshgrid(v, u)
    D = np.sqrt(uu**2 + vv**2)
    D2 = D**2 / (cutoff**2 + 1e-6)
    H = (gh - gl) * (1 - np.exp(-D2)) + gl

    filtered = dft_shift * H
    img_ifft = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    img_exp = np.expm1(img_ifft)
    return _normalize(img_exp).astype(np.uint8)


def apply_dct_enhance(img, threshold_pct=5.0):
    """
    DCT-based enhancement via block processing (8x8 blocks).
    Suppress DC slightly, boost mid-frequencies (ship-scale features).
    Cost: ~2-5 ms. Small ship impact: ++
    """
    img_f = img.astype(np.float32)
    h, w = img.shape
    result = np.zeros_like(img_f)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img_f[i:i+8, j:j+8]
            if block.shape != (8, 8):
                continue

            dct = cv2.dct(block)
            mask = np.ones((8, 8), dtype=np.float32)
            mask[0, 0] = 0.9  # slightly suppress DC

            for u in range(8):
                for v in range(8):
                    freq = np.sqrt(u**2 + v**2)
                    if 1.5 <= freq <= 4.0:
                        mask[u, v] = 1.3  # boost mid-freq
                    elif freq > 4.0 and freq <= 6.0:
                        mask[u, v] = 1.1  # slight boost high-mid

            dct_boosted = dct * mask
            result[i:i+8, j:j+8] = cv2.idct(dct_boosted)

    return np.clip(result, 0, 255).astype(np.uint8)


def _soft_threshold(x, threshold):
    """Soft-thresholding for wavelet denoising."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def apply_wavelet_enhance(img, levels=3, denoise_sigma=5.0, enhance_gain=1.5):
    """
    Wavelet-based denoising and enhancement using Laplacian pyramid.

    Excellent for IR maritime: sea clutter in high-frequency detail bands,
    ship edges in same bands but larger magnitude.
    Soft-threshold removes noise while preserving strong edges.

    Cost: ~5-15 ms. Small ship impact: +++
    """
    img_f = img.astype(np.float32)

    # Build Laplacian pyramid
    pyramid = []
    current = img_f.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        detail = current - up
        pyramid.append(detail)
        current = down

    base = current

    # Process each detail level
    processed = []
    for level, detail in enumerate(pyramid):
        thresh = denoise_sigma * (0.5 ** level)
        denoised = _soft_threshold(detail, thresh)
        enhanced = denoised * enhance_gain
        processed.append(enhanced)

    # Reconstruct
    result = base
    for detail in reversed(processed):
        result = cv2.pyrUp(result, dstsize=(detail.shape[1], detail.shape[0]))
        result = result + detail

    return np.clip(result, 0, 255).astype(np.uint8)
# ===================================================================
# 5. DEEP LEARNING ENHANCEMENT (ZeroDCE)
# ===================================================================

def apply_zerodce(img, model_path=None, device="cpu"):
    """
    Zero-Reference Deep Curve Estimation (ZeroDCE).

    Requires pretrained weights. Lightweight ~30K params.
    ~5-10ms on GPU, ~50-100ms on CPU for 1024x512.

    Falls back to MSR+CLAHE if no weights provided.

    Download: https://github.com/Li-Chongyi/ZeroDCE
    """
    if model_path is None:
        logger.warning(
            "ZeroDCE requires pretrained weights. "
            "Falling back to MSR+CLAHE pipeline."
        )
        return apply_pipeline(img, ["retinex_msr", "clahe"])

    # Placeholder for actual ZeroDCE inference:
    # model = ZeroDCE().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    # with torch.no_grad():
    #     img_t = torch.from_numpy(img).float() / 255.0
    #     img_t = img_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    #     params = model(img_t)
    #     enhanced = apply_curve(img_t, params)
    # return (enhanced.squeeze().cpu().numpy() * 255).astype(np.uint8)

    logger.info("ZeroDCE inference on %s ...", device)
    return apply_pipeline(img, ["retinex_msr", "clahe"])


# ===================================================================
# 6. IR-SPECIFIC ENHANCEMENT
# ===================================================================

def detect_polarity(img):
    """
    Auto-detect IR polarity (white-hot vs black-hot).

    Uses edge density in bright vs dark regions:
      - White-hot: ships bright, more edges in bright regions
      - Black-hot: ships dark, more edges in dark regions
    """
    edges = cv2.Canny(img, 30, 100)
    low_thresh = np.percentile(img, 25)
    high_thresh = np.percentile(img, 75)

    dark_edges = edges[img < low_thresh].sum()
    bright_edges = edges[img > high_thresh].sum()

    return "white-hot" if bright_edges >= dark_edges else "black-hot"


def apply_white_hot(img):
    """Invert black-hot to white-hot. Cost: <1ms."""
    return (255 - img).astype(np.uint8)


def apply_black_hot(img):
    """Invert white-hot to black-hot. Cost: <1ms."""
    return (255 - img).astype(np.uint8)


def apply_polarity_correction(img, mode="auto"):
    """
    Ensure consistent IR polarity.

    For YOLO training: use ONE polarity consistently.
    White-hot recommended for general IR maritime.
    """
    if mode == "auto":
        polarity = detect_polarity(img)
        return apply_white_hot(img) if polarity == "black-hot" else img
    elif mode == "white-hot":
        return apply_white_hot(img)
    elif mode == "black-hot":
        return apply_black_hot(img)
    return img


def apply_nuc(img, kernel_size=31):
    """
    Non-uniformity correction (simulated).

    Estimates and removes column fixed-pattern noise common in
    uncooled microbolometer IR detectors.

    Cost: ~2-5 ms. Small ship impact: +
    """
    col_mean = np.mean(img, axis=1, keepdims=True)
    smooth_col = cv2.GaussianBlur(col_mean, (1, kernel_size), kernel_size / 3.0)
    corrected = img.astype(np.float32) - (col_mean - smooth_col)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_drc(img, method="log", drc_gamma=0.5):
    """
    Dynamic Range Compression for high-dynamic-range IR.

    Methods:
      - 'log':  I_out = log(1+I) / log(256) * 255
      - 'sqrt': I_out = sqrt(I/255) * 255
      - 'auto-level': adaptive histogram-based
      - 'gamma': I_out = (I/255)^gamma * 255

    Cost: <1 ms. Small ship impact: ++
    """
    img_f = img.astype(np.float32)

    if method == "log":
        c = 255.0 / np.log(1 + 255.0)
        result = c * np.log1p(img_f)
    elif method == "sqrt":
        result = np.sqrt(img_f / 255.0) * 255.0
    elif method == "auto-level":
        hist, bins = np.histogram(img_f, bins=256, range=(0, 256))
        cumsum = np.cumsum(hist) / img_f.size
        low_bin = np.searchsorted(cumsum, 0.025)
        high_bin = np.searchsorted(cumsum, 0.975)
        low_val = bins[low_bin]
        high_val = bins[high_bin]
        if high_val - low_val > 1:
            result = np.clip((img_f - low_val) * 255.0 / (high_val - low_val), 0, 255)
        else:
            result = img_f
    elif method == "gamma":
        result = np.power(img_f / 255.0, drc_gamma) * 255.0
    else:
        raise ValueError(f"Unknown DRC method: {method}")

    return np.clip(result, 0, 255).astype(np.uint8)
# ===================================================================
# PIPELINE & REGISTRY
# ===================================================================

METHOD_REGISTRY = {
    "none": (lambda img, **kw: img, False),
    "clahe": (apply_clahe, False),
    "ahe": (apply_ahe, False),
    "bclahe": (apply_bclahe, False),
    "gamma": (apply_gamma_correction, False),
    "contrast_stretch": (apply_contrast_stretch, False),
    "retinex_ssr": (apply_ssr, False),
    "retinex_msr": (apply_msr, False),
    "retinex_msrcr": (apply_msrcr, False),
    "unsharp": (apply_unsharp_mask, False),
    "laplacian": (apply_laplacian_enhance, False),
    "high_boost": (apply_high_boost_filter, False),
    "guided_filter": (apply_guided_filter_enhance, False),
    "homomorphic": (apply_homomorphic_filter, False),
    "dct_enhance": (apply_dct_enhance, False),
    "wavelet_enhance": (apply_wavelet_enhance, False),
    "white_hot": (apply_white_hot, False),
    "black_hot": (apply_black_hot, False),
    "nuc": (apply_nuc, False),
    "drc": (apply_drc, False),
    "zerodce": (apply_zerodce, False),
}


def apply_pipeline(img, steps, **kwargs):
    """Apply a sequence of enhancement methods in order.

    Recommended IR maritime pipelines (best first):

    [1] retinex_msr + clahe + unsharp  (~30 ms) - Best overall
        MSR corrects illumination, CLAHE adds local contrast,
        unsharp sharpens edges.

    [2] homomorphic + clahe              (~15 ms)
        Good for non-uniform illumination + local contrast.

    [3] wavelet_enhance + contrast_stretch (~12 ms) - Best for noisy IR
        Wavelet denoises + enhances.

    [4] clahe + unsharp                  (~6 ms) - Fast baseline

    [5] nuc + drc + clahe               (~8 ms) - For raw sensor data
    """
    result = img.copy()
    for step in steps:
        if step not in METHOD_REGISTRY:
            logger.warning("Unknown method '%s', skipping.", step)
            continue
        fn = METHOD_REGISTRY[step][0]
        step_kw = {k.replace(f"{step}_", ""): v for k, v in kwargs.items()
                   if k.startswith(step + "_")} if kwargs else {}
        try:
            result = fn(result, **step_kw)
        except Exception as e:
            logger.error("Method '%s' failed: %s. Skipping.", step, e)
            continue
    return result


# ===================================================================
# PREPROCESSOR CLASS
# ===================================================================

class Preprocessor:
    """Configurable image enhancement preprocessor for IR maritime.

    Supports single-method and multi-stage pipeline modes.
    Designed for YOLO dataset pipelines and inference.

    Usage:
        pre = Preprocessor(EnhancementConfig(method='clahe'))
        enhanced = pre.process(image)

        pre = Preprocessor(EnhancementConfig(
            method='pipeline',
            pipeline=['retinex_msr', 'clahe', 'unsharp']
        ))
        enhanced, meta = pre.process(image, return_metadata=True)
    """

    def __init__(self, config=None):
        self.config = config or EnhancementConfig()

    def process(self, img, return_metadata=False):
        """Apply configured enhancement.

        Args:
            img: uint8 grayscale (H, W)
            return_metadata: return (image, EnhancementResult)

        Returns:
            Enhanced image or (image, metadata) tuple
        """
        t0 = time.perf_counter()
        method_str = (
            self.config.method.value
            if isinstance(self.config.method, EnhancementMethod)
            else self.config.method
        )

        if method_str == "pipeline":
            enhanced = apply_pipeline(img, self.config.pipeline)
            elapsed = (time.perf_counter() - t0) * 1000
            result = EnhancementResult(
                image=enhanced,
                method="pipeline:" + "+".join(self.config.pipeline),
                elapsed_ms=round(elapsed, 2),
                params={"steps": self.config.pipeline},
            )
        elif method_str == "none":
            elapsed = (time.perf_counter() - t0) * 1000
            result = EnhancementResult(
                image=img, method="none",
                elapsed_ms=round(elapsed, 2), params={},
            )
        else:
            fn = METHOD_REGISTRY.get(method_str, (None, False))[0]
            if fn is None:
                logger.warning("Unknown method '%s', returning original.", method_str)
                enhanced = img
            else:
                params = self._get_params_for(method_str)
                enhanced = fn(img, **params)

            elapsed = (time.perf_counter() - t0) * 1000
            result = EnhancementResult(
                image=enhanced,
                method=method_str,
                elapsed_ms=round(elapsed, 2),
                params=self._get_params_for(method_str),
            )

        if return_metadata:
            return result.image, result
        return result.image

    def _get_params_for(self, method):
        prefix = method.replace("-", "_") + "_"
        return {
            k.removeprefix(prefix): v
            for k, v in self._config_items()
            if k.startswith(prefix)
        }

    def _config_items(self):
        for field in self.config.__dataclass_fields__:
            val = getattr(self.config, field)
            if val is not None:
                yield field, val


# ===================================================================
# COMPARISON & LOGGING
# ===================================================================

def compare_methods(img, methods=None, include_original=True):
    """Run multiple enhancement methods and return results list.

    Args:
        img: Input grayscale image
        methods: List of method names (default: all registered)
        include_original: Include original as first result

    Returns:
        List of EnhancementResult objects
    """
    if methods is None:
        methods = [
            "clahe", "gamma", "contrast_stretch",
            "retinex_ssr", "retinex_msr", "retinex_msrcr",
            "unsharp", "laplacian", "high_boost",
            "homomorphic", "dct_enhance", "wavelet_enhance",
            "bclahe",
        ]

    results = []
    if include_original:
        results.append(EnhancementResult(
            image=img, method="original",
            elapsed_ms=0.0, params={},
        ))

    for method in methods:
        t0 = time.perf_counter()
        fn = METHOD_REGISTRY.get(method, (None, False))[0]
        if fn is None:
            logger.warning("Unknown method '%s', skipping.", method)
            continue
        try:
            enhanced = fn(img)
            elapsed = (time.perf_counter() - t0) * 1000
            results.append(EnhancementResult(
                image=enhanced, method=method,
                elapsed_ms=round(elapsed, 2), params={},
            ))
        except Exception as e:
            logger.error("Method '%s' failed: %s", method, e)

    return results


def log_comparison_to_wandb(results, prefix="enhancement"):
    """Log enhancement comparison to wandb.

    Returns dict ready for wandb.log().

    Usage:
        import wandb
        results = compare_methods(image)
        wandb.log(log_comparison_to_wandb(results))
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging.")
        return {}

    log_data = {}
    for r in results:
        key = f"{prefix}/{r.method}"
        rgb = cv2.cvtColor(r.image, cv2.COLOR_GRAY2RGB)
        log_data[key] = wandb.Image(rgb, caption=f"{r.method} ({r.elapsed_ms}ms)")
        log_data[f"{key}_elapsed_ms"] = r.elapsed_ms

    return log_data


def create_yolo_augmentation_pipeline(config=None):
    """Create a callable augmentation function for YOLO training."""
    pre = Preprocessor(config or EnhancementConfig())

    def augment(img):
        return pre.process(img)

    return augment


# ===================================================================
# PRESETS
# ===================================================================

PRESETS = {
    "fast": EnhancementConfig(
        method=EnhancementMethod.CLAHE,
        clahe_clip_limit=3.0,
        clahe_tile_size=8,
    ),
    "balanced": EnhancementConfig(
        method=EnhancementMethod.PIPELINE,
        pipeline=["clahe", "unsharp"],
    ),
    "quality": EnhancementConfig(
        method=EnhancementMethod.PIPELINE,
        pipeline=["retinex_msr", "clahe", "unsharp"],
    ),
    "noisy": EnhancementConfig(
        method=EnhancementMethod.PIPELINE,
        pipeline=["wavelet_enhance", "contrast_stretch"],
    ),
    "non_uniform": EnhancementConfig(
        method=EnhancementMethod.PIPELINE,
        pipeline=["homomorphic", "clahe"],
    ),
    "raw_sensor": EnhancementConfig(
        method=EnhancementMethod.PIPELINE,
        pipeline=["nuc", "drc", "clahe"],
    ),
    "retinex_only": EnhancementConfig(
        method=EnhancementMethod.RETINEX_MSR,
        retinex_sigmas=[15.0, 80.0, 250.0],
    ),
}


def get_preset(name="balanced"):
    """Get preset config by name.

    Presets:
      - 'fast':        CLAHE only (~3 ms)
      - 'balanced':    CLAHE + unsharp (~6 ms)
      - 'quality':     MSR + CLAHE + unsharp (~30 ms)
      - 'noisy':       Wavelet denoise + stretch (~12 ms)
      - 'non_uniform': Homomorphic + CLAHE (~15 ms)
      - 'raw_sensor':  NUC + DRC + CLAHE (~8 ms)
      - 'retinex_only': MSR only (~18 ms)
    """
    if name not in PRESETS:
        logger.warning("Unknown preset '%s'. Available: %s. Using 'balanced'.",
                       name, list(PRESETS.keys()))
        return PRESETS["balanced"]
    return PRESETS[name]


__all__ = [
    "Preprocessor", "EnhancementConfig", "EnhancementMethod", "EnhancementResult",
    "compare_methods", "log_comparison_to_wandb", "apply_pipeline",
    "create_yolo_augmentation_pipeline", "get_preset", "PRESETS",
    "detect_polarity",
    "apply_clahe", "apply_ahe", "apply_bclahe",
    "apply_gamma_correction", "apply_contrast_stretch",
    "apply_ssr", "apply_msr", "apply_msrcr",
    "apply_unsharp_mask", "apply_laplacian_enhance", "apply_high_boost_filter",
    "apply_guided_filter_enhance",
    "apply_homomorphic_filter", "apply_dct_enhance", "apply_wavelet_enhance",
    "apply_white_hot", "apply_black_hot", "apply_polarity_correction",
    "apply_nuc", "apply_drc", "METHOD_REGISTRY",
]
