"""
IR Maritime Preprocessing Module for Ship Detection.

Comprehensive preprocessing pipeline combining:
  - Sea clutter suppression (background estimation, filtering)
  - Image contrast enhancement (CLAHE, Retinex, gamma)
  - Edge enhancement (unsharp masking, high-boost)
  - Sea-sky-line (SSL) detection for targeted region processing
  - Morphological operations (top-hat/bottom-hat for target enhancement)

All methods support both PIL Image and numpy array (HWC/BGR) inputs,
and are wandb-logging compatible.

Usage:
    from preprocessing_module import PreprocessingPipeline, PreprocessingConfig

    pipe = PreprocessingPipeline(config=PreprocessingConfig(
        sea_clutter_method="tophat",
        enhancement_method="clahe",
        clip_limit=3.0,
    ))
    enhanced_img = pipe(image)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pywt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Configuration
# ---------------------------------------------------------------------------

class ColorCorrectionMethod(str, Enum):
    """Color correction / white balance methods for color IR images."""
    NONE = "none"
    SEA_WHITE_BALANCE = "sea_wb"    # Auto white balance using sea surface corners
    SEA_GRAY_WORLD = "sea_gray"     # Gray-world with sea-region prior


class SeaClutterMethod(str, Enum):
    """Sea clutter suppression / background subtraction methods."""
    NONE = "none"
    MEDIAN = "median"               # Median filter for salt-and-pepper
    BILATERAL = "bilateral"         # Edge-preserving smoothing
    TOPHAT = "tophat"               # White top-hat (bright targets on dark)
    BOTTOMHAT = "bottomhat"         # Black bottom-hat (dark targets on bright)
    TOPHAT_BOTTOMHAT = "tophat_bottomhat"  # Combined: (original + tophat) - bottomhat
    GAUSSIAN_DIFF = "gauss_diff"    # Difference of Gaussians (bandpass)
    ANISOTROPIC = "anisotropic"     # Anisotropic diffusion (edge-preserving)
    BUTTERWORTH_BPF = "butterworth_bpf"  # Butterworth band-pass (frequency domain)
    WAVELET = "wavelet"                  # Wavelet-domain clutter suppression (DWT + sub-band scaling)
    RPCA = "rpca"                        # RPCA low-rank background subtraction (IALM solver)


class EnhancementMethod(str, Enum):
    """Image contrast and detail enhancement methods."""
    NONE = "none"
    CLAHE = "clahe"                 # Contrast Limited Adaptive Histogram Equalization
    CLAHE_DETAIL = "clahe_detail"   # CLAHE + detail enhancement
    RETINEX_SSR = "retinex_ssr"     # Single Scale Retinex
    RETINEX_MSR = "retinex_msr"     # Multi-Scale Retinex
    GAMMA = "gamma"                 # Adaptive gamma correction
    HOMOMORPHIC = "homomorphic"     # Homomorphic filtering (frequency domain)
    UNMASK = "unmask"               # Unsharp masking
    ADAPTIVE_EQ = "adaptive_eq"     # Adaptive equalization (local)


class SSLMetric(str, Enum):
    """Sea-sky-line detection metrics."""
    ROW_EDGE = "row_edge"           # Row-wise edge energy
    ROW_VARIANCE = "row_variance"   # Row-wise intensity variance
    OTSU_SPLIT = "otsu_split"       # Otsu threshold on row gradient


@dataclass
class PreprocessingConfig:
    """Configuration for the full preprocessing pipeline.

    Attributes are organized by pipeline stage:
      0. color_correction → 1. sea_clutter → 2. ssl_detection → 3. enhancement → 4. edge_enhance
    """
    # -- Color correction (Stage 0: only for color IR imagery) --
    color_correction_method: ColorCorrectionMethod = ColorCorrectionMethod.NONE
    wb_corner_fraction: float = 0.06          # Fraction of width/height for corner sampling
    wb_gray_target: float = 128.0             # Target gray value for white balance (0-255)
    wb_gain_clamp: tuple = (0.5, 3.0)         # Per-channel gain clamp range [min, max]
    wb_apply_smoothing: bool = True           # Blur gains to avoid hard transitions

    # -- Sea clutter suppression --
    sea_clutter_method: SeaClutterMethod = SeaClutterMethod.TOPHAT_BOTTOMHAT
    median_kernel: int = 5               # Kernel size for median filter
    bilateral_d: int = 9                 # Diameter for bilateral filter
    bilateral_sigma_color: float = 75.0  # Color sigma for bilateral
    bilateral_sigma_space: float = 75.0  # Space sigma for bilateral
    tophat_kernel: int = 31              # Kernel size for top-hat (odd, large for sea)
    bottomhat_kernel: int = 31           # Kernel size for bottom-hat
    gauss_diff_sigma1: float = 1.0       # Small sigma for DoG
    gauss_diff_sigma2: float = 3.0       # Large sigma for DoG
    anisotropic_iterations: int = 5      # Anisotropic diffusion iterations
    anisotropic_k: float = 50.0          # Anisotropic diffusion K (edge sensitivity)
    butterworth_cutoff_low: float = 8.0  # Butterworth BPF low cutoff (sea clutter)
    butterworth_cutoff_high: float = 80.0  # Butterworth BPF high cutoff (noise)
    butterworth_order: int = 2           # Butterworth filter order
    wavelet_name: str = "db4"            # Wavelet family+order (e.g. 'db4', 'sym4')
    wavelet_level: int = 1               # DWT decomposition level
    wavelet_scale_ll: float = 0.3        # Scaling factor for LL sub-band (sea clutter)
    wavelet_scale_lh: float = 1.5        # Scaling factor for LH sub-band (horizontal edges)
    wavelet_scale_hl: float = 1.5        # Scaling factor for HL sub-band (vertical edges)
    wavelet_scale_hh: float = 0.3        # Scaling factor for HH sub-band (noise)

    # -- RPCA low-rank background subtraction --
    rpca_method: str = "truncated_svd"   # "truncated_svd" (fast) or "rpca_ialm" (slow, high-quality)
    rpca_rank: int = 10                  # Rank for truncated SVD background approximation
    rpca_max_iter: int = 30              # Max IALM iterations (only for rpca_ialm method)
    rpca_tol: float = 1e-6               # IALM convergence tolerance (only for rpca_ialm method)

    # -- Sea-sky-line detection --
    detect_ssl: bool = False             # Enable SSL detection (can help but adds compute)
    ssl_metric: SSLMetric = SSLMetric.ROW_EDGE
    ssl_smooth_sigma: float = 3.0        # Gaussian pre-smooth before SSL detection

    # -- Contrast enhancement --
    enhancement_method: EnhancementMethod = EnhancementMethod.CLAHE_DETAIL
    clahe_clip_limit: float = 3.0        # CLAHE clip limit (2-4 typical)
    clahe_tile_grid: tuple = (8, 8)      # CLAHE tile grid size
    retinex_scales: list = field(default_factory=lambda: [15, 80, 250])  # MSR scales
    retinex_gain: float = 1.0            # Retinex gain
    retinex_offset: float = 0.0          # Retinex offset
    gamma_auto: bool = True              # Auto-compute gamma from image stats
    gamma_value: float = 1.2             # Fixed gamma if auto=False
    homomorphic_cutoff: float = 30.0     # Homomorphic filter cutoff frequency
    homomorphic_high: float = 2.0        # High frequency gain
    homomorphic_low: float = 0.5         # Low frequency gain

    # -- Edge enhancement --
    edge_enhance: bool = True
    unsharp_strength: float = 0.5        # Unsharp masking strength (0.3-1.0)
    unsharp_kernel: int = 5              # Unsharp kernel size

    # -- Pipeline control --
    apply_to_train: bool = True          # Apply during training preprocessing
    apply_to_inference: bool = True      # Apply during inference
    target_size: tuple | None = None     # (W, H) to resize after preprocessing


# ---------------------------------------------------------------------------
# Color Correction (Stage 0: for color false-color IR imagery)
# ---------------------------------------------------------------------------

def correct_white_balance_sea(
    image: np.ndarray,
    corner_fraction: float = 0.06,
    gray_target: float = 128.0,
    gain_clamp: tuple = (0.5, 3.0),
    apply_smoothing: bool = True,
) -> np.ndarray:
    """Auto white balance using sea surface corners for color IR maritime images.

    For color IR (false-color thermal), the bottom-left and bottom-right corners
    are typically open sea surface which should be neutral gray. This function
    computes per-channel gains from these corner regions and applies them to
    correct color cast.

    Algorithm:
      1. Sample bottom-left and bottom-right corner regions
      2. Compute mean R, G, B across both regions → (r_mean, g_mean, b_mean)
      3. Gain = gray_target / mean for each channel
      4. Clamp gains to [gain_clamp_min, gain_clamp_max]
      5. Apply gains to full image (optionally smoothed)

    Args:
        image: Color BGR image (H, W, 3), uint8 [0, 255].
        corner_fraction: Fraction of image height/width for corner sampling.
        gray_target: Target neutral intensity for sea corners (0-255).
        gain_clamp: (min, max) for per-channel gain clamping.
        apply_smoothing: If True, apply Gaussian blur to gain map.

    Returns:
        Color-corrected image (same shape/dtype).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        logger.debug("White balance skipped: not a 3-channel color image")
        return image

    h, w = image.shape[:2]
    ch = max(8, int(h * corner_fraction))
    cw = max(8, int(w * corner_fraction))

    # Bottom-left corner
    bl = image[h - ch:, :cw, :]
    # Bottom-right corner
    br = image[h - ch:, w - cw:, :]

    # Mean across both corner regions (per channel)
    corner_mean = np.mean(np.concatenate([bl.ravel(), br.ravel()]))

    # Per-channel means from both corners
    bl_mean = bl.mean(axis=(0, 1))   # (B, G, R)
    br_mean = br.mean(axis=(0, 1))
    sea_mean = (bl_mean + br_mean) / 2.0  # (B, G, R) in BGR order

    # Compute gains: target / actual
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = np.where(sea_mean > 0, gray_target / sea_mean, 1.0)
    gains = np.clip(gains, gain_clamp[0], gain_clamp[1])

    logger.debug(
        "AWB: sea mean (BGR)=[%.1f, %.1f, %.1f], gains=[%.3f, %.3f, %.3f]",
        sea_mean[0], sea_mean[1], sea_mean[2],
        gains[0], gains[1], gains[2],
    )

    # Apply gains
    corrected = image.astype(np.float32)
    for c in range(3):
        corrected[:, :, c] *= gains[c]

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def correct_gray_world_sea(
    image: np.ndarray,
    corner_fraction: float = 0.06,
    gray_target: float = 128.0,
    sea_weight: float = 0.7,
    gain_clamp: tuple = (0.5, 3.0),
) -> np.ndarray:
    """Gray-world white balance with sea-region prior for IR maritime.

    Combines global gray-world assumption with sea-corner prior.
    Gains = gray_target / (sea_weight * sea_mean + (1-sea_weight) * global_mean)

    This is more robust when corners are partially occluded (e.g., wakes).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    h, w = image.shape[:2]
    ch = max(8, int(h * corner_fraction))
    cw = max(8, int(w * corner_fraction))

    # Sea corner mean
    bl = image[h - ch:, :cw, :]
    br = image[h - ch:, w - cw:, :]
    sea_mean = (bl.mean(axis=(0, 1)) + br.mean(axis=(0, 1))) / 2.0

    # Global mean
    global_mean = image.mean(axis=(0, 1))

    # Weighted combination
    combined_mean = sea_weight * sea_mean + (1.0 - sea_weight) * global_mean

    with np.errstate(divide="ignore", invalid="ignore"):
        gains = np.where(combined_mean > 0, gray_target / combined_mean, 1.0)
    gains = np.clip(gains, gain_clamp[0], gain_clamp[1])

    corrected = image.astype(np.float32)
    for c in range(3):
        corrected[:, :, c] *= gains[c]
    return np.clip(corrected, 0, 255).astype(np.uint8)


# Registry of color correction methods
COLOR_CORRECTION_FN: dict[ColorCorrectionMethod, Callable] = {
    ColorCorrectionMethod.NONE: lambda img, **kw: img,
    ColorCorrectionMethod.SEA_WHITE_BALANCE: correct_white_balance_sea,
    ColorCorrectionMethod.SEA_GRAY_WORLD: correct_gray_world_sea,
}


# ---------------------------------------------------------------------------
# Frequency-Domain Sea Clutter Suppression (Butterworth Band-Pass)
# ---------------------------------------------------------------------------

def suppress_butterworth_bpf(
    image: np.ndarray,
    cutoff_low: float = 8.0,     # Low cutoff → suppresses sea waves
    cutoff_high: float = 80.0,   # High cutoff → suppresses noise
    order: int = 2,
) -> np.ndarray:
    """Butterworth band-pass filter in frequency domain.

    Suppresses low-frequency sea clutter (wave swells, gradual illumination)
    and high-frequency noise while preserving mid-frequency ship targets.

    Pipeline integration: processes image in frequency domain using FFT,
    applies a Butterworth band-pass mask, then inverse FFT.

    Args:
        image: Grayscale image (H, W), uint8.
        cutoff_low: Low cutoff frequency (pixels). Frequencies below this are
                    suppressed (sea background, wave swells).
        cutoff_high: High cutoff frequency (pixels). Frequencies above this
                     are suppressed (sensor noise, speckle).
        order: Butterworth filter order (higher = sharper transition).

    Returns:
        Filtered image (uint8).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_float = gray.astype(np.float32)
    rows, cols = gray.shape

    # FFT
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)

    # Frequency grid
    center_r, center_c = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - center_r) ** 2 + (x - center_c) ** 2)

    # Butterworth band-pass: BPF = HPF(low) * LPF(high)
    # High-pass: suppress low frequencies (sea background)
    hpf = 1.0 / (1.0 + (cutoff_low / (dist + 1e-8)) ** (2 * order))
    # Low-pass: suppress high frequencies (noise)
    lpf = 1.0 / (1.0 + (dist / cutoff_high) ** (2 * order))
    bandpass = hpf * lpf

    # Apply & inverse FFT
    dft_filtered = dft_shift * bandpass
    img_ifft = np.fft.ifft2(np.fft.ifftshift(dft_filtered))
    result = np.real(img_ifft)

    # Normalize to [0, 255]
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


# Register Butterworth BPF in the sea clutter registry (will add below)


# ---------------------------------------------------------------------------
# Sea-Sky-Line Detection
# ---------------------------------------------------------------------------

def detect_ssl_row_edge(image: np.ndarray, sigma: float = 3.0) -> int:
    """Detect sea-sky-line using row-wise horizontal edge energy.

    Smoothes image, computes horizontal Sobel gradient, sums absolute
    gradient per row, returns row index with maximum edge response.

    Args:
        image: Grayscale image (H, W).
        sigma: Pre-smoothing Gaussian sigma.

    Returns:
        Row index of detected SSL, or image height // 2 if detection fails.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Pre-smooth
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    # Horizontal edges (vertical gradient)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_energy = np.sum(np.abs(grad_y), axis=1)
    # Smooth the energy profile
    energy_smooth = cv2.GaussianBlur(edge_energy.astype(np.float32), (1, 5), 1.0).flatten()
    ssl_row = int(np.argmax(energy_smooth))
    # Fallback: middle of image
    if ssl_row < image.shape[0] * 0.1 or ssl_row > image.shape[0] * 0.9:
        ssl_row = image.shape[0] // 2
    return ssl_row


def detect_ssl_variance(image: np.ndarray, sigma: float = 3.0) -> int:
    """Detect sea-sky-line using row-wise intensity variance.

    The sky region typically has low variance, sea has higher variance
    (wave texture). The SSL is at the transition.

    Args:
        image: Grayscale image (H, W).
        sigma: Pre-smoothing Gaussian sigma.

    Returns:
        Row index of detected SSL.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    height = gray.shape[0]
    # Compute variance in sliding windows
    half_h = 10
    variances = np.zeros(height)
    for r in range(half_h, height - half_h):
        patch = gray[r - half_h:r + half_h, :]
        variances[r] = np.var(patch)

    # Find the row where variance crosses a threshold (sky→sea transition)
    var_norm = variances / (variances.max() + 1e-8)
    # Look for sharp increase in variance moving top→bottom
    diff = np.diff(var_norm)
    diff_smooth = cv2.GaussianBlur(diff.astype(np.float32), (1, 7), 2.0).flatten()
    ssl_row = int(np.argmax(diff_smooth) + half_h)
    if ssl_row < image.shape[0] * 0.1 or ssl_row > image.shape[0] * 0.9:
        ssl_row = image.shape[0] // 2
    return ssl_row


SSL_DETECTORS: dict[SSLMetric, Callable] = {
    SSLMetric.ROW_EDGE: detect_ssl_row_edge,
    SSLMetric.ROW_VARIANCE: detect_ssl_variance,
    SSLMetric.OTSU_SPLIT: detect_ssl_row_edge,  # uses same signature signature
}


# ---------------------------------------------------------------------------
# Sea Clutter Suppression Methods
# ---------------------------------------------------------------------------

def suppress_median(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median filter for salt-and-pepper noise suppression."""
    return cv2.medianBlur(image, kernel)


def suppress_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Bilateral filter for edge-preserving noise reduction."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def suppress_tophat(image: np.ndarray, kernel: int = 31) -> np.ndarray:
    """White top-hat transform: enhances bright targets on dark background.

    Computes: img - morphological_open(img)
    This extracts bright ship targets from dark sea background.
    """
    if kernel % 2 == 0:
        kernel += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, k)
    return cv2.subtract(image, opened)


def suppress_bottomhat(image: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Black bottom-hat transform: enhances dark targets on bright background.

    Computes: morphological_close(img) - img
    """
    if kernel % 2 == 0:
        kernel += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
    return cv2.subtract(closed, image)


def suppress_tophat_bottomhat(image: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Combined top-hat + bottom-hat: enhances both bright and dark targets.

    result = original + tophat - bottomhat
    This is the most effective general-purpose clutter suppression for IR maritime.
    """
    th = suppress_tophat(image, kernel)
    bh = suppress_bottomhat(image, kernel)
    enhanced = cv2.add(image, th)
    enhanced = cv2.subtract(enhanced, bh)
    return enhanced


def suppress_gauss_diff(
    image: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 3.0,
) -> np.ndarray:
    """Difference of Gaussians bandpass filter.

    Suppresses low-frequency background (sea) and high-frequency noise,
    while preserving mid-frequency ship targets.
    """
    blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog = cv2.subtract(blur1, blur2)
    # Normalize to [0, 255]
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return dog


def suppress_anisotropic(image: np.ndarray, iterations: int = 5, k: float = 50.0) -> np.ndarray:
    """Anisotropic diffusion (Perona-Malik) for edge-preserving smoothing.

    Uses OpenCV's implementation internally.

    Args:
        image: Input grayscale image.
        iterations: Number of diffusion iterations.
        k: Conductance parameter (higher = more diffusion).
    """
    img_float = image.astype(np.float32)
    for _ in range(iterations):
        # Compute gradient in 4 directions
        n = np.roll(img_float, -1, axis=0)
        s = np.roll(img_float, 1, axis=0)
        e = np.roll(img_float, -1, axis=1)
        w = np.roll(img_float, 1, axis=1)

        delta_n = n - img_float
        delta_s = s - img_float
        delta_e = e - img_float
        delta_w = w - img_float

        # Conductance function: c = exp(-(grad/k)^2)
        c_n = np.exp(-(delta_n / k) ** 2)
        c_s = np.exp(-(delta_s / k) ** 2)
        c_e = np.exp(-(delta_e / k) ** 2)
        c_w = np.exp(-(delta_w / k) ** 2)

        img_float += 0.25 * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

    return np.clip(img_float, 0, 255).astype(np.uint8)


def suppress_wavelet(
    image: np.ndarray,
    wavelet_name: str = "db4",
    level: int = 1,
    scale_ll: float = 0.3,
    scale_lh: float = 1.5,
    scale_hl: float = 1.5,
    scale_hh: float = 0.3,
) -> np.ndarray:
    """Wavelet-domain clutter suppression via sub-band coefficient scaling.

    Decomposes the image using Discrete Wavelet Transform (DWT), then scales
    the sub-bands to suppress sea clutter and enhance ship targets:

      - **LL** (low-pass approximation): low-frequency sea background → *suppressed*
      - **LH** / **HL** (horizontal/vertical detail): ship edges → *enhanced*
      - **HH** (diagonal detail): high-frequency sensor noise → *suppressed*

    Reconstructs with Inverse DWT and normalises to [0, 255].

    This is an alternative to :func:`suppress_butterworth_bpf` that operates
    in the wavelet domain instead of the Fourier domain, offering better
    spatial localisation of frequency content.

    Args:
        image: Grayscale image (H, W), uint8.
        wavelet_name: Wavelet family + order (e.g. ``'db4'``, ``'sym4'``).
        level: DWT decomposition level (1 = single-level, adequate for 640px).
        scale_ll: Multiplier for LL coefficients (0.0-0.5 → suppress sea).
        scale_lh: Multiplier for LH coefficients (1.0-2.0 → enhance ships).
        scale_hl: Multiplier for HL coefficients (1.0-2.0 → enhance ships).
        scale_hh: Multiplier for HH coefficients (0.0-0.5 → suppress noise).

    Returns:
        Wavelet-filtered image (uint8), same shape as input.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_float = gray.astype(np.float64)

    # Single-level 2D DWT
    coeffs2 = pywt.dwt2(img_float, wavelet_name, mode='symmetric')
    LL, (LH, HL, HH) = coeffs2

    # Scale sub-bands
    LL *= scale_ll
    LH *= scale_lh
    HL *= scale_hl
    HH *= scale_hh

    # Inverse DWT
    reconstructed = pywt.idwt2((LL, (LH, HL, HH)), wavelet_name, mode='symmetric')

    # Clip to valid range
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed


# ---------------------------------------------------------------------------
# RPCA (Robust Principal Component Analysis) — Low-rank background subtraction
# ---------------------------------------------------------------------------

def _rpca_ialm(
    D: np.ndarray,
    lambda_weight: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
    rho: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve RPCA via Inexact Augmented Lagrange Multiplier (IALM).

    ``min  ‖L‖_* + λ‖S‖_1   s.t.   D = L + S``

    The low-rank component *L* captures the smooth sea/background, and the
    sparse component *S* captures ship targets (plus shot noise).

    Args:
        D: Input matrix (m, n), float64.
        lambda_weight: Regularisation weight (default: 1/sqrt(max(m,n))).
        max_iter: Maximum IALM iterations.
        tol: Relative convergence tolerance.
        rho: Augmentation multiplier (> 1.0).

    Returns:
        Tuple of (L, S) — low-rank background and sparse foreground.
    """
    m, n = D.shape
    if lambda_weight is None:
        lambda_weight = 1.0 / np.sqrt(max(m, n))

    # Normalise D to [0, 1] for numerical stability
    D_norm = D.copy()
    d_min, d_max = D_norm.min(), D_norm.max()
    if d_max > d_min:
        D_norm = (D_norm - d_min) / (d_max - d_min)
    else:
        D_norm = D_norm - d_min

    # Initialise
    D_norm_fro = np.linalg.norm(D_norm, 'fro')
    if D_norm_fro < 1e-12:
        return np.zeros_like(D), D

    Y = D_norm / max(np.linalg.norm(D_norm, 2), np.linalg.norm(D_norm, np.inf) / lambda_weight)
    S = np.zeros_like(D_norm)
    mu = 1.25 / np.linalg.norm(D_norm, 2)

    for _ in range(max_iter):
        # --- Step 1: Update L via singular-value soft-thresholding ---
        U, sigma, Vt = np.linalg.svd(D_norm - S + Y / mu, full_matrices=False)
        sigma_shrunk = np.maximum(sigma - 1.0 / mu, 0.0)
        L = (U * sigma_shrunk) @ Vt

        # --- Step 2: Update S via element-wise soft-thresholding ---
        residual = D_norm - L + Y / mu
        lam_mu = lambda_weight / mu
        S = np.sign(residual) * np.maximum(np.abs(residual) - lam_mu, 0.0)

        # --- Step 3: Dual variable update ---
        Y = Y + mu * (D_norm - L - S)

        # --- Step 4: Increase mu ---
        mu *= rho

        # --- Convergence check ---
        err = np.linalg.norm(D_norm - L - S, 'fro') / D_norm_fro
        if err < tol:
            break

    # De-normalise back to original intensity range
    if d_max > d_min:
        L = L * (d_max - d_min) + d_min
        S = S * (d_max - d_min)

    return L, S


def _lowrank_foreground(
    image: np.ndarray,
    rank: int = 10,
    method: str = "truncated_svd",
    **kwargs,
) -> np.ndarray:
    """Extract foreground via low-rank background subtraction.

    Two methods are available:

    * ``"truncated_svd"`` — Fast truncated SVD (single decomposition, top-k
      singular values).  The background is approximated as the rank-k
      reconstruction; foreground = original − background.  Suitable for
      large-scale preprocessing (≈ 0.1 s per 640×480 image).
    * ``"rpca_ialm"`` — Full RPCA via Inexact Augmented Lagrange Multiplier.
      Iterative solver (≈ 5 s per image) that jointly optimises low-rank and
      sparse components.  Higher quality but too slow for dataset-scale
      preprocessing.

    Args:
        image: Grayscale image (H, W), uint8.
        rank: Number of singular values / rank for ``"truncated_svd"``.
        method: ``"truncated_svd"`` (default) or ``"rpca_ialm"``.
        **kwargs: Passed through to :func:`_rpca_ialm` when method is
                  ``"rpca_ialm"``.

    Returns:
        Foreground image as uint8, same shape as input.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_float = gray.astype(np.float64)

    if method == "rpca_ialm":
        _, S = _rpca_ialm(img_float, **kwargs)
        foreground = np.abs(S)
    else:
        # Truncated SVD — fast low-rank approximation
        U, sigma, Vt = np.linalg.svd(img_float, full_matrices=False)
        bg = (U[:, :rank] * sigma[:rank]) @ Vt[:rank, :]
        foreground = np.abs(img_float - bg)

    # Normalise to [0, 255]
    fg_min, fg_max = foreground.min(), foreground.max()
    if fg_max > fg_min:
        foreground = (foreground - fg_min) / (fg_max - fg_min) * 255.0
    else:
        foreground = np.zeros_like(foreground)

    return np.clip(foreground, 0, 255).astype(np.uint8)


def suppress_rpca(image: np.ndarray, **kwargs) -> np.ndarray:
    """Low-rank background subtraction via RPCA (alias for :func:`_lowrank_foreground`).

    By default uses fast truncated SVD (rank=10).  Pass ``method="rpca_ialm"``
    to use the full iterative RPCA solver (slower but potentially higher
    quality).

    Args:
        image: Grayscale image (H, W), uint8.
        **kwargs: Forwarded to :func:`_lowrank_foreground`.

    Returns:
        Foreground (sparse) component as uint8, same shape as input.
    """
    return _lowrank_foreground(image, **kwargs)


# Registry of sea clutter suppression methods
SEA_CLUTTER_FN: dict[SeaClutterMethod, Callable] = {
    SeaClutterMethod.NONE: lambda img, **kw: img,
    SeaClutterMethod.MEDIAN: suppress_median,
    SeaClutterMethod.BILATERAL: suppress_bilateral,
    SeaClutterMethod.TOPHAT: suppress_tophat,
    SeaClutterMethod.BOTTOMHAT: suppress_bottomhat,
    SeaClutterMethod.TOPHAT_BOTTOMHAT: suppress_tophat_bottomhat,
    SeaClutterMethod.GAUSSIAN_DIFF: suppress_gauss_diff,
    SeaClutterMethod.ANISOTROPIC: suppress_anisotropic,
    SeaClutterMethod.BUTTERWORTH_BPF: suppress_butterworth_bpf,
    SeaClutterMethod.WAVELET: suppress_wavelet,
    SeaClutterMethod.RPCA: suppress_rpca,
}


# ---------------------------------------------------------------------------
# Image Enhancement Methods
# ---------------------------------------------------------------------------

def enhance_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    The most widely effective enhancement for IR maritime imagery.
    """
    if image.ndim == 3:
        # Convert to LAB, apply CLAHE to L-channel only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_eq = clahe.apply(l)
        eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image)


def enhance_clahe_detail(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: tuple = (8, 8),
    unsharp_strength: float = 0.5,
    unsharp_kernel: int = 5,
) -> np.ndarray:
    """CLAHE + detail enhancement via unsharp masking.

    This is the preferred enhancement for IR ship detection as it
    enhances local contrast while preserving fine ship details.
    """
    eq = enhance_clahe(image, clip_limit, tile_grid)
    # Light unsharp masking on the CLAHE result
    blurred = cv2.GaussianBlur(eq, (unsharp_kernel, unsharp_kernel), 0)
    sharpened = cv2.addWeighted(eq, 1.0 + unsharp_strength, blurred, -unsharp_strength, 0)
    return sharpened


def enhance_retinex_ssr(image: np.ndarray, scale: int = 80, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Single Scale Retinex (SSR).

    Computes: log(I) - log(G * I) where G is a Gaussian surround.
    Works well for IR: normalizes illumination, enhances reflectance details.
    """
    img_float = image.astype(np.float32) + 1.0  # avoid log(0)
    blur = cv2.GaussianBlur(img_float, (0, 0), scale)
    blur = np.where(blur < 1.0, 1.0, blur)  # avoid log(0)
    retinex = np.log(img_float) - np.log(blur)
    # Apply gain/offset and stretch to [0, 255]
    retinex = retinex * gain + offset
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return retinex


def enhance_retinex_msr(
    image: np.ndarray,
    scales: list | None = None,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Multi-Scale Retinex (MSR).

    Weighted average of SSR at multiple scales. More robust than SSR.
    Default scales [15, 80, 250] capture fine, medium, and coarse details.
    """
    if scales is None:
        scales = [15, 80, 250]
    img_float = image.astype(np.float32) + 1.0
    retinex_sum = np.zeros_like(img_float, dtype=np.float32)
    for scale in scales:
        blur = cv2.GaussianBlur(img_float, (0, 0), scale)
        blur = np.where(blur < 1.0, 1.0, blur)
        retinex_sum += (np.log(img_float) - np.log(blur)) / len(scales)
    retinex_sum = retinex_sum * gain + offset
    retinex_sum = cv2.normalize(retinex_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return retinex_sum


def enhance_gamma(image: np.ndarray, gamma: float | None = None, auto: bool = True) -> np.ndarray:
    """Gamma correction with optional auto-gamma from image statistics.

    Auto-gamma is computed from the mean intensity:
      gamma = log(0.5) / log(mean/255)
    This brightens dark images and darkens bright images.
    """
    if auto:
        mean_val = np.mean(image)
        if mean_val > 0:
            gamma = np.log(0.5) / np.log(mean_val / 255.0)
            gamma = np.clip(gamma, 0.4, 2.5)
        else:
            gamma = 1.0
    elif gamma is None:
        gamma = 1.2

    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def enhance_homomorphic(
    image: np.ndarray,
    cutoff: float = 30.0,
    high: float = 2.0,
    low: float = 0.5,
) -> np.ndarray:
    """Homomorphic filtering for illumination normalization.

    Compresses dynamic range (low frequencies = illumination) while
    enhancing contrast (high frequencies = reflectance).

    Args:
        image: Input grayscale image.
        cutoff: Butterworth filter cutoff frequency.
        high: High-frequency gain (reflectance enhancement).
        low: Low-frequency gain (illumination suppression).
    """
    img_float = image.astype(np.float32) + 1.0
    rows, cols = image.shape[:2]

    # Log transform
    img_log = np.log(img_float)

    # FFT
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    # Butterworth high-pass filter
    center_r, center_c = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - center_r) ** 2 + (x - center_c) ** 2)
    butterworth = 1.0 / (1.0 + (dist / cutoff) ** (2 * 2))  # order=2
    # Homomorphic filter: low emphasis, high emphasis
    h_filter = (high - low) * (1.0 - butterworth) + low

    # Apply filter
    dft_filtered = dft_shift * h_filter

    # Inverse FFT
    img_ifft = np.fft.ifft2(np.fft.ifftshift(dft_filtered))
    img_exp = np.exp(np.real(img_ifft))

    # Normalize
    img_exp = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_exp


def enhance_unmask(
    image: np.ndarray,
    strength: float = 0.5,
    kernel: int = 5,
) -> np.ndarray:
    """Unsharp masking for edge enhancement.

    sharpened = image + strength * (image - blurred)
    """
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def enhance_adaptive_eq(image: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Adaptive equalization using local statistics.

    Enhances local contrast where std is low (flat regions get more boost).
    """
    img_float = image.astype(np.float32)
    mean = cv2.boxFilter(img_float, -1, (kernel, kernel))
    sqr_mean = cv2.boxFilter(img_float ** 2, -1, (kernel, kernel))
    std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

    # Enhancement factor: higher where std is low
    max_std = std.max() if std.max() > 0 else 1.0
    factor = 1.0 + (1.0 - std / max_std) * 0.5
    enhanced = mean + factor * (img_float - mean)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# Registry of enhancement methods
ENHANCEMENT_FN: dict[EnhancementMethod, Callable] = {
    EnhancementMethod.NONE: lambda img, **kw: img,
    EnhancementMethod.CLAHE: enhance_clahe,
    EnhancementMethod.CLAHE_DETAIL: enhance_clahe_detail,
    EnhancementMethod.RETINEX_SSR: enhance_retinex_ssr,
    EnhancementMethod.RETINEX_MSR: enhance_retinex_msr,
    EnhancementMethod.GAMMA: enhance_gamma,
    EnhancementMethod.HOMOMORPHIC: enhance_homomorphic,
    EnhancementMethod.UNMASK: enhance_unmask,
    EnhancementMethod.ADAPTIVE_EQ: enhance_adaptive_eq,
}


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Configurable preprocessing pipeline for IR maritime ship detection.

    Pipeline order:
      1. Sea-sky-line detection (optional, for targeted processing)
      2. Sea clutter suppression (top-hat, median, bilateral, etc.)
      3. Contrast enhancement (CLAHE, Retinex, gamma, etc.)
      4. Edge enhancement (unsharp masking)

    Each stage is independently configurable and can be enabled/disabled.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run full preprocessing pipeline on a single image.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) BGR.

        Returns:
            Preprocessed image.
        """
        return self.process(image)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the preprocessing pipeline.

        Args:
            image: Input image.

        Returns:
            Preprocessed image (uint8, range [0, 255]).
        """
        cfg = self.config
        img = image.copy()
        is_color = img.ndim == 3

        # ---------------------------------------------------------------
        # Stage 0: Color correction (only for color IR imagery)
        # ---------------------------------------------------------------
        color_fn = COLOR_CORRECTION_FN.get(cfg.color_correction_method)
        if is_color and color_fn is not None and cfg.color_correction_method != ColorCorrectionMethod.NONE:
            if cfg.color_correction_method == ColorCorrectionMethod.SEA_WHITE_BALANCE:
                img = color_fn(
                    img,
                    corner_fraction=cfg.wb_corner_fraction,
                    gray_target=cfg.wb_gray_target,
                    gain_clamp=cfg.wb_gain_clamp,
                    apply_smoothing=cfg.wb_apply_smoothing,
                )
            elif cfg.color_correction_method == ColorCorrectionMethod.SEA_GRAY_WORLD:
                img = color_fn(
                    img,
                    corner_fraction=cfg.wb_corner_fraction,
                    gray_target=cfg.wb_gray_target,
                    gain_clamp=cfg.wb_gain_clamp,
                )
            logger.debug("Color correction applied: %s", cfg.color_correction_method.value)

        # Store grayscale for processing
        if is_color:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # ---------------------------------------------------------------
        # Stage 1: Sea-sky-line detection (optional)
        # ---------------------------------------------------------------
        ssl_row = None
        if cfg.detect_ssl:
            detector = SSL_DETECTORS.get(cfg.ssl_metric, detect_ssl_row_edge)
            try:
                ssl_row = detector(gray, cfg.ssl_smooth_sigma)
                logger.debug("SSL detected at row %d / %d", ssl_row, gray.shape[0])
            except Exception:
                ssl_row = gray.shape[0] // 2

        # ---------------------------------------------------------------
        # Stage 2: Sea clutter suppression
        # ---------------------------------------------------------------
        sea_clutter_fn = SEA_CLUTTER_FN.get(cfg.sea_clutter_method)
        if sea_clutter_fn is not None and cfg.sea_clutter_method != SeaClutterMethod.NONE:
            # Map config to function kwargs
            if cfg.sea_clutter_method == SeaClutterMethod.MEDIAN:
                gray = sea_clutter_fn(gray, kernel=cfg.median_kernel)
            elif cfg.sea_clutter_method == SeaClutterMethod.BILATERAL:
                gray = sea_clutter_fn(
                    gray, d=cfg.bilateral_d,
                    sigma_color=cfg.bilateral_sigma_color,
                    sigma_space=cfg.bilateral_sigma_space,
                )
            elif cfg.sea_clutter_method in (
                SeaClutterMethod.TOPHAT,
                SeaClutterMethod.BOTTOMHAT,
                SeaClutterMethod.TOPHAT_BOTTOMHAT,
            ):
                kernel = cfg.tophat_kernel if cfg.sea_clutter_method in (
                    SeaClutterMethod.TOPHAT, SeaClutterMethod.TOPHAT_BOTTOMHAT
                ) else cfg.bottomhat_kernel
                gray = sea_clutter_fn(gray, kernel=kernel)
            elif cfg.sea_clutter_method == SeaClutterMethod.GAUSSIAN_DIFF:
                gray = sea_clutter_fn(gray, sigma1=cfg.gauss_diff_sigma1, sigma2=cfg.gauss_diff_sigma2)
            elif cfg.sea_clutter_method == SeaClutterMethod.BUTTERWORTH_BPF:
                gray = sea_clutter_fn(
                    gray,
                    cutoff_low=cfg.butterworth_cutoff_low,
                    cutoff_high=cfg.butterworth_cutoff_high,
                    order=cfg.butterworth_order,
                )
            elif cfg.sea_clutter_method == SeaClutterMethod.ANISOTROPIC:
                gray = sea_clutter_fn(gray, iterations=cfg.anisotropic_iterations, k=cfg.anisotropic_k)
            elif cfg.sea_clutter_method == SeaClutterMethod.RPCA:
                gray = sea_clutter_fn(
                    gray,
                    method=cfg.rpca_method,
                    rank=cfg.rpca_rank,
                    max_iter=cfg.rpca_max_iter,
                    tol=cfg.rpca_tol,
                )
            else:
                gray = sea_clutter_fn(gray)
            logger.debug("Sea clutter applied: %s", cfg.sea_clutter_method.value)

        # ---------------------------------------------------------------
        # Stage 3: Contrast enhancement
        # ---------------------------------------------------------------
        enhance_fn = ENHANCEMENT_FN.get(cfg.enhancement_method)
        if enhance_fn is not None and cfg.enhancement_method != EnhancementMethod.NONE:
            if cfg.enhancement_method == EnhancementMethod.CLAHE:
                gray = enhance_fn(gray, clip_limit=cfg.clahe_clip_limit, tile_grid=cfg.clahe_tile_grid)
            elif cfg.enhancement_method == EnhancementMethod.CLAHE_DETAIL:
                gray = enhance_fn(
                    gray,
                    clip_limit=cfg.clahe_clip_limit,
                    tile_grid=cfg.clahe_tile_grid,
                    unsharp_strength=cfg.unsharp_strength,
                    unsharp_kernel=cfg.unsharp_kernel,
                )
            elif cfg.enhancement_method in (EnhancementMethod.RETINEX_SSR, EnhancementMethod.RETINEX_MSR):
                gray = enhance_fn(
                    gray,
                    scales=cfg.retinex_scales if cfg.enhancement_method == EnhancementMethod.RETINEX_MSR else None,
                    gain=cfg.retinex_gain,
                    offset=cfg.retinex_offset,
                )
            elif cfg.enhancement_method == EnhancementMethod.GAMMA:
                gray = enhance_fn(gray, gamma=cfg.gamma_value if not cfg.gamma_auto else None, auto=cfg.gamma_auto)
            elif cfg.enhancement_method == EnhancementMethod.HOMOMORPHIC:
                gray = enhance_fn(
                    gray,
                    cutoff=cfg.homomorphic_cutoff,
                    high=cfg.homomorphic_high,
                    low=cfg.homomorphic_low,
                )
            elif cfg.enhancement_method == EnhancementMethod.UNMASK:
                gray = enhance_fn(gray, strength=cfg.unsharp_strength, kernel=cfg.unsharp_kernel)
            elif cfg.enhancement_method == EnhancementMethod.ADAPTIVE_EQ:
                gray = enhance_fn(gray)
            else:
                gray = enhance_fn(gray)
            logger.debug("Enhancement applied: %s", cfg.enhancement_method.value)

        # ---------------------------------------------------------------
        # Stage 4: Edge enhancement
        # ---------------------------------------------------------------
        if cfg.edge_enhance and cfg.enhancement_method not in (
            EnhancementMethod.CLAHE_DETAIL,  # already has unsharp
            EnhancementMethod.UNMASK,         # already unsharp
            EnhancementMethod.NONE,
        ):
            gray = enhance_unmask(gray, strength=cfg.unsharp_strength, kernel=cfg.unsharp_kernel)

        # ---------------------------------------------------------------
        # Stage 5: Resize if requested
        # ---------------------------------------------------------------
        if cfg.target_size is not None:
            gray = cv2.resize(gray, cfg.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert back to 3-channel if input was color (maintains interface)
        if is_color:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray

    def visualize_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run each pipeline stage independently and return all intermediate results.

        Useful for wandb logging to compare stage effects.

        Args:
            image: Input image.

        Returns:
            Dict mapping stage names to output images.
        """
        cfg = self.config
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        results = {"input": gray}

        # Clutter suppression
        if cfg.sea_clutter_method != SeaClutterMethod.NONE:
            stage = self.process_stage(gray, "sea_clutter")
            results["after_clutter"] = stage

        # Enhancement
        if cfg.enhancement_method != EnhancementMethod.NONE:
            base = results.get("after_clutter", gray)
            stage = self.process_stage(base, "enhancement")
            results["after_enhance"] = stage

        # Edge enhance
        if cfg.edge_enhance:
            base = results.get("after_enhance", results.get("after_clutter", gray))
            stage = enhance_unmask(base, strength=cfg.unsharp_strength, kernel=cfg.unsharp_kernel)
            results["after_edge"] = stage

        # Process final result on grayscale for consistent output format
        if image.ndim == 3:
            results["final"] = self.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            results["final"] = self.process(image)
        return results

    def process_stage(self, image: np.ndarray, stage: str) -> np.ndarray:
        """Run a single pipeline stage with correct parameters for the active method."""
        cfg = self.config
        if stage == "sea_clutter":
            fn = SEA_CLUTTER_FN.get(cfg.sea_clutter_method)
            if fn is None:
                return image
            method = cfg.sea_clutter_method
            if method == SeaClutterMethod.MEDIAN:
                return fn(image, kernel=cfg.median_kernel)
            elif method == SeaClutterMethod.BILATERAL:
                return fn(image, d=cfg.bilateral_d,
                          sigma_color=cfg.bilateral_sigma_color,
                          sigma_space=cfg.bilateral_sigma_space)
            elif method == SeaClutterMethod.TOPHAT:
                return fn(image, kernel=cfg.tophat_kernel)
            elif method == SeaClutterMethod.BOTTOMHAT:
                return fn(image, kernel=cfg.bottomhat_kernel)
            elif method == SeaClutterMethod.TOPHAT_BOTTOMHAT:
                return fn(image, kernel=cfg.tophat_kernel)
            elif method == SeaClutterMethod.GAUSSIAN_DIFF:
                return fn(image, sigma1=cfg.gauss_diff_sigma1,
                          sigma2=cfg.gauss_diff_sigma2)
            elif method == SeaClutterMethod.ANISOTROPIC:
                return fn(image, iterations=cfg.anisotropic_iterations,
                          k=cfg.anisotropic_k)
            elif method == SeaClutterMethod.BUTTERWORTH_BPF:
                return fn(image,
                          cutoff_low=cfg.butterworth_cutoff_low,
                          cutoff_high=cfg.butterworth_cutoff_high,
                          order=cfg.butterworth_order)
            elif method == SeaClutterMethod.RPCA:
                return fn(
                    image,
                    method=cfg.rpca_method,
                    rank=cfg.rpca_rank,
                    max_iter=cfg.rpca_max_iter,
                    tol=cfg.rpca_tol,
                )
            else:
                return fn(image)
        elif stage == "enhancement":
            fn = ENHANCEMENT_FN.get(cfg.enhancement_method)
            if fn is None:
                return image
            return fn(image, clip_limit=cfg.clahe_clip_limit, tile_grid=cfg.clahe_tile_grid)
        return image


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_default_pipeline() -> PreprocessingPipeline:
    """Create a pipeline with known-good defaults for IR maritime ship detection.

    Pipeline:
      - Combined top-hat + bottom-hat for sea clutter suppression
      - CLAHE + detail enhancement for contrast
      - Light unsharp masking for edge sharpening
    """
    config = PreprocessingConfig(
        sea_clutter_method=SeaClutterMethod.TOPHAT_BOTTOMHAT,
        tophat_kernel=31,
        bottomhat_kernel=31,
        enhancement_method=EnhancementMethod.CLAHE_DETAIL,
        clahe_clip_limit=3.0,
        clahe_tile_grid=(8, 8),
        edge_enhance=True,
        unsharp_strength=0.5,
        unsharp_kernel=5,
        detect_ssl=False,
    )
    return PreprocessingPipeline(config)


def create_retinex_pipeline() -> PreprocessingPipeline:
    """Create a pipeline using Retinex MSR enhancement for low-contrast IR."""
    config = PreprocessingConfig(
        sea_clutter_method=SeaClutterMethod.TOPHAT_BOTTOMHAT,
        tophat_kernel=31,
        enhancement_method=EnhancementMethod.RETINEX_MSR,
        retinex_scales=[15, 80, 250],
        retinex_gain=1.0,
        edge_enhance=True,
        unsharp_strength=0.3,
    )
    return PreprocessingPipeline(config)


def create_light_pipeline() -> PreprocessingPipeline:
    """Lightweight pipeline for real-time inference (minimal compute)."""
    config = PreprocessingConfig(
        sea_clutter_method=SeaClutterMethod.TOPHAT,
        tophat_kernel=21,
        enhancement_method=EnhancementMethod.CLAHE,
        clahe_clip_limit=2.0,
        edge_enhance=False,
    )
    return PreprocessingPipeline(config)


# ---------------------------------------------------------------------------
# Dataset-compatible wrapper for YOLO training
# ---------------------------------------------------------------------------

class PreprocessingTransform:
    """Callable transform compatible with ultralytics YOLO dataset augmentation.

    Can be inserted into the YOLO training pipeline by overriding the dataset's
    `load_image` method or by applying as a mosaic-aware transform.

    Usage:
        transform = PreprocessingTransform()
        # Inside custom dataset:
        img = transform(image)
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.pipeline = PreprocessingPipeline(config)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing, preserving dtype and shape."""
        out = self.pipeline.process(image)
        return out


# ---------------------------------------------------------------------------
# Dual pipeline for fusion (runs spatial + freq separately, stacks as 3-ch)
# ---------------------------------------------------------------------------

class PreprocessingPipelineDual:
    """Runs TWO preprocessing pipelines on the same image and stacks results.

    Pipeline 1 (spatial):   AWB → tophat → CLAHE  → unsharp
    Pipeline 2 (frequency): AWB → butterworth BPF → CLAHE → unsharp

    Output is a 3-channel BGR image that, after Ultralytics' BGR→RGB conversion
    (``img[:, :, ::-1]``), becomes:

        Channel 0 (R) = spatial-domain filtered      (spatial_pipeline output)
        Channel 1 (G) = frequency-domain filtered     (freq_pipeline output)
        Channel 2 (B) = original grayscale            (for residual connection)

    This is designed for input to a learnable fusion network (SpatialFrequencyFusion)
    which expects its 3 input channels in the order [spatial, freq, original].
    
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        freq_method: SeaClutterMethod = SeaClutterMethod.BUTTERWORTH_BPF,
    ):
        """Initialise dual-stream pipeline.
        
        Args:
            config: Ignored (kept for API compatibility with ``get_preprocessing_pipeline``).
            freq_method: Sea clutter method for the frequency-domain pipeline.
                        Defaults to BUTTERWORTH_BPF.
        """
        self.spatial_pipeline = PreprocessingPipeline(
            PreprocessingConfig(
                color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
                wb_corner_fraction=0.06,
                wb_gray_target=128.0,
                sea_clutter_method=SeaClutterMethod.TOPHAT,
                tophat_kernel=31,
                enhancement_method=EnhancementMethod.CLAHE,
                clahe_clip_limit=3.0,
                edge_enhance=True,
                unsharp_strength=0.5,
            )
        )
        freq_config = PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=freq_method,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        )
        # Set wavelet params if applicable
        if freq_method == SeaClutterMethod.WAVELET:
            freq_config.wavelet_name = "db4"
        elif freq_method == SeaClutterMethod.BUTTERWORTH_BPF:
            freq_config.butterworth_cutoff_low = 8.0
            freq_config.butterworth_cutoff_high = 80.0
        self.freq_pipeline = PreprocessingPipeline(freq_config)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run dual pipelines and stack as 3-channel BGR.

        Args:
            image: BGR image (H, W, 3) uint8.

        Returns:
            3-channel BGR uint8. After Ultralytics' BGR→RGB (img[:,:,::-1]):
                channel 0 = spatial, channel 1 = freq, channel 2 = original_gray.

            Current BGR layout:
                B = original_gray, G = freq_gray, R = spatial_gray
        """
        # Original grayscale
        orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Spatial pipeline
        spatial_out = self.spatial_pipeline(image)           # (H, W, 3) BGR
        spatial_gray = cv2.cvtColor(spatial_out, cv2.COLOR_BGR2GRAY)

        # Frequency pipeline
        freq_out = self.freq_pipeline(image)                 # (H, W, 3) BGR
        freq_gray = cv2.cvtColor(freq_out, cv2.COLOR_BGR2GRAY)

        # Stack so after BGR→RGB (channel reversal) we get [spatial, freq, orig]:
        #   BGR: B=orig, G=freq, R=spatial
        #   RGB (after ::-1): [spatial, freq, orig]
        fused = np.stack([orig_gray, freq_gray, spatial_gray], axis=-1)  # (H, W, 3)
        return fused

    def visualize_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Return all stage images as a flat dict (same format as
        :meth:`PreprocessingPipeline.visualize_stages`).

        Keys are prefixed with ``spatial_`` / ``freq_`` / ``original_``
        so callers can iterate the dict uniformly.
        """
        orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        spatial_stages = self.spatial_pipeline.visualize_stages(image)
        freq_stages = self.freq_pipeline.visualize_stages(image)

        result: dict[str, np.ndarray] = {"original_gray": orig_gray}
        for key, val in spatial_stages.items():
            result[f"spatial_{key}"] = val
        for key, val in freq_stages.items():
            result[f"freq_{key}"] = val
        result["spatial_final"] = cv2.cvtColor(
            spatial_stages.get("final", image), cv2.COLOR_BGR2GRAY
        )
        result["freq_final"] = cv2.cvtColor(
            freq_stages.get("final", image), cv2.COLOR_BGR2GRAY
        )
        return result


# ---------------------------------------------------------------------------
# AVAILABLE_PREPROCESS — preset descriptions for CLI / wandb logging
# ---------------------------------------------------------------------------

AVAILABLE_PREPROCESS = {
    "none": "No preprocessing (raw input)",
    "tophat_clahe": (
        "Top-hat morphological sea clutter suppression (kernel=31) → "
        "CLAHE local contrast enhancement (clip=3.0, tile=8x8) → "
        "Unsharp masking edge enhancement (strength=0.5)"
    ),
    "tophat_detail": (
        "Top-hat+bottom-hat combined clutter suppression (kernel=31) → "
        "CLAHE + detail enhancement → Unsharp masking"
    ),
    "dog_retinex": (
        "Difference-of-Gaussians bandpass filter (σ1=1.0, σ2=3.0) → "
        "Multi-Scale Retinex enhancement (scales=[15,80,250])"
    ),
    "retinex_gamma": (
        "Auto gamma correction (mean-based adaptive gamma) → "
        "Multi-Scale Retinex (scales=[15,80,250]) → Unsharp masking"
    ),
    "median_clahe": (
        "Median filter (kernel=5) for speckle noise removal → "
        "CLAHE (clip=2.0) → Light unsharp masking"
    ),
    "bilateral_gamma": (
        "Bilateral filter (d=7, σ_color=40, σ_space=40) edge-preserving smooth → "
        "Auto-gamma correction → Adaptive equalization"
    ),
    "ssl_adaptive": (
        "Sea-sky-line detection → Regional division processing: "
        "top-hat on sea region + mild CLAHE on sky region → "
        "Adaptive contrast stretch per region"
    ),
    # ---- New presets with color correction + frequency/spatial domain ----
    "awb_tophat_clahe": (
        "Auto white balance (sea-corner gray reference) → "
        "Top-hat sea clutter suppression (kernel=31) → "
        "CLAHE contrast enhancement (clip=3.0) → "
        "Unsharp masking edge enhancement"
    ),
    "awb_butterworth_clahe": (
        "Auto white balance (sea-corner gray reference) → "
        "Butterworth band-pass frequency filtering (low=8, high=80) → "
        "CLAHE contrast enhancement (clip=3.0) → "
        "Unsharp masking edge enhancement"
    ),
    "awb_butterworth_retinex": (
        "AWB (sea-corner) → Butterworth band-pass (low=8, high=80) → "
        "Multi-Scale Retinex (scales=[15,80,250]) → Unsharp masking"
    ),
    "awb_dual_fusion": (
        "AWB (sea-corner) → Dual-stream pipeline: spatial (tophat+CLAHE) AND "
        "frequency (Butterworth BPF+CLAHE) → stacked as 3-channel "
        "[spatial, freq, original_gray] input for learnable fusion network"
    ),
    "awb_wavelet_clahe": (
        "Auto white balance (sea-corner gray reference) → "
        "Wavelet-domain clutter suppression (db4, LL*0.3, LH/HL*1.5, HH*0.3) → "
        "CLAHE contrast enhancement (clip=3.0) → "
        "Unsharp masking edge enhancement"
    ),
    "awb_wavelet_dual_fusion": (
        "AWB (sea-corner) → Dual-stream pipeline: spatial (tophat+CLAHE) AND "
        "frequency (wavelet domain+CLAHE) → stacked as 3-channel "
        "[spatial, freq, original_gray] input for learnable fusion network "
        "(wavelet replaces Butterworth as frequency method)"
    ),
    "awb_rpca_dual_fusion": (
        "AWB (sea-corner) → Dual-stream pipeline: spatial (tophat+CLAHE) AND "
        "frequency (RPCA low-rank background subtraction+CLAHE) → stacked as 3-channel "
        "[spatial, rpca_foreground, original_gray] input for learnable fusion network "
        "(RPCA replaces Butterworth as background-modeling method)"
    ),
}


def get_preprocessing_pipeline(method: str):
    """Return a (PreprocessingPipeline | PreprocessingPipelineDual, description_string, params_dict).

    Args:
        method: One of ``AVAILABLE_PREPROCESS`` keys.

    Returns:
        Tuple of (pipeline, description, params_dict).
    """
    # Handle dual fusion presets specially
    if method == "awb_dual_fusion":
        pipeline = PreprocessingPipelineDual()
        description = AVAILABLE_PREPROCESS.get(method, "Dual-stream fusion pipeline")
        params = {
            "spatial": "tophat(k=31)+CLAHE(clip=3.0)+unsharp",
            "frequency": "butterworth(low=8,high=80)+CLAHE(clip=3.0)+unsharp",
            "stacking": "[spatial, freq, original_gray] → 3-ch input",
        }
        return pipeline, description, params
    if method == "awb_wavelet_dual_fusion":
        pipeline = PreprocessingPipelineDual(freq_method=SeaClutterMethod.WAVELET)
        description = AVAILABLE_PREPROCESS.get(method, "Wavelet-based dual-stream fusion pipeline")
        params = {
            "spatial": "tophat(k=31)+CLAHE(clip=3.0)+unsharp",
            "frequency": "wavelet(db4,LL*0.3,LH/HL*1.5,HH*0.3)+CLAHE(clip=3.0)+unsharp",
            "stacking": "[spatial, freq, original_gray] → 3-ch input",
        }
        return pipeline, description, params
    if method == "awb_rpca_dual_fusion":
        pipeline = PreprocessingPipelineDual(freq_method=SeaClutterMethod.RPCA)
        description = AVAILABLE_PREPROCESS.get(method, "RPCA-based dual-stream fusion pipeline")
        params = {
            "spatial": "tophat(k=31)+CLAHE(clip=3.0)+unsharp",
            "frequency": "rpca_truncated_svd(rank=10)+CLAHE(clip=3.0)+unsharp",
            "stacking": "[spatial, rpca_foreground, original_gray] → 3-ch input",
        }
        return pipeline, description, params

    config = _PRESET_CONFIGS.get(method)
    if config is None:
        # Use NONE config
        config = PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.NONE,
            enhancement_method=EnhancementMethod.NONE,
            edge_enhance=False,
        )
    pipeline = PreprocessingPipeline(config)
    description = AVAILABLE_PREPROCESS.get(method, AVAILABLE_PREPROCESS["none"])

    # Extract display params from config
    params = {}
    if method in ("tophat_clahe", "tophat_detail", "ssl_adaptive"):
        params["tophat_kernel"] = config.tophat_kernel
    if method == "tophat_clahe":
        params["clahe_clip"] = config.clahe_clip_limit
    if method == "dog_retinex":
        params["dog_sigma"] = f"({config.gauss_diff_sigma1}, {config.gauss_diff_sigma2})"
        params["retinex_scales"] = str(config.retinex_scales)
    if method == "median_clahe":
        params["median_kernel"] = config.median_kernel

    return pipeline, description, params


# Lookup table for presets
_PRESET_CONFIGS: dict[str, PreprocessingConfig] = {}


def _register_presets():
    """Build preset config lookup."""
    global _PRESET_CONFIGS
    _PRESET_CONFIGS = {
        "none": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.NONE,
            enhancement_method=EnhancementMethod.NONE,
            edge_enhance=False,
        ),
        "tophat_clahe": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.TOPHAT,
            tophat_kernel=31,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            edge_enhance=True,
            unsharp_strength=0.5,
        ),
        "tophat_detail": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.TOPHAT_BOTTOMHAT,
            tophat_kernel=31,
            bottomhat_kernel=31,
            enhancement_method=EnhancementMethod.CLAHE_DETAIL,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            edge_enhance=True,
            unsharp_strength=0.5,
            unsharp_kernel=5,
        ),
        "dog_retinex": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.GAUSSIAN_DIFF,
            gauss_diff_sigma1=1.0,
            gauss_diff_sigma2=3.0,
            enhancement_method=EnhancementMethod.RETINEX_MSR,
            retinex_scales=[15, 80, 250],
            edge_enhance=False,
        ),
        "retinex_gamma": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.NONE,
            enhancement_method=EnhancementMethod.RETINEX_MSR,
            retinex_scales=[15, 80, 250],
            edge_enhance=True,
            unsharp_strength=0.3,
        ),
        "median_clahe": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.MEDIAN,
            median_kernel=5,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=2.0,
            edge_enhance=True,
            unsharp_strength=0.3,
        ),
        "bilateral_gamma": PreprocessingConfig(
            sea_clutter_method=SeaClutterMethod.BILATERAL,
            bilateral_d=7,
            bilateral_sigma_color=40.0,
            bilateral_sigma_space=40.0,
            enhancement_method=EnhancementMethod.ADAPTIVE_EQ,
            edge_enhance=False,
        ),
        "ssl_adaptive": PreprocessingConfig(
            detect_ssl=True,
            ssl_metric=SSLMetric.ROW_VARIANCE,
            sea_clutter_method=SeaClutterMethod.TOPHAT,
            tophat_kernel=25,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=2.5,
            edge_enhance=True,
            unsharp_strength=0.4,
        ),
        # ---- New presets ----
        "awb_tophat_clahe": PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.TOPHAT,
            tophat_kernel=31,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        ),
        "awb_butterworth_clahe": PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.BUTTERWORTH_BPF,
            butterworth_cutoff_low=8.0,
            butterworth_cutoff_high=80.0,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        ),
        "awb_butterworth_retinex": PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.BUTTERWORTH_BPF,
            butterworth_cutoff_low=8.0,
            butterworth_cutoff_high=80.0,
            enhancement_method=EnhancementMethod.RETINEX_MSR,
            retinex_scales=[15, 80, 250],
            edge_enhance=False,
        ),
        # Wavelet-based single pipeline
        "awb_wavelet_clahe": PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.WAVELET,
            wavelet_name="db4",
            wavelet_level=1,
            wavelet_scale_ll=0.3,
            wavelet_scale_lh=1.5,
            wavelet_scale_hl=1.5,
            wavelet_scale_hh=0.3,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        ),
        # Dual-stream fusion presets (virtual — returns PreprocessingPipelineDual)
        "awb_dual_fusion": "DUAL_PLACEHOLDER",
        "awb_wavelet_dual_fusion": "DUAL_PLACEHOLDER",
    }


# Register presets on import
_register_presets()


# ===========================================================================
# Quick test / visualization
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    pipe = create_default_pipeline()
    config = pipe.config
    print(f"Pipeline config:")
    print(f"  Sea clutter:  {config.sea_clutter_method.value}")
    print(f"  Enhancement:  {config.enhancement_method.value}")
    print(f"  Edge enhance: {config.edge_enhance}")
    print(f"  Tophat kernel: {config.tophat_kernel}")
    print(f"  CLAHE clip:   {config.clahe_clip_limit}")
    print("\nPreprocessing module loaded successfully.")
    print("Configure via PreprocessingConfig or use create_default_pipeline().")
