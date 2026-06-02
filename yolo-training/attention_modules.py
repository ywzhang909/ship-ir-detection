"""
Lightweight attention modules for YOLO backbone insertion.

Implements:
- ECA (Efficient Channel Attention) — 1D conv, no dimension reduction
- CA (Coordinate Attention) — encodes position + channel info via 1D pooling

References:
    ECA-Net: https://arxiv.org/abs/1910.03151
    Coordinate Attention: https://arxiv.org/abs/2103.02907
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Efficient Channel Attention (ECA)
# ===================================================================

class ECA(nn.Module):
    """Efficient Channel Attention.

    Uses 1D convolution over the channel dimension after GAP.
    Kernel size is adaptively determined by channel count:
        k = max(3, (log2(C) + 1) // 2 * 2 + 1)   (odd kernel)

    Args:
        channels: Number of input channels.
        gamma: Scaling factor for kernel size (default 2).
        b: Bias term for kernel size (default 1).
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = max(3, |log2(C)/gamma + b/gamma|_odd)
        t = int(abs(torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma)
        k = t if t % 2 else t + 1  # ensure odd
        k = max(3, k)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)  # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = self.sigmoid(y).view(b, c, 1, 1) # (B, C, 1, 1)
        return x * y


# ===================================================================
# Coordinate Attention (CA)
# ===================================================================

class CoordAtt(nn.Module):
    """Coordinate Attention.

    Encodes both channel and position information by decomposing
    2D global pooling into two 1D encoding steps (X and Y directions).

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio for bottleneck (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(4, channels // reduction)

        # Two 1D pooling ops
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)

        # Separate convs for h and w after split
        self.conv_h = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # (B, C, H, 1) and (B, C, 1, W)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concat → conv1 → BN → SiLU
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.silu(y)

        # Split back into h and w branches
        x_h, x_w = torch.split(y, [h, w], dim=2)  # (B, C, H, 1), (B, C, W, 1)

        # Separate convs → sigmoid
        a_h = self.sigmoid(self.conv_h(x_h))  # (B, C, H, 1)
        a_w = self.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)

        out = x * a_h * a_w
        return out


# ===================================================================
# Frequency Enhanced Channel Attention (FECA)
# ===================================================================

class FECA(nn.Module):
    """Frequency Enhanced Channel Attention.

    Applies FFT to extract frequency-domain features, computes channel
    attention weights from the magnitude spectrum, and gates the original
    features. Helps the model focus on frequency patterns characteristic
    of IR imagery (thermal signatures, periodic clutter).

    Args:
        channels: Number of input channels.
        gamma: Scaling factor for adaptive kernel (default 2).
        b: Bias for kernel size (default 1).
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size (same heuristic as ECA)
        t = int(abs(torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma)
        k = t if t % 2 else t + 1
        k = max(3, k)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Frequency channel descriptor — learns to weigh freq components
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-enhanced channel attention.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Gated tensor (B, C, H, W) with frequency-aware weighting.
        """
        # ── FFT magnitude spectrum ──
        # Apply 2D FFT, shift DC to center, get log magnitude
        x_fft = torch.fft.fft2(x.float())
        x_fft_shifted = torch.fft.fftshift(x_fft)
        mag = torch.abs(x_fft_shifted)
        log_mag = torch.log(mag + 1e-8)  # (B, C, H, W)

        # ── Channel-wise frequency descriptor ──
        # Global average across spatial dimensions of the log-magnitude
        y = self.avg_pool(log_mag).flatten(2)  # (B, C, 1) → (B, 1, C) after view
        y = y.view(x.size(0), 1, x.size(1))   # (B, 1, C)

        # 1D convolution across channels
        y = self.conv(y)                        # (B, 1, C)
        y = self.sigmoid(y).view(x.size(0), x.size(1), 1, 1)  # (B, C, 1, 1)

        # Gate the ORIGINAL spatial features (not the spectral ones)
        return x * y.to(x.dtype)


# ===================================================================
# Insertion helpers
# ===================================================================

def insert_attention_into_backbone(
    model: "torch.nn.Module",
    attn_type: Literal["eca", "ca", "feca"],
    reduction: int = 16,
) -> int:
    """Insert ECA or CA attention modules after each backbone stage.

    Inserts attention after each C3k2 block in the backbone (the
    main feature extraction stages). Modifies the model in-place.
    All neck/head layers are preserved unchanged.

    Args:
        model: A YOLO or DetectionModel object.
            - YOLO: model.model is DetectionModel, model.model.model is Sequential
            - DetectionModel: model.model is Sequential
        attn_type: 'eca' or 'ca'.
        reduction: Reduction ratio (CA only).

    Returns:
        Number of attention modules inserted.
    """
    from ultralytics.nn.modules import C3k2

    # Unwrap YOLO → DetectionModel → Sequential
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        detection_model = model.model
        model_seq = detection_model.model
    elif hasattr(model, 'model'):
        detection_model = model
        model_seq = detection_model.model
    else:
        detection_model = None
        model_seq = model

    backbone_end_idx = _find_backbone_end(model_seq)
    inserted = 0

    # Build new layer list: keep backbone (with inserted attn) + neck/head unchanged
    new_layers = []
    for i in range(len(model_seq)):
        layer = model_seq[i]
        new_layers.append(layer)

        # Insert attention after backbone C3k2 blocks only
        if i <= backbone_end_idx and isinstance(layer, C3k2):
            c_out = _get_c3k2_out_channels(layer)
            if attn_type == "eca":
                attn = ECA(c_out)
            elif attn_type == "feca":
                attn = FECA(c_out)
            else:
                attn = CoordAtt(c_out, reduction=reduction)
            new_layers.append(attn)
            inserted += 1

    new_seq = nn.Sequential(*new_layers)

    # Fix indices and 'f' attributes — Ultralytics _predict_once uses
    # m.f (from indices) to route features between layers
    for i, layer in enumerate(new_seq):
        layer.i = i
        if not hasattr(layer, 'f') or layer.f is None:
            layer.f = -1  # default: from previous layer
        if isinstance(layer, (ECA, CoordAtt)):
            layer.f = -1

    if detection_model is not None:
        detection_model.model = new_seq
    else:
        model = new_seq  # type: ignore[assignment]

    # Build old→new index mapping: each inserted CA shifts subsequent layers
    def _remap(ref: int) -> int:
        """Map an original layer index to its new position."""
        if ref < 0:
            return ref
        offset = sum(
            1 for j in range(ref)  # 0..ref-1
            if j <= backbone_end_idx and isinstance(model_seq[j], C3k2)
        )
        return ref + offset

    # Fix f indices in neck/head layers
    for i, layer in enumerate(new_seq):
        if isinstance(layer, (ECA, CoordAtt)):
            continue
        orig_f = getattr(layer, 'f', -1)
        if isinstance(orig_f, int):
            new_f = _remap(orig_f)
            layer.f = -1 if new_f == i else new_f
        elif isinstance(orig_f, list):
            layer.f = [_remap(ref) for ref in orig_f]

    # Fix self.save (indices whose outputs are cached for skip connections)
    if detection_model is not None:
        detection_model.save = {_remap(idx) for idx in list(detection_model.save)}

    logger.info(
        "Inserted %d %s attention modules into backbone",
        inserted, attn_type.upper(),
    )
    return inserted


def _get_c3k2_out_channels(layer: nn.Module) -> int:
    """Extract output channels from a C3k2 module."""
    from ultralytics.nn.modules import Conv as UltralyticsConv

    # C2f/C3k2: cv2 is the output projection Conv (Ultralytics wrapper)
    if hasattr(layer, 'cv2'):
        cv2 = layer.cv2
        if isinstance(cv2, UltralyticsConv):
            return cv2.conv.out_channels
        if hasattr(cv2, 'out_channels'):
            return cv2.out_channels  # type: ignore[return-value]
    if hasattr(layer, 'c2'):
        return layer.c2  # type: ignore[return-value]
    if hasattr(layer, 'cv3'):
        cv3 = layer.cv3
        if isinstance(cv3, UltralyticsConv):
            return cv3.conv.out_channels
        if hasattr(cv3, 'out_channels'):
            return cv3.out_channels  # type: ignore[return-value]
    # Fallback: last nn.Conv2d in module tree (NOT Ultralytics Conv wrapper)
    for m in reversed(list(layer.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    return 512


def _find_backbone_end(model_seq: nn.Sequential) -> int:
    """Find the last index of backbone layers (before head/neck)."""
    from ultralytics.nn.modules import SPPF, C2PSA

    # Backbone ends at SPPF or C2PSA typically
    for i in range(len(model_seq) - 1, -1, -1):
        if isinstance(model_seq[i], (SPPF, C2PSA)):
            return i
    # Fallback: return ~60% of layers
    return int(len(model_seq) * 0.6)
