"""
C2EMA — C2f block with EMA (Efficient Multi-Scale Attention).

Replaces standard Neck C3k2/C2f blocks with an enhanced version
that integrates EMA attention for cross-spatial learning.

Reference:
    "Efficient Multi-Scale Attention Module with Cross-Spatial Learning"
    (Ouyang et al., IEEE ICASSP 2023)

Usage:
    from c2ema import replace_neck_with_c2ema
    model = YOLO("yolo11l.pt")
    replace_neck_with_c2ema(model)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# EMA — Efficient Multi-Scale Attention Module
# ===================================================================

class EMA(nn.Module):
    """Efficient Multi-Scale Attention.

    Architecture (per group):
        1. Parallel 1×1 and 3×3 conv branches
        2. Cross-dimension interaction via 2D GAP + softmax
        3. Shared FC for channel attention
        4. Cross-spatial aggregation with 3×3 depthwise conv

    Args:
        channels: Number of input channels.
        groups: Number of parallel groups (default 4).
        ratio: Reduction ratio for the shared FC (default 16).
    """

    def __init__(self, channels: int, groups: int = 4, ratio: int = 16):
        super().__init__()
        assert channels % groups == 0, f"channels ({channels}) must be divisible by groups ({groups})"
        self.groups = groups
        self.group_ch = channels // groups

        # 1×1 branch (per group, implemented with grouped conv)
        self.conv1x1 = nn.Conv2d(
            channels, channels, kernel_size=1, groups=groups, bias=False,
        )
        # 3×3 branch (per group)
        self.conv3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=groups, bias=False,
        )

        # Shared channel attention: C→C/ratio→C
        reduced = max(8, channels // ratio)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)

        # Cross-spatial aggregation (depthwise per group)
        self.agg = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,  # depthwise
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, C, H, W).
        """
        # Parallel 1×1 and 3×3 convolutions
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)

        # Combine
        y = x1 + x3  # (B, C, H, W)

        # Cross-dimension channel attention
        gap = y.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        attn = self.fc2(F.silu(self.fc1(gap)))  # (B, C, 1, 1)
        attn = torch.sigmoid(attn)
        y = y * attn

        # Cross-spatial aggregation (depthwise 3×3)
        y = self.agg(y)
        y = self.bn1(y)

        # Residual connection
        out = y + x
        out = self.bn2(out)
        return out


# ===================================================================
# Bottleneck_EMA — Bottleneck with EMA integration
# ===================================================================

class BottleneckEMA(nn.Module):
    """Bottleneck block with EMA attention.

    Conv1×1 → EMA → Conv3×3 → shortcut

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to use shortcut connection.
        g: Groups for the 3×3 conv.
        e: Expansion ratio.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules import Conv

        self.cv1 = Conv(c1, c_, 1, 1)
        self.ema = EMA(c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional shortcut."""
        y = self.cv1(x)
        y = self.ema(y)
        y = self.cv2(y)
        return (y + x) if self.add else y


# ===================================================================
# C2EMA — C2f-like block with EMA Bottlenecks
# ===================================================================

class C2EMA(nn.Module):
    """CSP Bottleneck with EMA-enhanced convolutions.

    Follows the same interface as C2f / C3k2 for drop-in replacement
    in the Neck.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of BottleneckEMA blocks.
        shortcut: Whether to use shortcut connections.
        g: Groups for convolutions.
        e: Expansion ratio.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        from ultralytics.nn.modules import Conv

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BottleneckEMA(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CSP-style forward: split → EMA bottlenecks → concat → project."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===================================================================
# Replace neck C3k2 blocks with C2EMA
# ===================================================================

def replace_neck_with_c2ema(
    model: nn.Module,
    target_layers: Optional[list[int]] = None,
) -> int:
    """Replace C3k2/C2f blocks in the Neck with C2EMA.

    By default replaces all Neck C3k2 blocks (after C2PSA/SPPF).

    Args:
        model: A YOLO or DetectionModel object.
               Unwrapping: YOLO.model.model → DetectionModel.model → Sequential
        target_layers: Optional list of layer indices to replace.
                       If None, replaces all C3k2 after backbone end.

    Returns:
        Number of blocks replaced.
    """
    from ultralytics.nn.modules import C3k2, C2f

    # Unwrap model
    if hasattr(model, "model") and hasattr(model.model, "model"):
        detection_model = model.model
        model_seq = detection_model.model
    elif hasattr(model, "model"):
        detection_model = model
        model_seq = detection_model.model
    else:
        detection_model = None
        model_seq = model

    from attention_modules import _find_backbone_end
    backbone_end = _find_backbone_end(model_seq)

    replaced = 0
    for i, layer in enumerate(model_seq):
        if target_layers is not None and i not in target_layers:
            continue
        if target_layers is None and i <= backbone_end:
            continue  # only replace Neck (post-backbone) blocks
        if isinstance(layer, (C3k2, C2f)):
            # Build C2EMA with the same parameters
            c1 = _infer_c1(model_seq, i, layer)
            c2 = getattr(layer, "c", None) or _get_out_channels(layer)
            # Actually we need c2 (output channels), not hidden channels
            c2 = _get_output_channels(layer)
            # Get the original init params from the module
            c2ema = C2EMA(
                c1=c1,
                c2=c2,
                n=len(layer.m) if hasattr(layer, "m") else 1,
                shortcut=getattr(layer, "add", False) if hasattr(layer, "add") else False,
                g=getattr(layer, "g", 1),
                e=0.5,
            )
            model_seq[i] = c2ema
            c2ema.i = i
            c2ema.f = getattr(layer, "f", -1)
            replaced += 1
            logger.info("Replaced layer %d (%s) → C2EMA(c1=%d, c2=%d)", i, type(layer).__name__, c1, c2)

    if replaced:
        logger.info("C2EMA: Replaced %d blocks in Neck", replaced)
    else:
        logger.warning("C2EMA: No eligible blocks found to replace")

    return replaced


def _infer_c1(model_seq: nn.Sequential, idx: int, layer: nn.Module) -> int:
    """Infer input channels of a layer from its cv1 conv."""
    from ultralytics.nn.modules import Conv as UltralyticsConv

    for attr in ("cv1", "cv2"):
        cv = getattr(layer, attr, None)
        if cv is not None:
            if isinstance(cv, UltralyticsConv):
                return cv.conv.in_channels
            if hasattr(cv, "in_channels"):
                return cv.in_channels
    return 512


def _get_output_channels(layer: nn.Module) -> int:
    """Get the actual output channels (not hidden) of a C2f/C3k2."""
    from ultralytics.nn.modules import Conv as UltralyticsConv

    if hasattr(layer, "cv2"):
        cv = layer.cv2
        if isinstance(cv, UltralyticsConv):
            return cv.conv.out_channels
        if hasattr(cv, "out_channels"):
            return cv.out_channels
    if hasattr(layer, "c2"):
        return layer.c2
    return 512
