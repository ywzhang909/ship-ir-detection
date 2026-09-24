"""
LeConv Backbone — Decomposed Convolutional Kernel for YOLO.

Replaces the Bottleneck/C3k modules inside YOLO backbone C3k2 blocks with
LeConvBlock modules that decompose convolution into:
  1. Local detail path (standard small-kernel conv)
  2. Global context path (adaptive pooling + element-wise scaling)
  3. Element-wise fusion (local × scale)

This captures long-range spatial dependencies without large kernels or
attention, using 1/3 the parameters of StarNet at comparable accuracy.

Reference:
    "LeConv: Decomposed Convolutional Kernel for Long-Range Embedded Vision"
    (Wang et al., 2024) — https://arxiv.org/abs/2407.02488

Usage:
    from leconv_backbone import replace_backbone_with_leconv
    model = YOLO("yolo11l.pt")
    n = replace_backbone_with_leconv(model)  # replaces backbone C3k2s in-place

Model Variants (from paper):
    - LeConv-T:  0.9M params, 74.0% Top-1   — ultralightweight
    - LeConv-S:  2.9M params, 78.9% Top-1   — matches StarNet-S1 scale
    - LeConv-B:  8.7M params, 82.7% Top-1   — medium (not tested here)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C3k2

logger = logging.getLogger(__name__)


# ===================================================================
# LeConvBlock
# ===================================================================

class LeConvBlock(nn.Module):
    """Decomposed convolution block: local conv + global context scaling.

    Structure:
        x ──► Conv(k) ──► local ──┐
        x ──► Pool ──► Conv(1) ──► Sigmoid ──► scale ──► * ──► +identity
                                            element-wise ▲

    The local path captures fine-grained spatial details via a standard
    small-kernel convolution. The global path computes channel-wise
    scaling factors from an adaptively pooled context vector — this
    gives the block a long-range receptive field without large kernels.

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Enable residual connection (if c1 == c2).
        g: Groups for local conv.
        k: Kernel size for local conv.
        e: Expansion ratio (unused, kept for interface compatibility).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: int = 3,
        e: float = 0.5,
    ):
        super().__init__()

        # ── Local detail path (depthwise conv for efficiency at high res) ──
        g_dw = c1 if c1 == c2 else 1          # depthwise when c1==c2 (CSP blocks)
        self.local_conv = Conv(c1, c2, k, 1, g=g_dw)

        # ── Global context path ──
        # Adaptive pooling captures spatial context at any resolution
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Channel-wise scaling projection
        self.scale_conv = nn.Conv2d(c1, c2, 1)
        self.sigmoid = nn.Sigmoid()

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: local conv × global context scale + residual."""
        identity = x

        # Local: spatial detail extraction
        local = self.local_conv(x)

        # Global: channel-wise context scaling
        context = self.pool(x)                # (B, C, 1, 1)
        scale = self.sigmoid(self.scale_conv(context))  # (B, C2, 1, 1)

        # Fusion: element-wise scaling
        out = local * scale

        if self.add:
            out = out + identity

        return out


# ===================================================================
# C3k2LeConv — CSP block with LeConvBlock inside
# ===================================================================

class C3k2LeConv(nn.Module):
    """C3k2-compatible CSP block using LeConvBlock instead of Bottleneck/C3k.

    Preserves the exact C2f-based CSP structure that YOLO's C3k2 uses:
        cv1(x) → chunk(2) ─┬─ y0 ──────────────┐
                            └─ y1 → LeConvBlock × n ─┐
                                                    ↓
                                            cv2(cat[y0, y1, LeConvBlock×n])

    The ``c3k`` parameter is accepted (for API compatibility with C3k2)
    but ignored — LeConvBlock is always used as the internal module.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of LeConvBlocks.
        c3k: Unused (compatibility). Always uses LeConvBlock.
        e: Expansion ratio for hidden channels.
        shortcut: Whether internal LeConvBlocks use residual connections.
        g: Groups (unused, kept for interface compatibility).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        shortcut: bool = True,
        g: int = 1,
    ):
        super().__init__()
        self.c = int(c2 * e)                                        # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)                      # split point
        self.cv2 = Conv((2 + n) * self.c, c2, 1)                   # fusion
        self.m = nn.ModuleList(
            LeConvBlock(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: C2f-style CSP with LeConvBlocks in the gradient path."""
        y = list(self.cv1(x).chunk(2, 1))                           # (y0, y1)
        y.extend(m(y[-1]) for m in self.m)                          # LeConvBlocks on y1
        return self.cv2(torch.cat(y, 1))


# ===================================================================
# Replacement function
# ===================================================================

def replace_backbone_with_leconv(model: nn.Module) -> int:
    """Replace backbone C3k2 blocks with C3k2LeConv (in-place).

    Scans ``model.model.model`` for ``C3k2`` instances (the backbone CSP
    blocks) and replaces each with a ``C3k2LeConv`` that uses LeConvBlock
    internally. Only C3k2 blocks are targeted — Conv, SPPF, C2PSA, and
    all neck/head components are left untouched.

    The replacement preserves:
      - All channel dimensions (c1, c2, hidden)
      - All feature pyramid levels (P3/P4/P5)
      - All skip connection indices
      - The CSP gradient structure

    Args:
        model: An Ultralytics YOLO model (post-initialisation).

    Returns:
        Number of C3k2 blocks replaced.
    """
    # ── Unwrap ────────────────────────────────────────────────────────
    if hasattr(model, "model") and hasattr(model.model, "model"):
        seq = model.model.model
    elif hasattr(model, "model"):
        seq = model.model
    else:
        logger.error("LeConv: Cannot unwrap model — unknown structure")
        return 0

    replaced = 0
    for i, module in enumerate(seq):
        if not isinstance(module, C3k2):
            continue

        # ── Read original parameters ──────────────────────────────────
        orig = module
        c1 = orig.cv1.conv.in_channels
        c2 = orig.cv2.conv.out_channels
        hidden = orig.cv1.conv.out_channels // 2
        n = len(orig.m)
        shortcut = getattr(orig, "add", True)

        # Reconstruct e (preserves hidden through the CSP structure)
        e = round(hidden / c2, 4) if c2 > 0 else 0.5

        logger.debug(
            "LeConv: replacing seq[%d]: C3k2 → C3k2LeConv "
            "(c1=%d, c2=%d, n=%d, e=%.2f, shortcut=%s)",
            i, c1, c2, n, e, shortcut,
        )

        # ── Create replacement ────────────────────────────────────────
        new_block = C3k2LeConv(
            c1=c1, c2=c2, n=n, e=e, shortcut=shortcut,
        )

        # Move new block to the same device as the original model
        # (new nn.Modules start on CPU; model may already be on GPU)
        device = next(orig.parameters()).device
        if device.type == "cuda":
            new_block = new_block.cuda()

        # Copy metadata attributes set by YOLO's parse_model()
        for attr in ("i", "f", "type"):
            if hasattr(orig, attr):
                setattr(new_block, attr, getattr(orig, attr))

        seq[i] = new_block
        replaced += 1

    logger.info(
        "LeConv backbone: Replaced %d C3k2 block(s) with C3k2LeConv ✓",
        replaced,
    )
    return replaced
