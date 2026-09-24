"""
StarNet Backbone — Element-Wise Feature Multiplication for YOLO.

Replaces the Bottleneck/C3k modules inside YOLO backbone C3k2 blocks with
StarBlock modules that use element-wise feature multiplication (the "star"
operation). This follows the StarNet paradigm: two parallel 1×1 projections
are multiplied element-wise, then processed by a depthwise conv and fused
via a 1×1 conv — all within the existing CSP (C2f) structure so channel
dimensions and skip connections remain unchanged.

Reference:
    "Rewrite the Stars" (Ma et al., 2024) — https://arxiv.org/abs/2403.19967

Usage:
    from starnet_backbone import replace_backbone_with_starnet
    model = YOLO("yolo11l.pt")
    n = replace_backbone_with_starnet(model)  # replaces backbone C3k2s in-place
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C3k2

logger = logging.getLogger(__name__)


# ===================================================================
# StarBlock
# ===================================================================

class StarBlock(nn.Module):
    """Core StarNet building block with element-wise feature multiplication.

    The "star operation" (element-wise multiply of two parallel projections)
    produces non-linear feature interactions at each spatial position without
    relying on conventional activation functions. This has been shown to
    improve representational efficiency in lightweight vision backbones.

    Flow:
        branch1: 1×1 conv → [star: *branch2] → DWConv → 1×1 conv → +shortcut
        branch2: 1×1 conv ↗

    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Enable residual connection (if c1 == c2).
        g: Groups for conv (unused, kept for interface compatibility).
        k: Kernel size for depthwise conv.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: int = 3):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)              # branch1 pointwise
        self.cv2 = Conv(c1, c2, 1, 1)              # branch2 pointwise
        self.dwconv = Conv(c2, c2, k, 1, g=c2)     # depthwise conv after multiply
        self.cv3 = Conv(c2, c2, 1, 1)              # fusion pointwise
        self.add = shortcut and c1 == c2            # residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: two parallel 1×1 → multiply → DWConv → 1×1 → +residual."""
        identity = x
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = x1 * x2                              # star operation (element-wise multiply)
        out = self.dwconv(out)                     # depthwise spatial mixing
        out = self.cv3(out)                        # fusion
        if self.add:
            out = out + identity
        return out


# ===================================================================
# C3k2Star — CSP block with StarBlock inside
# ===================================================================

class C3k2Star(nn.Module):
    """C3k2-compatible CSP block using StarBlock instead of Bottleneck/C3k.

    Preserves the exact C2f-based CSP structure that YOLO's C3k2 uses:
        cv1(x) → chunk(2) ─┬─ y0 ──────────────┐
                            └─ y1 → StarBlock × n ─┐
                                                    ↓
                                            cv2(cat[y0, y1, StarBlock×n])

    The ``c3k`` parameter is accepted (for API compatibility with C3k2)
    but ignored — StarBlock is always used as the internal module.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of StarBlocks.
        c3k: Unused (compatibility). Always uses StarBlock.
        e: Expansion ratio for hidden channels.
        shortcut: Whether internal StarBlocks use residual connections.
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
        self.c = int(c2 * e)                                           # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)                         # split point
        self.cv2 = Conv((2 + n) * self.c, c2, 1)                      # fusion
        self.m = nn.ModuleList(
            StarBlock(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: C2f-style CSP with StarBlocks in the gradient path."""
        y = list(self.cv1(x).chunk(2, 1))                              # (y0, y1)
        y.extend(m(y[-1]) for m in self.m)                             # StarBlocks on y1
        return self.cv2(torch.cat(y, 1))


# ===================================================================
# Replacement function
# ===================================================================

def replace_backbone_with_starnet(model: nn.Module) -> int:
    """Replace backbone C3k2 blocks with C3k2Star (in-place).

    Scans ``model.model.model`` for ``C3k2`` instances (the backbone CSP
    blocks) and replaces each with a ``C3k2Star`` that uses StarBlock
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
        logger.error("StarNet: Cannot unwrap model — unknown structure")
        return 0

    replaced = 0
    for i, module in enumerate(seq):
        if not isinstance(module, C3k2):
            continue

        # ── Read original parameters ──────────────────────────────────
        orig = module
        c1 = orig.cv1.conv.in_channels
        c2 = orig.cv2.conv.out_channels
        hidden = orig.cv1.conv.out_channels // 2    # recover hidden dim
        n = len(orig.m)                              # number of internal blocks
        shortcut = getattr(orig, "add", True)        # original shortcut preference

        # Reconstruct e (preserves hidden through the CSP structure)
        e = round(hidden / c2, 4) if c2 > 0 else 0.5

        logger.debug(
            "StarNet: replacing seq[%d]: C3k2 → C3k2Star (c1=%d, c2=%d, n=%d, e=%.2f, shortcut=%s)",
            i, c1, c2, n, e, shortcut,
        )

        # ── Create replacement ────────────────────────────────────────
        new_block = C3k2Star(c1=c1, c2=c2, n=n, e=e, shortcut=shortcut)

        # Copy metadata attributes set by YOLO's parse_model()
        # (required for the forward loop routing and skip connections)
        for attr in ("i", "f", "type"):
            if hasattr(orig, attr):
                setattr(new_block, attr, getattr(orig, attr))

        seq[i] = new_block
        replaced += 1

    logger.info(
        "StarNet backbone: Replaced %d C3k2 block(s) with C3k2Star ✓",
        replaced,
    )
    return replaced
