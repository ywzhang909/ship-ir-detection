"""
C3K3 — Replace C3k2 blocks with C3k (C3-style topology, 3x3 bottlenecks).

Replaces C2f-style C3k2 blocks with C3-style C3k blocks. While both
use 3x3 Bottleneck convs by default, the key difference is topology:

    C2f (C3k2): cv1(1x1x2ch) -> [bottleneck x n] -> concat -> cv2(1x1)
    C3  (C3k):  cv1(1x1) -> [bottleneck x n] -> concat + cv2(skip) -> cv3(1x1)

C3 adds an explicit skip path (cv2) that preserves original features,
similar to a residual connection around the bottleneck stack.

Reference:
    YOLOv5/v8 C3 module -- CSP bottleneck with 3 convolutions.

Usage:
    from c3k3 import replace_c3k2_with_c3k
    model = YOLO("yolo11l.pt")
    replace_c3k2_with_c3k(model)
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def replace_c3k2_with_c3k(model: nn.Module, kernel_size: int = 3) -> int:
    """Replace all C3k2 blocks in backbone + neck with C3k blocks.

    C3k uses C3-style CSP topology with an explicit skip path around
    the bottleneck stack, and 3x3 bottleneck kernels (k=3 default).

    Args:
        model: A YOLO or DetectionModel object.
        kernel_size: Kernel size for C3k bottlenecks (default 3).

    Returns:
        Number of C3k2 blocks replaced.
    """
    from ultralytics.nn.modules.block import C3k, C3k2

    # Unwrap model: YOLO -> DetectionModel -> Sequential
    if hasattr(model, "model") and hasattr(model.model, "model"):
        model_seq = model.model.model
    elif hasattr(model, "model"):
        model_seq = model.model
    else:
        model_seq = model

    replaced = 0
    for i, layer in enumerate(model_seq):
        if not isinstance(layer, C3k2):
            continue

        # Infer parameters from existing C3k2 module
        cv1_in_ch = layer.cv1.conv.in_channels
        cv2_out_ch = layer.cv2.conv.out_channels
        n = len(layer.m)
        c = layer.c  # hidden channels
        e = c / cv2_out_ch if cv2_out_ch > 0 else 0.5
        e = min(max(e, 0.1), 1.0)
        shortcut = getattr(layer, "shortcut", True)

        # Create replacement C3k (C3-style, k=3 bottleneck)
        c3k = C3k(
            c1=cv1_in_ch,
            c2=cv2_out_ch,
            n=n,
            shortcut=shortcut,
            g=1,
            e=e,
            k=kernel_size,
        )

        # Copy routing indices
        c3k.i = layer.i
        c3k.f = layer.f

        model_seq[i] = c3k
        replaced += 1
        logger.info(
            "Replaced C3k2 [%d] -> C3k (k=%d, n=%d, e=%.2f, in=%d, out=%d)",
            i, kernel_size, n, e, cv1_in_ch, cv2_out_ch,
        )

    if replaced:
        logger.info("C3K3: Replaced %d C3k2 blocks with C3k (k=%d)", replaced, kernel_size)
    else:
        logger.warning("C3K3: No C3k2 blocks found")

    return replaced
