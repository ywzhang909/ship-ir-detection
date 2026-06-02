"""
ASFF — Adaptive Spatial Feature Fusion for YOLO Neck.

Replaces Concat layers in the Neck with learnable weighted fusion.
Each input gets a learned spatial weight (via 1x1 conv + softmax),
improving multi-scale feature aggregation. The weighted outputs are
concatenated (same channel count as Concat), keeping downstream
layers unchanged.

Key design: lazy initialization from actual tensor shapes on first
forward pass — no fragile static channel inference needed.

Reference:
    "Learning Spatial Fusion for Single-Shot Object Detection" (arXiv:1911.09516)

Usage:
    from asff import replace_concat_with_asff
    model = YOLO("yolo11l.pt")
    replace_concat_with_asff(model)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# ASFF — Adaptive Spatial Feature Fusion
# ===================================================================

class ASFF(nn.Module):
    """Adaptive Spatial Feature Fusion — drop-in for Concat.

    Takes N feature tensors, learns per-pixel fusion weights (from
    the first input's features), applies weighted fusion per input,
    then concatenates results. Weighted outputs have same dims as
    Concat, so downstream modules are unaffected.

    Lazy init: weight_conv is created from actual tensor shapes on
    the first forward() call.
    """

    def __init__(self):
        super().__init__()
        self.weight_conv: nn.Conv2d | None = None
        self._n_inputs: int = 0

    def _lazy_init(self, x: list[torch.Tensor]) -> None:
        """Create weight_conv from first input's channel count."""
        if self.weight_conv is not None:
            return
        self._n_inputs = len(x)
        in_ch = x[0].shape[1]

        self.weight_conv = nn.Conv2d(in_ch, self._n_inputs, kernel_size=1, bias=False)
        device, dtype = x[0].device, x[0].dtype
        self.weight_conv.to(device=device, dtype=dtype)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse N feature maps with learned weights, then concat.

        Args:
            x: List of N tensors [(B, C1, H1, W1), (B, C2, H2, W2), ...].

        Returns:
            Weighted + concatenated tensor (B, sum(Ci), H1, W1).
        """
        if not isinstance(x, list) or len(x) < 2:
            return x[0] if isinstance(x, list) else x

        self._lazy_init(x)

        # Resize all to first input's spatial size
        target_spatial = x[0].shape[2:]
        aligned: list[torch.Tensor] = []
        for t in x:
            if t.shape[2:] != target_spatial:
                t = F.interpolate(t, size=target_spatial, mode="bilinear", align_corners=False)
            aligned.append(t)

        # Predict fusion weights from first input's features
        # (weight_conv output = per-pixel weight per input)
        weights = self.weight_conv(aligned[0])  # (B, N, H, W)
        weights = F.softmax(weights, dim=1)

        # Apply weighted fusion per input (broadcasts: (B,1,H,W) * (B,Ci,H,W))
        weighted = [
            weights[:, i:i+1, :, :] * aligned[i]
            for i in range(len(aligned))
        ]

        # Concatenate weighted outputs (same as Concat)
        out = torch.cat(weighted, dim=1)
        return out


# ===================================================================
# Replace Concat with ASFF in Neck
# ===================================================================

def replace_concat_with_asff(model: nn.Module) -> int:
    """Replace Concat layers in the Neck with ASFF modules.

    Uses lazy-initialized ASFF that reads the actual input tensor
    shapes on the first forward pass — no static channel inference.

    Args:
        model: A YOLO or DetectionModel object.

    Returns:
        Number of Concat layers replaced.
    """
    from ultralytics.nn.modules import Concat
    from attention_modules import _find_backbone_end

    # Unwrap model
    if hasattr(model, "model") and hasattr(model.model, "model"):
        model_seq = model.model.model
    elif hasattr(model, "model"):
        model_seq = model.model
    else:
        model_seq = model

    backbone_end = _find_backbone_end(model_seq)
    replaced = 0

    for i, layer in enumerate(model_seq):
        if i <= backbone_end:
            continue
        if not isinstance(layer, Concat):
            continue

        asff = ASFF()
        asff.i = i
        asff.f = layer.f
        model_seq[i] = asff
        replaced += 1
        logger.info("Replaced Concat [%d] → ASFF (lazy)", i)

    if replaced:
        logger.info("ASFF: Replaced %d Concat layers in Neck", replaced)
    else:
        logger.warning("ASFF: No eligible Concat layers found")

    return replaced
