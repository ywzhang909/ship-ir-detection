"""
DynamicHead — Scale-aware + Spatial-aware Attention for YOLO Detect Head.

Replaces the standard YOLO Detect module with a Dynamic Head that applies
multi-level attention before the final detection convolutions.

Reference:
    "Dynamic Head: Unifying Object Detection Heads with Attentions" (CVPR 2021)

Architecture:
    1. Scale-aware attention — learned per-level importance weight
    2. Spatial-aware attention — per-pixel gating via lightweight conv
    3. Original YOLO detection convs (cv2 box / cv3 cls) — unchanged

Usage:
    from dynamic_head import replace_detect_with_dynamic_head
    model = YOLO("yolo11l.pt")
    replace_detect_with_dynamic_head(model)
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Detect, DFL, DWConv
from ultralytics.utils.tal import dist2bbox, make_anchors

logger = logging.getLogger(__name__)


# ===================================================================
# Dynamic Head
# ===================================================================

class DynamicHead(Detect):
    """YOLO Detect head with scale-aware + spatial-aware attention.

    Extends the standard YOLO Detect module by adding two lightweight
    attention mechanisms before the original detection convolutions:

    - **Scale-aware**: a learnable per-level scalar (after sigmoid) that
      reweights each feature pyramid level. Helps the model attend to
      the most informative scale for small IR targets.

    - **Spatial-aware**: a per-level 1×1 → 3×3 conv → sigmoid gating
      that produces a pixel-wise attention map. Helps suppress clutter
      and highlight ship locations.

    The original cv2 (box regression) and cv3 (classification) branches
    are preserved unchanged, so downstream YOLO training logic (loss,
    NMS, stride computation) works without modification.

    Args:
        nc: Number of classes.
        reg_max: DFL channels (16 for YOLO11).
        ch: Tuple of input channel counts per detection level.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, ch: tuple = ()):
        super().__init__(nc, reg_max, ch=ch)

        # ── Scale-aware attention ────────────────────────────────────
        # Per-level learnable scalar; sigmoid-normalised so range is (0, 1).
        # Initialised to ~0.5 so attention starts near-identity.
        self.scale_attn = nn.Parameter(torch.zeros(len(ch)))

        # ── Spatial-aware attention ───────────────────────────────────
        # Per-level lightweight gating: 1×1 reduce → 3×3 → sigmoid.
        # Each level gets its own predictor because spatial patterns differ
        # between shallow (high-res) and deep (low-res) features.
        self.spatial_convs = nn.ModuleList()
        for c in ch:
            reduced = max(4, c // 2)
            self.spatial_convs.append(
                nn.Sequential(
                    Conv(c, reduced, 1),              # 1×1 channel reduction
                    nn.Conv2d(reduced, 1, 3, padding=1),  # 3×3 → single-channel map
                    nn.Sigmoid(),                     # pixel-wise gate in (0,1)
                )
            )

        logger.info(
            "DynamicHead: scale_attn (L=%d) + spatial_attn (per-level 1×1→3×3 gate)",
            len(ch),
        )

    # ------------------------------------------------------------------
    # Forward head — called by both training and inference paths
    # ------------------------------------------------------------------

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict:
        """Apply scale → spatial attention, then run detection convs.

        Args:
            x: List of feature tensors per level [(B, Ci, Hi, Wi), ...].
            box_head: Box regression head (cv2 or one2one_cv2).
            cls_head: Classification head (cv3 or one2one_cv3).

        Returns:
            Dictionary with keys 'boxes', 'scores', 'feats'.
        """
        if box_head is None or cls_head is None:
            return dict()

        # ── 1. Scale-aware attention ─────────────────────────────────
        # Each level's entire feature map is scaled by a learned weight.
        scale_weights = torch.sigmoid(self.scale_attn)  # (L,)
        x = [scale_weights[i] * x[i] for i in range(len(x))]

        # ── 2. Spatial-aware attention ───────────────────────────────
        # Per-level pixel-wise gating: conv → sigmoid → element-wise multiply.
        x = [self.spatial_convs[i](x[i]) * x[i] for i in range(len(x))]

        # ── 3. Original detection convs ──────────────────────────────
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)


# ===================================================================
# Replacement function
# ===================================================================

def replace_detect_with_dynamic_head(model: nn.Module) -> bool:
    """Replace the YOLO Detect head with a DynamicHead (in-place).

    Extracts channel dimensions from the existing Detect module, creates
    a DynamicHead with identical cv2/cv3 architecture, copies stride and
    weights, and swaps it into ``model.model[-1]``.

    Args:
        model: A YOLO or DetectionModel object.

    Returns:
        True if replacement succeeded, False otherwise.
    """
    from ultralytics.nn.modules import Detect as UltralyticsDetect

    # ── Unwrap YOLO → DetectionModel → Sequential ────────────────────
    if hasattr(model, "model") and hasattr(model.model, "model"):
        model_seq = model.model.model
        detection_model = model.model
    elif hasattr(model, "model"):
        model_seq = model.model
        detection_model = model
    else:
        logger.error("DynamicHead: Cannot unwrap model — unknown structure")
        return False

    head = model_seq[-1]
    if not isinstance(head, UltralyticsDetect):
        logger.warning(
            "DynamicHead: Last layer is %s, not Detect — skipping",
            type(head).__name__,
        )
        return False

    # ── Extract channel dims from existing Detect ────────────────────
    # Each cv2 entry: Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), Conv2d(c2, ...))
    # The first Conv's in_channels = x
    ch = [cv2[0].conv.in_channels for cv2 in head.cv2]
    nc = head.nc
    reg_max = head.reg_max

    logger.info(
        "DynamicHead: Replacing Detect (nc=%d, reg_max=%d, levels=%d, ch=%s)",
        nc, reg_max, len(ch), ch,
    )

    # ── Create DynamicHead ───────────────────────────────────────────
    dynamic_head = DynamicHead(nc=nc, reg_max=reg_max, ch=tuple(ch))

    # ── Copy state from original Detect ──────────────────────────────
    # Copy stride (computed during model build)
    dynamic_head.stride = head.stride.clone() if hasattr(head, "stride") else head.stride
    dynamic_head.nl = head.nl
    dynamic_head.no = head.no
    dynamic_head.shape = getattr(head, "shape", None)
    dynamic_head.anchors = getattr(head, "anchors", torch.empty(0))
    dynamic_head.strides = getattr(head, "strides", torch.empty(0))
    dynamic_head.inplace = getattr(head, "inplace", True)
    dynamic_head.legacy = getattr(head, "legacy", False)
    dynamic_head.export = getattr(head, "export", False)

    # Copy end2end attributes if present
    if hasattr(head, "end2end") and head.end2end:
        dynamic_head.end2end = True
        dynamic_head.one2one_cv2 = deepcopy(head.one2one_cv2)
        dynamic_head.one2one_cv3 = deepcopy(head.one2one_cv3)

    # ── Copy pretrained weights for cv2/cv3/dfl ─────────────────────
    # This preserves the pre-trained detection capability
    for i in range(len(ch)):
        dynamic_head.cv2[i].load_state_dict(head.cv2[i].state_dict())
        dynamic_head.cv3[i].load_state_dict(head.cv3[i].state_dict())

    if hasattr(head, "dfl") and hasattr(dynamic_head, "dfl"):
        try:
            dynamic_head.dfl.load_state_dict(head.dfl.state_dict())
        except (RuntimeError, AttributeError):
            pass  # DFL might be Identity or have different structure

    # ── Copy device and dtype ────────────────────────────────────────
    device = next(head.parameters()).device if list(head.parameters()) else None
    if device is not None:
        dynamic_head.to(device=device)

    # ── Swap into model ──────────────────────────────────────────────
    model_seq[-1] = dynamic_head
    dynamic_head.i = head.i
    dynamic_head.f = head.f
    dynamic_head.type = head.type if hasattr(head, "type") else "DetectionModel"

    # Update detection_model.save if needed (for skip connections)
    if hasattr(detection_model, "save"):
        save_set = set(detection_model.save)
        detection_model.save = save_set

    logger.info("DynamicHead: Replacement complete ✓")
    return True
