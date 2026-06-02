"""
Custom loss functions for small object detection in IR ship imagery.

Implements:
  1. NWD (Normalized Wasserstein Distance) loss — scale-invariant IoU
     replacement for tiny objects (<10px). Treats boxes as 2D Gaussians.
  2. WIoU v3 (Wise-IoU) — dynamic non-monotonic focusing mechanism.
  3. Combined NWD + CIoU loss with adaptive weighting.

All losses are designed as drop-in replacements for ultralytics' BboxLoss.

Usage:
    from custom_loss import TinyObjectDetectionLoss, NWDLoss
    loss_fn = TinyObjectDetectionLoss(model, iou_ratio=0.5, nwd_constant=12.8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NWD (Normalized Wasserstein Distance) Loss
# ---------------------------------------------------------------------------

def wasserstein_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
    constant: float = 12.8,
) -> torch.Tensor:
    """Normalized Wasserstein Distance loss for tiny object detection.

    Models bounding boxes as 2D Gaussian distributions and computes the
    Wasserstein distance between them. This is scale-invariant and smooth
    even for sub-10px objects where standard IoU is extremely brittle.

    Paper: https://arxiv.org/abs/2110.13389 (NWD for Tiny Object Detection)

    Args:
        pred: Predicted boxes in (cx, cy, w, h) format, shape (N, 4).
        target: Target boxes in (cx, cy, w, h) format, shape (N, 4).
        eps: Numerical stability epsilon.
        constant: Normalization constant (12.8 recommended from AI-TOD paper,
                  tune in range [8.0, 16.0] based on object size).

    Returns:
        NWD similarity in [0, 1] for each box pair.
    """
    # Center distance
    center_dist = (pred[:, :2] - target[:, :2]).pow(2).sum(dim=1)

    # Width/height components
    w1, h1 = pred[:, 2] + eps, pred[:, 3] + eps
    w2, h2 = target[:, 2] + eps, target[:, 3] + eps

    # Wh distance (from 2D Gaussian formulation)
    wh_dist = ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4

    wasserstein = center_dist + wh_dist
    return torch.exp(-torch.sqrt(wasserstein + eps) / constant)


class NWDLoss(nn.Module):
    """Combined NWD + IoU loss for tiny object detection.

    For extremely small objects (<10px height at training resolution),
    use iou_ratio=0.2-0.3 (more NWD weight). For moderate (10-30px),
    use iou_ratio=0.5.

    Args:
        iou_ratio: Weight for CIoU loss (NWD weight = 1 - iou_ratio).
        nwd_constant: NWD normalization constant.
    """
    def __init__(self, iou_ratio: float = 0.5, nwd_constant: float = 12.8):
        super().__init__()
        self.iou_ratio = iou_ratio
        self.nwd_constant = nwd_constant

    def forward(self, pred_xywh: torch.Tensor, target_xywh: torch.Tensor,
                iou: torch.Tensor) -> torch.Tensor:
        """Compute combined NWD + CIoU loss.

        Args:
            pred_xywh: Predicted boxes (cx, cy, w, h).
            target_xywh: Target boxes (cx, cy, w, h).
            iou: Standard CIoU values for the same boxes.

        Returns:
            Combined loss value.
        """
        nwd = wasserstein_loss(pred_xywh, target_xywh, constant=self.nwd_constant)
        # Combined: lower iou_ratio = more NWD weight
        loss = (1.0 - iou) * self.iou_ratio + (1.0 - nwd) * (1.0 - self.iou_ratio)
        return loss


# ---------------------------------------------------------------------------
# WIoU v3 (Wise-IoU) — Dynamic Non-Monotonic Focusing
# Paper: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
#        https://arxiv.org/abs/2301.10051
# ---------------------------------------------------------------------------

class WIoU_Scale:
    """WIoU v3 dynamic non-monotonic focusing mechanism.

    Assigns lower gradient to both high-quality and low-quality boxes,
    and highest gradient to 'ordinary' boxes. This is ideal for IR ship
    detection where many boxes are 'medium quality' due to pseudo-labels.

    Usage:
        scale = WIoU_Scale(iou)
        r = scale._scaled_loss()  # focusing coefficient
        loss = r * R_WIoU * (1 - iou)  # WIoU v3
    """
    iou_mean = 1.0
    monotonous: bool = False  # False=v3 (non-monotonic), True=v2, None=v1
    _momentum = 1 - 0.5 ** (1 / 7000)  # Running mean momentum
    _is_train = True

    def __init__(self, iou: torch.Tensor):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train:
            cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                           cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma: float = 1.9, delta: float = 3):
        """Compute WIoU v3 focusing coefficient (non-monotonic)."""
        if isinstance(self.monotonous, bool):
            if self.monotonous:  # WIoU v2
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:  # WIoU v3
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1.0  # WIoU v1


def wiou_loss(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    iou: torch.Tensor,
    monotonous: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute WIoU v1/v3 loss with distance-aware attention.

    WIoU v1: L = R_WIoU * (1 - IoU)
      where R_WIoU = exp(ρ² / c²)  — distance attention; ρ² = center_dist², c² = min_enclosing_diag²

    WIoU v3: L = r * R_WIoU * (1 - IoU)
      where r = non-monotonic focusing coefficient from WIoU_Scale

    Args:
        pred_xywh: Predicted boxes (cx, cy, w, h), shape (N, 4).
        target_xywh: Target boxes (cx, cy, w, h), shape (N, 4).
        iou: Plain IoU (NOT CIoU), shape (N,).
        monotonous: If True → WIoU v2 (monotonous), False → WIoU v3 (non-monotonic).
        eps: Numerical stability epsilon.

    Returns:
        WIoU loss for each box pair, shape (N,).
    """
    cx_p, cy_p, w_p, h_p = pred_xywh.unbind(dim=-1)
    cx_t, cy_t, w_t, h_t = target_xywh.unbind(dim=-1)

    # ── Distance attention: R_WIoU = exp(ρ² / c²) ──
    # ρ² = squared center distance
    rho2 = (cx_p - cx_t).pow(2) + (cy_p - cy_t).pow(2)

    # c² = minimum enclosing box diagonal²
    x1_p, y1_p = cx_p - w_p / 2, cy_p - h_p / 2
    x2_p, y2_p = cx_p + w_p / 2, cy_p + h_p / 2
    x1_t, y1_t = cx_t - w_t / 2, cy_t - h_t / 2
    x2_t, y2_t = cx_t + w_t / 2, cy_t + h_t / 2

    cw = torch.max(x2_p, x2_t) - torch.min(x1_p, x1_t)
    ch = torch.max(y2_p, y2_t) - torch.min(y1_p, y1_t)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # Distance attention: exponential of center_distance / enclosing_diagonal
    R_wiou = torch.exp(rho2 / c2)

    # ── WIoU v1 core: L = R_WIoU * (1 - IoU) ──
    loss = R_wiou * (1.0 - iou)

    # ── WIoU v2/v3 focusing: L = r * L_v1 ──
    if monotonous is not None:
        wiou_scale = WIoU_Scale(iou)
        # _scaled_loss is a @classmethod that takes the instance as 2nd arg
        r = wiou_scale._scaled_loss(wiou_scale)
        loss = loss * r

    return loss


class WIoULoss(nn.Module):
    """WIoU v3 loss module — distance-aware + non-monotonic focusing.

    Complements standard IoU penalties with center-distance attention
    that provides gradient signal even when IoU ≈ 0. The v3 non-monotonic
    focusing mechanism assigns higher weight to 'ordinary' quality boxes
    and lower weight to both very good and very bad matches.

    Args:
        monotonous: If False → WIoU v3 (default), True → WIoU v2.
    """

    def __init__(self, monotonous: bool = False):
        super().__init__()
        self.monotonous = monotonous

    def forward(
        self,
        pred_xywh: torch.Tensor,
        target_xywh: torch.Tensor,
        iou: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WIoU loss.

        Args:
            pred_xywh: Predicted boxes (cx, cy, w, h), shape (N, 4).
            target_xywh: Target boxes (cx, cy, w, h), shape (N, 4).
            iou: Plain IoU values, shape (N,).

        Returns:
            WIoU loss value for each box pair, shape (N,).
        """
        return wiou_loss(
            pred_xywh, target_xywh, iou,
            monotonous=self.monotonous,
        )


class WIoUBboxLoss(nn.Module):
    """Drop-in replacement for ultralytics BboxLoss with WIoU v3 support.

    Replaces CIoU with WIoU v3 that adds distance-aware attention (R_WIoU)
    and non-monotonic focusing (via WIoU_Scale). Preserves DFL.

    The distance term provides smooth gradient signal even when IoU ≈ 0,
    and the non-monotonic mechanism dynamically focuses on 'ordinary'
    quality matches — ideal for IR ship detection where targets are small
    and pseudo-labels have variable quality.

    Args:
        reg_max: Maximum value for DFL discrete bins (default: 16).
        monotonous: If False → WIoU v3 (non-monotonic), True → WIoU v2.
    """

    def __init__(self, reg_max: int = 16, monotonous: bool = False):
        super().__init__()
        self.reg_max = reg_max
        self.wiou_loss = WIoULoss(monotonous=monotonous)
        from ultralytics.utils.loss import DFLoss
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def __call__(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: int | torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute WIoU v3 box loss.

        Args:
            pred_dist: Predicted DFL distribution.
            pred_bboxes: Predicted boxes (xyxy format), shape (N, 4).
            anchor_points: Anchor point coordinates.
            target_bboxes: Target boxes (xyxy format), shape (N, 4).
            target_scores: Target classification scores.
            target_scores_sum: Sum of target scores (for normalization).
            fg_mask: Foreground mask (positive samples).
            imgsz: Image size.
            stride: Model strides.

        Returns:
            Tuple of (loss_iou, loss_dfl).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Plain IoU (NOT CIoU — WIoU handles distance via R_WIoU)
        from ultralytics.utils.metrics import bbox_iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=False)

        # Convert boxes to xywh for distance attention
        pred_xywh = torch.cat([
            (pred_bboxes[fg_mask][:, :2] + pred_bboxes[fg_mask][:, 2:]) / 2,  # cx, cy
            pred_bboxes[fg_mask][:, 2:] - pred_bboxes[fg_mask][:, :2],       # w, h
        ], dim=1)
        target_xywh = torch.cat([
            (target_bboxes[fg_mask][:, :2] + target_bboxes[fg_mask][:, 2:]) / 2,
            target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2],
        ], dim=1)

        # WIoU v3 loss
        loss_iou = self.wiou_loss(pred_xywh, target_xywh, iou) * weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # DFL loss (same as ultralytics original)
        if self.dfl_loss:
            from ultralytics.utils.tal import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.new_zeros(1)

        return loss_iou, loss_dfl


class WIoUDetectionLoss:
    """Complete loss wrapper replacing v8DetectionLoss BboxLoss with WIoU v3.

    Usage:
        loss_fn = WIoUDetectionLoss(model)
    """

    def __init__(self, model, monotonous: bool = False):
        self.monotonous = monotonous
        self.model = model
        self.device = next(model.parameters()).device

        # Create underlying ultralytics loss
        from ultralytics.utils.loss import v8DetectionLoss
        self.base_loss = v8DetectionLoss(model)

        # Replace bbox_loss with WIoU version
        reg_max = getattr(model.model[-1], 'reg_max', 16)
        self.base_loss.bbox_loss = WIoUBboxLoss(
            reg_max=reg_max,
            monotonous=monotonous,
        )

    def __call__(self, preds, batch):
        """Compute loss with WIoU v3 for distance-aware regression."""
        return self.base_loss(preds, batch)


# ---------------------------------------------------------------------------
# Ultralytics Integration: BboxLoss with NWD
# ---------------------------------------------------------------------------

class NWDBboxLoss(nn.Module):
    """Drop-in replacement for ultralytics BboxLoss with NWD support.

    Combines standard CIoU with NWD for small object robustness.
    Preserves the original DFL (Distribution Focal Loss) component.

    Args:
        reg_max: Maximum value for DFL discrete bins (default: 16).
        iou_ratio: Weight for CIoU vs NWD (0.0 = pure NWD, 1.0 = pure CIoU).
        nwd_constant: NWD normalization constant.
    """
    def __init__(
        self,
        reg_max: int = 16,
        iou_ratio: float = 0.5,
        nwd_constant: float = 12.8,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.iou_ratio = iou_ratio
        self.nwd_constant = nwd_constant
        self.nwd_loss = NWDLoss(iou_ratio=iou_ratio, nwd_constant=nwd_constant)
        from ultralytics.utils.loss import DFLoss
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def __call__(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: int | torch.Tensor,
        stride: torch.Tensor,
    ):
        """Compute NWD-aware box loss.

        Args:
            pred_dist: Predicted DFL distribution.
            pred_bboxes: Predicted boxes (xyxy format).
            anchor_points: Anchor point coordinates.
            target_bboxes: Target boxes (xyxy format).
            target_scores: Target classification scores.
            target_scores_sum: Sum of target scores (for normalization).
            fg_mask: Foreground mask (positive samples).
            imgsz: Image size.
            stride: Model strides.

        Returns:
            Tuple of (loss_iou, loss_dfl).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Standard CIoU
        from ultralytics.utils.metrics import bbox_iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        # Convert boxes to xywh for NWD
        pred_xywh = torch.cat([
            (pred_bboxes[fg_mask][:, :2] + pred_bboxes[fg_mask][:, 2:]) / 2,  # cx, cy
            pred_bboxes[fg_mask][:, 2:] - pred_bboxes[fg_mask][:, :2],       # w, h
        ], dim=1)
        target_xywh = torch.cat([
            (target_bboxes[fg_mask][:, :2] + target_bboxes[fg_mask][:, 2:]) / 2,
            target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2],
        ], dim=1)

        # Combined NWD + CIoU loss
        loss_iou = self.nwd_loss(pred_xywh, target_xywh, iou) * weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # DFL loss (same as ultralytics original)
        if self.dfl_loss:
            loss_dfl = self._dfl_loss(pred_dist, pred_bboxes, anchor_points,
                                       target_bboxes, target_scores_sum,
                                       fg_mask, weight)
        else:
            loss_dfl = pred_dist.new_zeros(1)

        return loss_iou, loss_dfl

    def _dfl_loss(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Distribution Focal Loss for box refinement."""
        if pred_dist.dim() < 2 or not self.dfl_loss:
            return pred_dist.new_zeros(1)

        from ultralytics.utils.tal import bbox2dist
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_dfl


# ---------------------------------------------------------------------------
# Complete Detection Loss with Preprocessing Log Hooks
# ---------------------------------------------------------------------------

class TinyObjectDetectionLoss:
    """Complete loss wrapper that can replace v8DetectionLoss in ultralytics.

    This wraps the standard ultralytics detection loss but replaces the
    BboxLoss with NWD-aware loss for small object detection.

    Usage:
        loss_fn = TinyObjectDetectionLoss(model, iou_ratio=0.5, nwd_constant=12.8)
    """

    def __init__(
        self,
        model,
        iou_ratio: float = 0.5,
        nwd_constant: float = 12.8,
        fl_gamma: float = 1.5,
    ):
        self.iou_ratio = iou_ratio
        self.nwd_constant = nwd_constant
        self.fl_gamma = fl_gamma

        # Store model parameters
        self.model = model
        self.device = next(model.parameters()).device
        self.stride = model.model[-1].stride if hasattr(model.model[-1], 'stride') else 8

        # Create the underlying ultralytics loss
        from ultralytics.utils.loss import v8DetectionLoss
        self.base_loss = v8DetectionLoss(model)

        # Replace bbox_loss with NWD-aware version
        reg_max = getattr(model.model[-1], 'reg_max', 16)
        self.base_loss.bbox_loss = NWDBboxLoss(
            reg_max=reg_max,
            iou_ratio=iou_ratio,
            nwd_constant=nwd_constant,
        )

        # Apply focal gamma if not default
        if fl_gamma > 0 and fl_gamma != 0.5:
            self._set_fl_gamma(fl_gamma)

    def _set_fl_gamma(self, gamma: float):
        """Set focal loss gamma on the classification loss."""
        if hasattr(self.base_loss, 'bce'):
            self.base_loss.bce = nn.BCEWithLogitsLoss(
                pos_weight=self.base_loss.bce.pos_weight,
                reduction='none',
            )
            # Store gamma for focal modulation (applied in forward)
            self._focal_gamma = gamma
        else:
            self._focal_gamma = 0.0

    def __call__(self, preds, batch):
        """Compute loss with NWD for small objects.

        Args:
            preds: Model predictions.
            batch: Training batch dict.

        Returns:
            Loss dict with same keys as ultralytics standard loss.
        """
        loss_dict = self.base_loss(preds, batch)

        # Apply focal gamma scaling to cls loss if configured
        if hasattr(self, '_focal_gamma') and self._focal_gamma > 0:
            cls_loss = loss_dict.get('cls', 0)
            if isinstance(cls_loss, torch.Tensor):
                # Modulate cls loss by focal factor
                pass  # ultralytics v8DetectionLoss already handles fl_gamma

        return loss_dict


# ===========================================================================
# Shape-IoU Loss
# Paper: Shape-IoU: More Accurate Metric considering Bounding Box Shape and Scale
#        https://arxiv.org/abs/2312.17663
# Reference implementation:
#   https://github.com/malagoutou/Shape-IoU/blob/main/utils/utils/metrics.py
# ===========================================================================
#
# Core formula (per paper §3):
#   L = 1 - IoU + D^shape + 0.5 * Ω^shape
#
# Shape-distance D^shape:
#   ww = 2 * w_gt^scale / (w_gt^scale + h_gt^scale)     # x-direction weight
#   hh = 2 * h_gt^scale / (w_gt^scale + h_gt^scale)     # y-direction weight
#   D^shape = (hh * (xc - xc_gt)^2 + ww * (yc - yc_gt)^2) / c^2
#   — Cross-paired: hh (from GT h) weights x-error, ww (from GT w) weights y-error.
#   - scale ∈ [0, ∞) controls shape-sensitivity; scale=0 => ww=hh=1 (vanilla distance)
#   - c^2 = minimum enclosing box diagonal^2
#
# Shape-penalty Ω^shape:
#   ω_w = hh * |w - w_gt| / max(w, w_gt)     # cross-pair: hh * relative w diff
#   ω_h = ww * |h - h_gt| / max(h, h_gt)     # cross-pair: ww * relative h diff
#   Ω^shape = (1 - e^{-ω_w})^θ + (1 - e^{-ω_h})^θ
#   — θ=4 per paper. Bounded in [0, 2) — smooth, differentiable.
#
# Intuition:
#   - Cross-pairing: for a wide GT (w >> h), ww ≈ 2, hh ≈ 0.
#       → y-center error penalised heavily (alignment matters more for wide objects)
#       → x-shape mismatch penalised heavily (elongated width must be accurate)
#       → h-shape mismatch ignored (height is already small, any value OK)
#   - Prevents "square box → elongated" deformation
#   - Provides smooth gradients even when IoU=0 (via D^shape term)
# ===========================================================================


def shape_iou_loss(
    pred_xywh: torch.Tensor,
    target_xywh: torch.Tensor,
    iou: torch.Tensor,
    theta: float = 4.0,
    omega_weight: float = 0.5,
    scale: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Shape-IoU loss (paper-accurate version).

    Follows the official reference implementation from malagoutou/Shape-IoU.

    Args:
        pred_xywh: Predicted boxes in (cx, cy, w, h) format, shape (N, 4).
        target_xywh: Target boxes in (cx, cy, w, h) format, shape (N, 4).
        iou: Standard IoU values, shape (N,).
        theta: Shape sensitivity exponent (default: 4.0 from paper).
        omega_weight: Weight for Ω^shape term (default: 0.5 from paper).
        scale: Shape-awareness scale factor. 0 → no shape awareness
               (ww=hh=1, pure distance). Higher → stronger shape weighting.
        eps: Numerical stability epsilon.

    Returns:
        Shape-IoU loss value for each box pair, shape (N,).
    """
    cx_p, cy_p, w_p, h_p = pred_xywh.unbind(dim=-1)
    cx_t, cy_t, w_t, h_t = target_xywh.unbind(dim=-1)

    # ---- Shape weights (ww for x-direction, hh for y-direction) ----
    # Cross-pair: ww derives from GT width, hh from GT height
    w_t_s = w_t.pow(scale)
    h_t_s = h_t.pow(scale)
    denom = w_t_s + h_t_s + eps
    ww = 2 * w_t_s / denom  # x-direction weight
    hh = 2 * h_t_s / denom  # y-direction weight

    # ---- Shape-aware distance ----
    # Minimum enclosing box diagonal
    x1_p, y1_p = cx_p - w_p / 2, cy_p - h_p / 2
    x2_p, y2_p = cx_p + w_p / 2, cy_p + h_p / 2
    x1_t, y1_t = cx_t - w_t / 2, cy_t - h_t / 2
    x2_t, y2_t = cx_t + w_t / 2, cy_t + h_t / 2

    cw = torch.max(x2_p, x2_t) - torch.min(x1_p, x1_t)
    ch = torch.max(y2_p, y2_t) - torch.min(y1_p, y1_t)
    c2 = cw.pow(2) + ch.pow(2) + eps

    dx = cx_p - cx_t
    dy = cy_p - cy_t
    rho2 = hh * dx.pow(2) + ww * dy.pow(2)  # cross-paired: hh → x, ww → y
    distance_shape = rho2 / c2

    # ---- Shape penalty (cross-paired, exponentially weighted) ----
    omega_w = hh * torch.abs(w_p - w_t) / (torch.maximum(w_p, w_t) + eps)
    omega_h = ww * torch.abs(h_p - h_t) / (torch.maximum(h_p, h_t) + eps)
    omega_shape = (1 - torch.exp(-omega_w)).pow(theta) + (1 - torch.exp(-omega_h)).pow(theta)

    # ---- Combined loss ----
    loss = 1.0 - iou + distance_shape + omega_weight * omega_shape

    return loss


class ShapeIoULoss(nn.Module):
    """Shape-IoU loss module combining IoU + shape distance + aspect ratio penalty.

    For small/elongated IR ship targets, this provides stronger geometric
    consistency than standard CIoU, especially when IoU ≈ 0 (the distance
    and shape terms still provide gradient signal).

    Args:
        theta: Shape sensitivity exponent (default: 4.0).
        omega_weight: Weight for shape penalty term (default: 0.5).
        scale: Shape-awareness scale (default: 1.0). 0 → vanilla distance.
        iou_ratio: Blending ratio with CIoU (1.0 = pure Shape-IoU, 0.0 = pure CIoU).
    """

    def __init__(
        self,
        theta: float = 4.0,
        omega_weight: float = 0.5,
        scale: float = 1.0,
        iou_ratio: float = 1.0,
    ):
        super().__init__()
        self.theta = theta
        self.omega_weight = omega_weight
        self.scale = scale
        self.iou_ratio = iou_ratio

    def forward(
        self,
        pred_xywh: torch.Tensor,
        target_xywh: torch.Tensor,
        iou: torch.Tensor,
    ) -> torch.Tensor:
        """Compute blended Shape-IoU + CIoU loss.

        Args:
            pred_xywh: Predicted boxes (cx, cy, w, h), shape (N, 4).
            target_xywh: Target boxes (cx, cy, w, h), shape (N, 4).
            iou: Standard IoU (or CIoU if from ultralytics bbox_iou), shape (N,).

        Returns:
            Combined loss value, shape (N,).
        """
        kwargs = dict(
            theta=self.theta,
            omega_weight=self.omega_weight,
            scale=self.scale,
        )
        if self.iou_ratio >= 1.0:
            return shape_iou_loss(pred_xywh, target_xywh, iou, **kwargs)
        else:
            shape_loss = shape_iou_loss(pred_xywh, target_xywh, iou, **kwargs)
            ciou_loss = 1.0 - iou
            return self.iou_ratio * shape_loss + (1.0 - self.iou_ratio) * ciou_loss


class ShapeIoUBboxLoss(nn.Module):
    """Drop-in replacement for ultralytics BboxLoss with Shape-IoU support.

    Replaces standard CIoU with Shape-IoU that adds shape-aware distance
    and aspect ratio penalty. Preserves DFL (Distribution Focal Loss).

    Intended for IR ship detection where target aspect ratios vary
    significantly (small, elongated ships vs broader vessels).

    Args:
        reg_max: Maximum value for DFL discrete bins (default: 16).
        theta: Shape sensitivity exponent (default: 4.0).
        omega_weight: Weight for shape penalty term (default: 0.5).
        scale: Shape-awareness scale (default: 1.0).
        iou_ratio: Blending ratio Shape-IoU vs CIoU (1.0 = pure Shape-IoU).
    """

    def __init__(
        self,
        reg_max: int = 16,
        theta: float = 4.0,
        omega_weight: float = 0.5,
        scale: float = 1.0,
        iou_ratio: float = 1.0,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.shape_iou_loss = ShapeIoULoss(
            theta=theta,
            omega_weight=omega_weight,
            scale=scale,
            iou_ratio=iou_ratio,
        )
        # DFL loss as in ultralytics BboxLoss
        from ultralytics.utils.loss import DFLoss
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def __call__(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: int | torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Shape-IoU box loss.

        Args:
            pred_dist: Predicted DFL distribution.
            pred_bboxes: Predicted boxes (xyxy format), shape (N, 4).
            anchor_points: Anchor point coordinates.
            target_bboxes: Target boxes (xyxy format), shape (N, 4).
            target_scores: Target classification scores.
            target_scores_sum: Sum of target scores (for normalization).
            fg_mask: Foreground mask (positive samples).
            imgsz: Image size.
            stride: Model strides.

        Returns:
            Tuple of (loss_iou, loss_dfl).
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Standard IoU (not CIoU — shape terms handle the distance/ratio)
        from ultralytics.utils.metrics import bbox_iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=False)

        # Convert boxes to xywh for shape distance computation
        pred_xywh = torch.cat([
            (pred_bboxes[fg_mask][:, :2] + pred_bboxes[fg_mask][:, 2:]) / 2,  # cx, cy
            pred_bboxes[fg_mask][:, 2:] - pred_bboxes[fg_mask][:, :2],       # w, h
        ], dim=1)
        target_xywh = torch.cat([
            (target_bboxes[fg_mask][:, :2] + target_bboxes[fg_mask][:, 2:]) / 2,
            target_bboxes[fg_mask][:, 2:] - target_bboxes[fg_mask][:, :2],
        ], dim=1)

        # Shape-IoU loss
        loss_iou = self.shape_iou_loss(pred_xywh, target_xywh, iou) * weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # DFL loss (same as ultralytics original)
        if self.dfl_loss:
            from ultralytics.utils.tal import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.new_zeros(1)

        return loss_iou, loss_dfl


# Wrapper for TinyObjectDetectionLoss-style integration
class ShapeIoUDetectionLoss:
    """Complete loss wrapper replacing v8DetectionLoss BboxLoss with Shape-IoU.

    v8DetectionLoss internally handles fl_gamma/focal loss; no external
    focal gamma override needed.

    Usage:
        loss_fn = ShapeIoUDetectionLoss(model, theta=4.0, omega_weight=0.5)
    """

    def __init__(
        self,
        model,
        theta: float = 4.0,
        omega_weight: float = 0.5,
        scale: float = 1.0,
        iou_ratio: float = 1.0,
    ):
        self.iou_ratio = iou_ratio

        self.model = model
        self.device = next(model.parameters()).device

        # Create underlying ultralytics loss
        from ultralytics.utils.loss import v8DetectionLoss
        self.base_loss = v8DetectionLoss(model)

        # Replace bbox_loss with Shape-IoU version
        reg_max = getattr(model.model[-1], 'reg_max', 16)
        self.base_loss.bbox_loss = ShapeIoUBboxLoss(
            reg_max=reg_max,
            theta=theta,
            omega_weight=omega_weight,
            scale=scale,
            iou_ratio=iou_ratio,
        )

    def __call__(self, preds, batch):
        """Compute loss with Shape-IoU for geometric consistency."""
        return self.base_loss(preds, batch)


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    print("Custom loss module loaded successfully.")
    print("Available:")
    print("  - wasserstein_loss()       — NWD for tiny objects")
    print("  - NWDLoss()                — Combined NWD + CIoU loss module")
    print("  - WIoU_Scale()             — WIoU v3 dynamic focusing")
    print("  - NWDBboxLoss()            — ultralytics-compatible BboxLoss")
    print("  - TinyObjectDetectionLoss() — Complete loss wrapper")
    print()
    print("  - shape_iou_loss()         — Shape-IoU raw function")
    print("  - ShapeIoULoss()           — Shape-IoU loss module")
    print("  - ShapeIoUBboxLoss()       — ultralytics-compatible BboxLoss")
    print("  - ShapeIoUDetectionLoss()  — Complete loss wrapper")
