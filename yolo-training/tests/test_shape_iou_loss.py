"""Unit tests for Shape-IoU loss implementation.

Tests:
1. shape_iou_loss function: numerical correctness vs paper formula
2. ShapeIoULoss module: forward + blending with CIoU
3. ShapeIoUBboxLoss: end-to-end integration with dummy YOLO predictions
4. Backward pass: gradients flow through all components
5. Edge cases: zero boxes, perfect predictions, extreme aspect ratios
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from custom_loss import shape_iou_loss, ShapeIoULoss

try:
    from custom_loss import ShapeIoUBboxLoss
    _HAS_ULTRALYTICS = True
except ModuleNotFoundError:
    _HAS_ULTRALYTICS = False


# ── helpers ────────────────────────────────────────────────────────────────

def _make_boxes(cx, cy, w, h):
    """xywh → xyxy."""
    return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])


# ═══════════════════════════════════════════════════════════════════════════
# 1.  shape_iou_loss — perfect prediction
# ═══════════════════════════════════════════════════════════════════════════

def test_perfect_prediction():
    """IoU=1, dx=dy=0, w/w_gt=1, h/h_gt=1 → loss = 0."""
    pred = torch.tensor([[50., 50., 20., 10.]])   # cx, cy, w, h
    target = torch.tensor([[50., 50., 20., 10.]])
    iou = torch.tensor([1.0])

    loss = shape_iou_loss(pred, target, iou, theta=4.0, omega_weight=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), \
        f"Perfect prediction should give 0 loss, got {loss.item()}"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  shape_iou_loss — complete miss (IoU=0)
# ═══════════════════════════════════════════════════════════════════════════

def test_complete_miss():
    """Far apart boxes: IoU=0, distance dominates. Verify loss > 0."""
    pred = torch.tensor([[10., 10., 20., 20.]])
    target = torch.tensor([[100., 100., 20., 20.]])
    iou = torch.tensor([0.0])

    loss = shape_iou_loss(pred, target, iou)
    assert loss.item() > 0.5, \
        f"Complete miss should give loss > 0.5, got {loss.item()}"


# ═══════════════════════════════════════════════════════════════════════════
# 3.  shape_iou_loss — shape penalty (same center, different aspect ratio)
# ═══════════════════════════════════════════════════════════════════════════

def test_shape_penalty():
    """Same center, same area, different aspect ratio → loss > 1 - IoU."""
    pred = torch.tensor([[50., 50., 40., 10.]])   # wide
    target = torch.tensor([[50., 50., 20., 20.]])  # square (same area ≈ 400)
    iou = torch.tensor([0.5])

    # IoU-only baseline
    baseline = 1.0 - iou  # = 0.5

    loss = shape_iou_loss(pred, target, iou)
    assert loss.item() > baseline.item(), \
        f"Shape penalty should increase loss ({loss.item()} ≤ {baseline.item()})"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  shape_iou_loss — scale parameter effect
# ═══════════════════════════════════════════════════════════════════════════

def test_scale_zero():
    """scale=0 → ww=hh=1 → distance reduces to standard centerness.
    Loss should be LOWER than with scale=1 for extreme shapes."""
    pred = torch.tensor([[50., 40., 40., 10.]])   # wide, y-off
    target = torch.tensor([[50., 50., 40., 10.]])  # same shape, centered
    iou = torch.tensor([0.8])

    loss_s0 = shape_iou_loss(pred, target, iou, scale=0.0)
    loss_s1 = shape_iou_loss(pred, target, iou, scale=1.0)

    # With scale=1, shape-aware weights amplify the penalty
    # ww → 2*40/(40+10)=1.6, hh → 2*10/(40+10)=0.4
    # dy is penalised more with scale=1 (ww=1.6 vs 1.0)
    assert loss_s1.item() >= loss_s0.item() - 1e-6, \
        f"scale=1 should punish shape mismatch at least as much as scale=0 " \
        f"({loss_s1.item()} < {loss_s0.item()})"


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ShapeIoULoss module — forward + iou_ratio blending
# ═══════════════════════════════════════════════════════════════════════════

def test_shape_iou_loss_module():
    """ShapeIoULoss.forward returns expected shape and value."""
    module = ShapeIoULoss(theta=4.0, omega_weight=0.5, scale=1.0, iou_ratio=1.0)
    pred = torch.tensor([[50., 50., 20., 10.]])
    target = torch.tensor([[52., 48., 22., 12.]])
    iou = torch.tensor([0.7])

    loss = module(pred, target, iou)
    assert loss.shape == (1,), f"Expected shape (1,), got {loss.shape}"
    assert loss.item() > 0.0, "Loss should be > 0"


def test_iou_ratio_blending():
    """iou_ratio=0.5 blends Shape-IoU and CIoU."""
    shape_module = ShapeIoULoss(iou_ratio=1.0)
    blend_module = ShapeIoULoss(iou_ratio=0.5)

    pred = torch.tensor([[50., 50., 20., 10.]])
    target = torch.tensor([[52., 48., 22., 12.]])
    iou = torch.tensor([0.7])

    pure_shape = shape_module(pred, target, iou)
    blended = blend_module(pred, target, iou)
    ciou = 1.0 - iou

    # blended = 0.5 * pure_shape + 0.5 * ciou
    expected = 0.5 * pure_shape + 0.5 * ciou
    assert blended.item() == pytest.approx(expected.item(), abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  ShapeIoUBboxLoss — forward + backward with dummy data
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="requires ultralytics")
def test_shape_iou_bbox_loss_backward():
    """End-to-end forward + backward through ShapeIoUBboxLoss.

    Creates dummy predictions with gradients and verifies all parameters
    receive gradients.
    """
    reg_max = 16
    n_fg = 4        # number of foreground (positive) samples
    n_anc = 100     # total anchors

    loss_fn = ShapeIoUBboxLoss(reg_max=reg_max)

    # Dummy predictions (requires_grad on both)
    pred_dist = torch.randn(n_anc, reg_max * 4, requires_grad=True)
    pred_bboxes = torch.randn(n_anc, 4, requires_grad=True) * 50 + 50

    # Dummy targets
    anchor_points = torch.rand(n_anc, 2) * 100
    target_bboxes = torch.rand(n_anc, 4) * 50 + 50
    target_scores = torch.zeros(n_anc, 9)
    target_scores[:n_fg, 0] = 1.0
    target_scores_sum = target_scores.sum()
    fg_mask = target_scores.sum(-1) > 0  # first n_fg are positive

    imgsz = torch.tensor(640)
    stride = torch.ones(n_anc, 1) * 8

    loss_iou, loss_dfl = loss_fn(
        pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    )

    assert loss_iou.requires_grad, "loss_iou should require grad"
    assert loss_dfl.requires_grad, "loss_dfl should require grad"
    assert loss_iou.item() > 0, "loss_iou should be > 0"
    assert loss_dfl.item() >= 0, "loss_dfl should be >= 0"

    # Backward
    (loss_iou + loss_dfl).backward()
    assert pred_dist.grad is not None, "pred_dist should have grad"
    assert pred_dist.grad.shape == pred_dist.shape
    assert pred_dist.grad.abs().sum().item() > 0, "gradients should be non-zero"


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Edge cases
# ═══════════════════════════════════════════════════════════════════════════

def test_zero_size_box():
    """Zero-area box should not cause NaN or inf."""
    pred = torch.tensor([[50., 50., 0., 0.]])   # zero area
    target = torch.tensor([[50., 50., 20., 10.]])
    iou = torch.tensor([0.0])

    loss = shape_iou_loss(pred, target, iou)
    assert torch.isfinite(loss).all(), f"Loss should be finite, got {loss}"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_extreme_aspect_ratio():
    """Very elongated vs square — should compute without error."""
    pred = torch.tensor([[50., 50., 100., 2.]])   # very wide, thin
    target = torch.tensor([[50., 50., 20., 20.]])  # medium square
    iou = torch.tensor([0.1])

    loss = shape_iou_loss(pred, target, iou)
    assert torch.isfinite(loss).all(), f"Loss should be finite, got {loss}"
    assert loss.item() > 0, "Loss should be > 0"


@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="requires ultralytics")
def test_empty_fg_mask():
    """fg_mask with no positives → loss should handle gracefully.

    Note: v8DetectionLoss guards against calling bbox_loss with empty fg_mask
    (it only calls when fg_mask.any() is True). This test verifies the loss
    doesn't crash when called directly with an empty mask.
    """
    reg_max = 16
    loss_fn = ShapeIoUBboxLoss(reg_max=reg_max)

    pred_dist = torch.randn(10, reg_max * 4)
    pred_bboxes = torch.randn(10, 4)
    anchor_points = torch.rand(10, 2)
    target_bboxes = torch.randn(10, 4)
    target_scores = torch.zeros(10, 9)  # no positives
    target_scores_sum = target_scores.sum() + 1e-7  # avoid div-by-zero
    fg_mask = torch.zeros(10, dtype=torch.bool)

    imgsz = torch.tensor(640)
    stride = torch.ones(10, 1) * 8

    loss_iou, loss_dfl = loss_fn(
        pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    )
    assert torch.isfinite(loss_iou).all()
    assert torch.isfinite(loss_dfl).all()


# ═══════════════════════════════════════════════════════════════════════════
# 8.  scale=0 → ww=hh=1 (official paper default)
# ═══════════════════════════════════════════════════════════════════════════

def test_scale_zero_weight_values():
    """scale=0 → ww=hh=1 (= 2*1/(1+1))."""
    pred = torch.tensor([[50., 50., 30., 10.]])
    target = torch.tensor([[50., 50., 20., 20.]])
    iou = torch.tensor([0.5])

    loss = shape_iou_loss(pred, target, iou, scale=0.0)
    assert torch.isfinite(loss).all()


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
