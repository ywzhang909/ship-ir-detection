"""
Learnable Spatial–Frequency Fusion Network for IR Maritime Ship Detection.

Architecture (default 2-layer)
────────────
    Input:  3-channel [spatial_domain │ frequency_domain │ original_grayscale]   H×W×3
              │
    ┌─────────┴─────────┐
    │  Conv1(3→h, k=3) + BN + SiLU       ← project to hidden
    │  Conv2(h→h, k=3) + BN + SiLU       ← non-linear mixing (1 or more layers)
    │  Attention(h, r=4)                 ← SE or CBAM channel/spatial re-weight
    │  ConvN(h→3, k=3)                   ← reconstruct 3-ch fused
    │  + residual (learnable alpha)       ← skip connection
    └─────────┬─────────┘
              │
    Output: 3-channel fused image (same spatial size)  →  YOLO first Conv

Architecture search knobs:
  - hidden ∈ {16, 32, 64}         – capacity
  - num_layers ∈ {2, 3, 4}        – depth  (total Conv blocks before attention)
  - attention ∈ {"se", "cbam"}    – attention type
  - residual_alpha_init ∈ [0, 1]  – initial skip strength

The module is lightweight (< 15k parameters even at h=64, 4 layers) so it adds
negligible overhead during joint training with YOLO.

Usage:
    from fusion_module import SpatialFrequencyFusion, insert_fusion_into_model

    model = YOLO("yolo11m.pt")
    fusion = SpatialFrequencyFusion(in_channels=3, hidden=16)
    insert_fusion_into_model(model, fusion)   # model.model[0] → Sequential(Fusion, Conv)

Reference:
    Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Channel Attention (Squeeze-and-Excitation)
# ===================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel-wise attention.

    Adaptively recalibrates channel-wise feature responses by
    squeezing global spatial information into a channel descriptor
    then using it to excite each channel.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(4, channels // reduction), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ===================================================================
# CBAM: Convolutional Block Attention Module
# ===================================================================

class ChannelAttentionCBAM(nn.Module):
    """CBAM channel attention: uses both avg-pool and max-pool branches."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(4, channels // reduction), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttentionCBAM(nn.Module):
    """CBAM spatial attention: concat avg/max pooling → conv → sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel → spatial.

    Applies channel attention first, then spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttentionCBAM(channels, reduction)
        self.spatial_attn = SpatialAttentionCBAM(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ===================================================================
# Attention factory
# ===================================================================

def _build_attention(attn_type: str, channels: int, reduction: int = 4) -> nn.Module:
    if attn_type == "se":
        return ChannelAttention(channels, reduction)
    elif attn_type == "cbam":
        return CBAM(channels, reduction)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}, expected 'se' or 'cbam'")


# ===================================================================
# Conv-BN-SiLU block
# ===================================================================

class _ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)), inplace=False)


# ===================================================================
# Spatial–Frequency Fusion Module
# ===================================================================

class SpatialFrequencyFusion(nn.Module):
    """Learnable fusion network for spatial-domain and frequency-domain
    preprocessed IR imagery.

    Takes 3 input channels: [spatial, frequency, original_grayscale]
    and produces a fused 3-channel output via a lightweight CNN with
    channel attention and residual connection.

    The module preserves spatial resolution (stride=1) and channel
    count (3→3) so it can be inserted transparently before YOLO's
    first convolution layer.

    Architecture search parameters:
        hidden:         Width of hidden layers {16, 32, 64}
        num_layers:     Number of Conv blocks before attention {2, 3, 4}
        attention:      Attention type {"se", "cbam"}
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 16,
        out_channels: int = 3,
        kernel_size: int = 3,
        reduction: int = 4,
        residual_alpha_init: float = 0.5,
        num_layers: int = 2,
        attention: str = "se",
    ):
        super().__init__()

        # ── Feature extraction blocks ──
        layers: list[nn.Module] = []
        # First block: in_channels → hidden
        layers.append(_ConvBlock(in_channels, hidden, kernel_size))
        # Middle blocks (num_layers - 1): hidden → hidden
        for _ in range(num_layers - 1):
            layers.append(_ConvBlock(hidden, hidden, kernel_size))
        self.blocks = nn.Sequential(*layers)

        # ── Attention ──
        self.attention = _build_attention(attention, hidden, reduction)

        # ── Reconstruction ──
        self.conv_out = nn.Conv2d(hidden, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

        # ── Learnable residual scaling ──
        self.residual_alpha = nn.Parameter(torch.tensor(residual_alpha_init))

        self._init_weights()

    def _init_weights(self):
        """Initialize conv weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W) — [spatial, freq, original]

        Returns:
            Fused tensor (B, 3, H, W).
        """
        identity = x  # (B, 3, H, W)

        x = self.blocks(x)
        x = self.attention(x)
        x = self.conv_out(x)  # (B, 3, H, W)

        # Residual: fused + α * input
        out = x + self.residual_alpha * identity

        return out


# ===================================================================
# Model Integration Helpers
# ===================================================================

def _insert_fusion_into_detection_model(
    model: "torch.nn.Module",
    fusion_module: SpatialFrequencyFusion,
) -> None:
    """Insert fusion module into a raw DetectionModel (``model.model`` of YOLO).

    Wraps ``model.model[0]`` from ``Conv(3→ch, k=3, s=2)`` into
    ``Sequential(SpatialFrequencyFusion(3→3, s=1), Conv(3→ch, k=3, s=2))``.

    Args:
        model: A DetectionModel (ultralytics.nn.tasks.DetectionModel).
        fusion_module: Initialised SpatialFrequencyFusion instance.
    """
    original_first = model.model[0]
    model.model[0] = nn.Sequential(fusion_module, original_first)

    device = next(original_first.parameters()).device
    dtype = next(original_first.parameters()).dtype
    model.model[0].to(device=device, dtype=dtype)

    logger.info(
        "Inserted SpatialFrequencyFusion (%s params) before YOLO first Conv",
        sum(p.numel() for p in fusion_module.parameters()),
    )


def insert_fusion_into_model(model, fusion_module: SpatialFrequencyFusion) -> None:
    """Insert a SpatialFrequencyFusion module before YOLO's first conv layer.

    Convenience wrapper that unwraps the YOLO object and delegates to
    :func:`_insert_fusion_into_detection_model`.

    Args:
        model: A loaded YOLO object (ultralytics.YOLO).
        fusion_module: Initialised SpatialFrequencyFusion instance.
    """
    _insert_fusion_into_detection_model(model.model, fusion_module)


def create_fusion_model(
    model_name: str = "yolo11m.pt",
    hidden: int = 16,
    num_layers: int = 2,
    attention: str = "se",
    **kwargs,
):
    """Convenience factory: load YOLO model and insert fusion module.

    Args:
        model_name: YOLO model name or path.
        hidden: Hidden dimension of the fusion module.
        num_layers: Number of Conv blocks before attention {2, 3, 4}.
        attention: Attention type {"se", "cbam"}.
        **kwargs: Extra args passed to SpatialFrequencyFusion.

    Returns:
        (model, fusion_module) tuple, both on same device.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    fusion = SpatialFrequencyFusion(
        in_channels=3, hidden=hidden,
        num_layers=num_layers, attention=attention,
        **kwargs,
    )
    insert_fusion_into_model(model, fusion)
    return model, fusion


# ===================================================================
# Custom Trainer — injects fusion module at the right point in the
# Ultralytics training loop (after model is created from YAML but
# before training starts)
# ===================================================================


def make_fusion_trainer_class(
    base_trainer_cls: type,
    fusion_kwargs: dict[str, Any],
    fusion_lr_scale: float = 1.0,
    residual_warmup_epochs: int = 10,
    residual_alpha_target: float = 0.5,
) -> type:
    """Create a trainer subclass that inserts fusion in ``get_model()``.

    Overrides:

    * ``get_model()`` — insert fusion module after model creation
    * ``build_optimizer()`` — separate param group with ``fusion_lr_scale``
    * ``_setup_train()`` — attach residual-α warmup callback

    Args:
        base_trainer_cls: The trainer class to subclass (usually
            ``ultralytics.models.yolo.detect.DetectionTrainer``).
        fusion_kwargs: Keyword arguments forwarded to
            :class:`SpatialFrequencyFusion` (includes hidden, num_layers,
            attention, reduction, etc.).
        fusion_lr_scale: Multiplier for the fusion module's learning rate
            *vs.* the backbone (default 1.0 = same LR).
        residual_warmup_epochs: Number of epochs over which to linearly
            ramp ``residual_alpha`` from 0.0 → ``residual_alpha_target``.
        residual_alpha_target: Final value of ``residual_alpha`` after warmup.

    Returns:
        A trainer subclass with the overrides described above.
    """

    class FusionDetectionTrainer(base_trainer_cls):  # type: ignore[valid-type]
        def __init__(self, *args, **kwargs):
            self._fusion_lr_scale = fusion_lr_scale
            self._fusion = None  # will be set by get_model()
            super().__init__(*args, **kwargs)

        # ── model creation ────────────────────────────────────────────

        def get_model(self, cfg=None, weights=None, verbose=True):
            """Build DetectionModel, load weights, insert fusion."""
            model = super().get_model(cfg, weights, verbose)
            fusion = SpatialFrequencyFusion(**fusion_kwargs)
            _insert_fusion_into_detection_model(model, fusion)
            self._fusion = fusion
            logger.info(
                "FusionDetectionTrainer: inserted SpatialFrequencyFusion (%s params)",
                sum(p.numel() for p in fusion.parameters()),
            )
            return model

        # ── parameter-group LR scaling ────────────────────────────────

        def build_optimizer(self):
            """Build optimizer; separate fusion params with higher LR."""
            opt = super().build_optimizer()

            if self._fusion_lr_scale == 1.0 or self._fusion is None:
                return opt

            # Collect fusion parameter ids
            fusion_param_ids = {id(p) for p in self._fusion.parameters()}

            # Reorganise existing groups — move fusion params into new groups
            new_groups: list[dict] = []
            for group in opt.param_groups:
                fusion_params = [p for p in group["params"] if id(p) in fusion_param_ids]
                backbone_params = [p for p in group["params"] if id(p) not in fusion_param_ids]

                if backbone_params:
                    bg = dict(group)
                    bg["params"] = backbone_params
                    new_groups.append(bg)

                if fusion_params:
                    fg = dict(group)
                    fg["params"] = fusion_params
                    fg["lr"] *= self._fusion_lr_scale
                    new_groups.append(fg)

            # Replace param groups
            opt.param_groups.clear()
            for g in new_groups:
                opt.param_groups.append(g)

            return opt

        # ── residual-α warmup ─────────────────────────────────────────

        def _setup_train(self, *args, **kwargs):
            """Set up training and attach residual-α warmup callback."""
            super()._setup_train(*args, **kwargs)

            if self._fusion is None or residual_warmup_epochs <= 0:
                return

            # Initialise to zero so the backbone sees clean input early on
            with torch.no_grad():
                self._fusion.residual_alpha.fill_(0.0)

            # Attach a per-epoch callback that ramps α linearly
            start_alpha = 0.0
            delta_per_epoch = (residual_alpha_target - start_alpha) / max(residual_warmup_epochs, 1)

            def _warmup_callback(trainer):
                if trainer.epoch >= residual_warmup_epochs:
                    return  # warmup complete
                new_alpha = start_alpha + delta_per_epoch * (trainer.epoch + 1)
                with torch.no_grad():
                    self._fusion.residual_alpha.fill_(min(new_alpha, residual_alpha_target))

            self.add_callback("on_train_epoch_start", _warmup_callback)
            logger.info(
                "Residual α warmup: 0.0 → %.2f over %d epochs",
                residual_alpha_target,
                residual_warmup_epochs,
            )

    return FusionDetectionTrainer
