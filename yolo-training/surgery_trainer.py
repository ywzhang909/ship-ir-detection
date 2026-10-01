"""Apply architecture surgery at the only point where it survives training.

Why this module exists
----------------------
When Ultralytics training starts from a ``.pt`` checkpoint,
``BaseTrainer.setup_model()`` discards whatever model object the caller
built and constructs a brand-new one from the checkpoint's YAML
(``ultralytics/engine/trainer.py``)::

    if str(self.model).endswith(".pt"):
        weights, ckpt = load_checkpoint(self.model)
        cfg = weights.yaml
    self.model = self.get_model(cfg=cfg, weights=weights, verbose=...)

``get_model()`` builds ``DetectionModel(cfg)`` — a stock ``yolo11m.yaml``
detector — and then transfers pretrained weights by name/shape intersection.

Consequence: any in-place surgery applied to the ``YOLO(...)`` object
*before* ``.train()`` is silently thrown away.  ``train.py`` used to apply
``--backbone`` / ``--attention`` / ``--neck`` / ``--head`` that way and then
logged a success banner, so those runs trained *and saved* a plain YOLO11m
while the logs claimed the replacement had worked.  Concretely, the
``starnet-s1`` and ``leconv-t`` runs both produced ``best.pt`` checkpoints
that were stock YOLO11m (20,059,947 params, ``yaml_file=yolo11m.yaml``, zero
custom modules) — i.e. duplicate baselines rather than the requested
architectures.

The fusion path already hit this and worked around it by overriding
``get_model()`` (see :func:`fusion_module.make_fusion_trainer_class`).
This module generalises that fix to every surgery flag.

Ordering
--------
``build -> surgery -> load(weights) -> fusion`` so that retained
sub-modules (stem, SPPF, C2PSA, neck, head) still inherit their pretrained
weights through ``intersect_dicts``; only the genuinely replaced blocks
train from scratch.
"""

from __future__ import annotations

import logging
from typing import Any

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

__all__ = ["SurgerySpec", "apply_surgery", "make_surgery_trainer_class", "needs_surgery"]

logger = logging.getLogger("train")

#: Free-form mapping describing which surgeries to apply.
SurgerySpec = dict[str, Any]


def needs_surgery(spec: SurgerySpec) -> bool:
    """Return ``True`` when at least one architecture surgery is requested."""
    return any(spec.get(key) for key in ("attention_type", "neck_type", "backbone", "head_type"))


def apply_surgery(model: DetectionModel, spec: SurgerySpec) -> dict[str, Any]:
    """Apply every requested surgery to ``model`` in place.

    Each replacement helper accepts either a ``YOLO`` wrapper or a bare
    ``DetectionModel``, so passing the freshly built model is supported.

    Args:
        model: Detection model built from the stock YAML, before weight loading.
        spec: Surgery selection, see :func:`needs_surgery` for the keys.

    Returns:
        Mapping of surgery kind to the number of replaced modules.  The
        ``head`` entry is a bool for backwards compatibility with the
        pre-existing "replaced / failed" reporting.
    """
    applied: dict[str, Any] = {}

    attention_type = spec.get("attention_type")
    if attention_type:
        from attention_modules import insert_attention_into_backbone

        applied["attention"] = insert_attention_into_backbone(model, attention_type)
        logger.info("Surgery: inserted %d attention module(s) (%s)", applied["attention"], attention_type)

    neck_type = spec.get("neck_type")
    if neck_type == "c2ema":
        from c2ema import replace_neck_with_c2ema

        applied["neck"] = replace_neck_with_c2ema(model)
    elif neck_type == "asff":
        from asff import replace_concat_with_asff

        applied["neck"] = replace_concat_with_asff(model)
    elif neck_type == "c3k3":
        from c3k3 import replace_c3k2_with_c3k

        applied["neck"] = replace_c3k2_with_c3k(model, kernel_size=3)
    if neck_type:
        logger.info("Surgery: replaced %s neck component(s) (%s)", applied["neck"], neck_type)

    backbone = spec.get("backbone")
    if backbone == "starnet":
        from starnet_backbone import replace_backbone_with_starnet

        applied["backbone"] = replace_backbone_with_starnet(model)
    elif backbone == "leconv":
        from leconv_backbone import replace_backbone_with_leconv

        applied["backbone"] = replace_backbone_with_leconv(model)
    if backbone:
        logger.info("Surgery: replaced %s backbone block(s) (%s)", applied["backbone"], backbone)

    head_type = spec.get("head_type")
    if head_type == "dynamic":
        from dynamic_head import replace_detect_with_dynamic_head

        applied["head"] = bool(replace_detect_with_dynamic_head(model))
        logger.info("Surgery: dynamic head replacement=%s", applied["head"])

    return applied


def make_surgery_trainer_class(
    base_trainer_cls: type,
    spec: SurgerySpec,
    fusion_kwargs: dict[str, Any] | None = None,
) -> type:
    """Create a trainer subclass that applies ``spec`` inside ``get_model()``.

    Args:
        base_trainer_cls: Trainer to subclass.  Pass
            :class:`~fusion_module.make_fusion_trainer_class`'s result when
            fusion is enabled so its optimizer/scheduler overrides are kept.
        spec: Surgery selection, see :func:`needs_surgery`.
        fusion_kwargs: When given, the fusion module is inserted after weight
            loading (matching the historical order) and cached on ``self._fusion``
            so the inherited ``build_optimizer()`` LR scaling still works.

    Returns:
        A ``DetectionTrainer`` subclass whose ``get_model()`` honours the surgery.
    """

    class SurgeryDetectionTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        _surgery_spec: SurgerySpec = spec

        def get_model(self, cfg=None, weights=None, verbose=True):
            """Build the stock model, apply surgery, then load pretrained weights."""
            model = DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
            applied = apply_surgery(model, self._surgery_spec)
            logger.info("SurgeryDetectionTrainer: surgery applied: %s", applied or "none")
            if weights:
                model.load(weights)
            if fusion_kwargs is not None:
                from fusion_module import (
                    SpatialFrequencyFusion,
                    _insert_fusion_into_detection_model,
                )

                fusion = SpatialFrequencyFusion(**fusion_kwargs)
                _insert_fusion_into_detection_model(model, fusion)
                self._fusion = fusion
                logger.info(
                    "SurgeryDetectionTrainer: inserted fusion (%d params)",
                    sum(p.numel() for p in fusion.parameters()),
                )
            return model

    SurgeryDetectionTrainer.__name__ = f"Surgery{base_trainer_cls.__name__}"
    SurgeryDetectionTrainer.__qualname__ = SurgeryDetectionTrainer.__name__
    return SurgeryDetectionTrainer
