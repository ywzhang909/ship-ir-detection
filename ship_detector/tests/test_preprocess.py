"""Contract tests for training-domain input alignment (core/preprocess.py)."""

from __future__ import annotations

import numpy as np
import pytest

from core.preprocess import (
    KNOWN_PREPS,
    apply_prep,
    is_prep_available,
    normalize_ir,
    preprocess_frame,
)

cv2 = pytest.importorskip("cv2")


def test_normalize_ir_collapses_colour_to_single_band():
    """BGR→GRAY→BGR must yield a neutral image (R==G==B)."""
    img = np.zeros((16, 16, 3), np.uint8)
    img[:, :8] = (200, 10, 10)      # strongly red-ish
    img[:, 8:] = (10, 10, 200)      # strongly blue-ish

    out = normalize_ir(img)

    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_normalize_ir_accepts_gray_and_bgra():
    gray = np.full((8, 8), 120, np.uint8)
    assert normalize_ir(gray).shape == (8, 8, 3)

    bgra = np.full((8, 8, 4), 90, np.uint8)
    assert normalize_ir(bgra).shape == (8, 8, 3)


def test_normalize_ir_is_idempotent_on_neutral_input():
    neutral = np.full((8, 8, 3), 77, np.uint8)
    assert np.array_equal(normalize_ir(neutral), neutral)


def test_normalize_ir_rejects_bad_channel_count():
    with pytest.raises(ValueError):
        normalize_ir(np.zeros((4, 4, 5), np.uint8))


def test_apply_prep_raw_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (12, 12, 3), dtype=np.uint8)
    assert np.array_equal(apply_prep(img, "raw"), img)


def test_apply_prep_rejects_unknown_name():
    with pytest.raises(ValueError):
        apply_prep(np.zeros((4, 4, 3), np.uint8), "not_a_prep")


def test_preprocess_frame_preserves_geometry_for_all_preps():
    """Geometry must be preserved or box coords could not map back to the frame.

    The UI and the video exporter draw boxes computed on the PREPROCESSED image
    onto the ORIGINAL frame, so any resize here would silently misplace every box.
    """
    img = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    for prep in KNOWN_PREPS:
        if not is_prep_available(prep):
            pytest.skip(f"preprocessing backend unavailable for {prep!r}")
        out = preprocess_frame(img, prep)
        assert out.shape == img.shape, f"{prep}: {out.shape} != {img.shape}"
        assert out.dtype == np.uint8, f"{prep}: dtype {out.dtype}"


def test_preprocess_frame_can_skip_normalization():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :, 2] = 200
    out = preprocess_frame(img, "raw", normalize=False)
    assert np.array_equal(out, img)


def test_raw_prep_needs_no_backend():
    assert is_prep_available("raw") is True


def test_detector_exposes_and_validates_preprocess():
    from core.yolo_detector import YoloDetector

    det = YoloDetector()
    assert det.preprocess == "raw"
    det.set_preprocess("dual")
    assert det.preprocess == "dual"
    with pytest.raises(ValueError):
        det.set_preprocess("bogus")
    assert det.preprocess == "dual", "failed set must not corrupt state"
