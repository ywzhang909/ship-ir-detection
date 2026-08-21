"""Tests for the framework-free detection core (detector.py).

Heavy paths (real weights, CUDA) are exercised against the actual repo
artifacts; GPU-only assertions are skipped when CUDA is unavailable.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import detector
from detector import (
    Detection,
    LoadedModel,
    aggregate_metrics,
    compute_image_metrics,
    discover_models,
    gpu_display_name,
    load_model,
    predict_image,
    preprocess_for,
    read_image_bgr,
    resolve_device,
)

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs" / "detect" / "ship-detection"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# 1. Model discovery
# ---------------------------------------------------------------------------


def test_discover_models_defaults_to_t5():
    models = discover_models()
    assert models, "discover_models() returned no models"
    assert models[0].name == "T5_yolo11l_fusion"
    assert models[0].prep == "dual"
    assert "68.02" in models[0].label
    assert "[dual]" in models[0].label
    for m in models:
        assert m.path.exists(), f"missing weights: {m.path}"
        assert m.path.name == "best.pt"


# ---------------------------------------------------------------------------
# 2. Device resolution
# ---------------------------------------------------------------------------


@requires_cuda
def test_resolve_device_is_cuda():
    assert resolve_device() == "cuda:0"


def test_gpu_display_name_never_empty():
    name = gpu_display_name()
    assert isinstance(name, str) and name


# ---------------------------------------------------------------------------
# 3-5. Model loading: training input distribution (prep) routing
# ---------------------------------------------------------------------------


def test_load_smoke_model_is_raw():
    loaded = load_model(RUNS_DIR / "smoke_shape_iou" / "weights" / "best.pt")
    assert loaded.prep == "raw"
    assert loaded.device == resolve_device()


def test_load_t5_routes_dual():
    loaded = load_model(RUNS_DIR / "T5_yolo11l_fusion" / "weights" / "best.pt")
    assert loaded.prep == "dual"


def test_load_wavelet_run_routes_wavelet():
    loaded = load_model(RUNS_DIR / "T4_wavelet_dual" / "weights" / "best.pt")
    assert loaded.prep == "wavelet"


# ---------------------------------------------------------------------------
# 3b. prep classification from args.yaml + name fallback
# ---------------------------------------------------------------------------


def test_discover_models_classifies_prep(tmp_path):
    """Every prep category is resolved from args.yaml's data line; runs
    without args.yaml fall back to keyword rules on the run-dir name.
    Empty best.pt files are fine — discovery must not load weights."""
    cases = {
        # name: (data line, expected prep)
        "run_wavelet": ("D:/ds/_preprocessed_data_awb_wavelet_dual_fusion.yaml", "wavelet"),
        "run_rpca": ("D:/ds/_preprocessed_data_awb_rpca_dual_fusion.yaml", "rpca"),
        "run_dual": ("D:/ds/_preprocessed_data_awb_dual_fusion.yaml", "dual"),
        "run_tophat": ("D:/ds/_preprocessed_data_awb_tophat_clahe.yaml", "tophat"),
        "run_butterworth": ("D:/ds/_preprocessed_data_awb_butterworth_clahe.yaml", "butterworth"),
        "run_plain": ("D:/ds/configs/data.yaml", "raw"),
        # Fallbacks: no args.yaml at all -> classify the run-dir name.
        "fallback_wavelet_run": (None, "wavelet"),
        "fallback_plain_run": (None, "raw"),
    }
    for run_name, (data_line, _expected) in cases.items():
        weights_dir = tmp_path / run_name / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"")  # empty; never loaded
        if data_line is not None:
            (tmp_path / run_name / "args.yaml").write_text(
                f"data: {data_line}\n", encoding="utf-8"
            )

    models = discover_models(runs_dir=tmp_path, metrics_json=tmp_path / "metrics.json")

    by_name = {m.name: m for m in models}
    assert set(by_name) == set(cases)
    for run_name, (_data_line, expected_prep) in cases.items():
        assert by_name[run_name].prep == expected_prep, (
            f"{run_name}: expected prep={expected_prep}"
        )
        assert f"[{expected_prep}]" in by_name[run_name].label


# ---------------------------------------------------------------------------
# 6. End-to-end prediction on a synthetic image (GPU)
# ---------------------------------------------------------------------------


@requires_cuda
def test_predict_smoke_on_synthetic_image():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Bright white blob (~60x20 px), center-ish.
    img[300:320, 290:350] = 255

    loaded = load_model(RUNS_DIR / "smoke_shape_iou" / "weights" / "best.pt")
    detections, annotated, latency_ms = predict_image(
        loaded, img, conf=0.05, imgsz=640
    )

    # Structure, not count: synthetic blob may legitimately yield zero dets.
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert len(det.xyxy) == 4

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    assert annotated.ndim == 3 and annotated.shape[2] == 3
    assert latency_ms > 0

    metrics = compute_image_metrics("synthetic.png", detections, latency_ms)
    assert metrics["num_detections"] == len(detections)

    # GPU residency evidence: model parameters live on cuda after predict.
    assert next(loaded.model.model.parameters()).device.type == "cuda"


# ---------------------------------------------------------------------------
# 7. Unicode-safe image decoding
# ---------------------------------------------------------------------------


def test_read_image_bgr_unicode_safe(tmp_path):
    unicode_dir = tmp_path / "äöü_测试"
    unicode_dir.mkdir()
    img_path = unicode_dir / "img.png"

    small = np.full((24, 32, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", small)
    assert ok
    img_path.write_bytes(buf.tobytes())

    decoded = read_image_bgr(img_path.read_bytes())
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (24, 32, 3)


def test_read_image_bgr_rejects_garbage():
    with pytest.raises(ValueError, match="Could not decode image data"):
        read_image_bgr(b"not-an-image")


def test_read_image_bgr_collapses_channels():
    """Pseudo-color input (B=255, R=G=0) must decode to equal channels:
    single-band IR training domain, matching the reference script."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR

    ok, buf = cv2.imencode(".png", img)
    assert ok
    decoded = read_image_bgr(buf.tobytes())

    assert decoded.shape == (16, 16, 3)
    b, g, r = decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2]
    assert np.array_equal(b, g)
    assert np.array_equal(g, r)


# ---------------------------------------------------------------------------
# 8. Metrics computation and aggregation
# ---------------------------------------------------------------------------


def test_compute_and_aggregate_metrics():
    d1 = Detection(cls_id=0, cls_name="ship", conf=0.9, xyxy=(10.0, 10.0, 50.0, 40.0))
    d2 = Detection(cls_id=0, cls_name="ship", conf=0.5, xyxy=(20.0, 20.0, 60.0, 60.0))

    m1 = compute_image_metrics("a.png", [d1, d2], 12.5)
    assert m1["filename"] == "a.png"
    assert m1["latency_ms"] == 12.5
    assert m1["num_detections"] == 2
    assert m1["per_class"] == {"ship": 2}
    assert m1["conf_mean"] == pytest.approx(0.7)
    assert m1["conf_max"] == pytest.approx(0.9)
    assert m1["conf_min"] == pytest.approx(0.5)

    m2 = compute_image_metrics("b.png", [], 7.5)
    assert m2["num_detections"] == 0
    assert m2["per_class"] == {}
    assert m2["conf_mean"] == 0.0
    assert m2["conf_max"] == 0.0
    assert m2["conf_min"] == 0.0

    agg = aggregate_metrics([m1, m2])
    assert agg["total_images"] == 2
    assert agg["images_with_detections"] == 1
    assert agg["total_detections"] == 2
    assert agg["class_distribution"] == {"ship": 2}
    assert agg["total_latency_ms"] == pytest.approx(20.0)
    assert agg["mean_latency_ms"] == pytest.approx(10.0)

    empty = aggregate_metrics([])
    assert empty["total_images"] == 0
    assert empty["total_detections"] == 0
    assert empty["total_latency_ms"] == 0.0
    assert empty["mean_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# 9. preprocess_for keeps 3-channel uint8 layout for every pipeline category
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prep", ["dual", "wavelet", "rpca", "tophat", "butterworth"])
def test_preprocess_for_returns_3ch(prep):
    rng = np.random.default_rng(42)
    bgr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    loaded = LoadedModel(model=None, prep=prep, device="cpu")
    out = preprocess_for(loaded, bgr)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_preprocess_for_raw_passthrough():
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    loaded = LoadedModel(model=None, prep="raw", device="cpu")
    assert preprocess_for(loaded, bgr) is bgr
