"""Framework-free detection core for infrared ship detection.

Pure-Python module with no UI-framework coupling. Heavy dependencies
(ultralytics, torch, preprocessing_module) are imported lazily inside the
functions that need them so that ``import detector`` stays light and works
even without model files present.

The ``sys.path`` bootstrap below must run before any heavy import: the
preprocessing pipelines that reproduce each model's training input
distribution live in ``preprocessing_module`` (under ``yolo-training/``).
This mirrors the convention used by
``yolo-training/scripts/inference_customn_fusion.py``.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_YOLO_TRAINING = _REPO_ROOT / "yolo-training"
if str(_YOLO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_YOLO_TRAINING))

RUNS_DIR = _REPO_ROOT / "runs" / "detect" / "ship-detection"
METRICS_JSON = _REPO_ROOT / "yolo-training" / "test_set_results.json"


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelInfo:
    name: str  # run dir name e.g. "T5_yolo11l_fusion"
    path: Path  # .../weights/best.pt
    map50_95: float | None
    map50: float | None
    prep: str  # training input distribution: dual|wavelet|rpca|tophat|butterworth|raw

    @property
    def label(self) -> str:
        if self.map50_95 is None or self.map50 is None:
            return f"{self.name} [{self.prep}]"
        return (
            f"{self.name} [{self.prep}] "
            f"(mAP50-95={self.map50_95:.2f} | mAP50={self.map50:.2f})"
        )


@dataclass
class LoadedModel:
    model: object  # ultralytics YOLO instance
    prep: str  # training input distribution: dual|wavelet|rpca|tophat|butterworth|raw
    device: str


def _classify_prep(data_str: str) -> str:
    """Map a dataset path/name string to its training input distribution.

    Case-insensitive substring rules matching this repo's preprocessed-data
    naming (``_preprocessed_data_awb_<variant>.yaml`` and run-dir names).
    Order matters: ``wavelet_dual_fusion`` and ``rpca_dual_fusion`` both
    contain ``dual_fusion``, so they are tested first.
    """
    s = data_str.lower()
    if "wavelet_dual_fusion" in s:
        return "wavelet"
    if "rpca_dual_fusion" in s:
        return "rpca"
    if "dual_fusion" in s:
        return "dual"
    if "tophat_clahe" in s:
        return "tophat"
    if "butterworth_clahe" in s:
        return "butterworth"
    # Weaker signal, only reachable for name-based fallbacks such as
    # "T4_wavelet_dual": every repo data line carries the full variant
    # phrase above, so this never reclassifies an authoritative data line.
    if "wavelet" in s:
        return "wavelet"
    return "raw"


def _prep_from_args_yaml(weights_path: Path) -> str | None:
    """Classify the ``data:`` line of ``args.yaml`` sitting next to the weights.

    The authoritative input spec for a run is the ``data:`` entry of the
    ``args.yaml`` Ultralytics writes beside ``best.pt``. Returns ``None``
    when that file is missing or unparsable so callers can fall back to
    name-based classification.
    """
    import yaml

    args_path = weights_path.parent.parent / "args.yaml"
    try:
        cfg = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(cfg, dict):
        data_str = cfg.get("data")
        if isinstance(data_str, str) and data_str:
            return _classify_prep(data_str)
    return None


def discover_models(
    runs_dir: Path = RUNS_DIR, metrics_json: Path = METRICS_JSON
) -> list[ModelInfo]:
    """Find every ``<run>/weights/best.pt`` under *runs_dir* and join test-set
    metrics by run-dir name.

    Each run's training input distribution (``prep``) is classified from the
    ``data:`` line of its sibling ``args.yaml``, falling back to keyword rules
    on the run-dir name when that file is missing or unparsable.

    Sorted by mAP50-95 descending with unknown-metric runs last (stable), so
    index 0 is the default best model.
    """
    if not runs_dir.is_dir():
        return []

    metrics: dict = {}
    if metrics_json.is_file():
        try:
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}

    models: list[ModelInfo] = []
    for best in sorted(runs_dir.glob("*/weights/best.pt")):
        run_name = best.parent.parent.name
        entry = metrics.get(run_name) or {}
        m = entry.get("metrics") or {}
        prep = _prep_from_args_yaml(best)
        if prep is None:
            prep = _classify_prep(run_name)
        models.append(
            ModelInfo(
                name=run_name,
                path=best,
                map50_95=m.get("mAP50-95"),
                map50=m.get("mAP50"),
                prep=prep,
            )
        )

    # None LAST (stable): has-metric (False) sorts before None (True);
    # within known metrics, descending via negation.
    models.sort(key=lambda mi: (mi.map50_95 is None, -(mi.map50_95 or 0.0)))
    return models


def resolve_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gpu_display_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def load_model(path: str | Path) -> LoadedModel:
    """Load a YOLO checkpoint and resolve its training input distribution."""
    from ultralytics import YOLO

    model_path = Path(path)
    model = YOLO(str(model_path))

    # Authoritative spec: the data line of args.yaml next to the weights.
    # Fallback when absent/unparsable: keyword rules on the checkpoint path.
    prep = _prep_from_args_yaml(model_path)
    if prep is None:
        prep = _classify_prep(str(model_path))

    return LoadedModel(model=model, prep=prep, device=resolve_device())


def read_image_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes to a 3-channel BGR ndarray (unicode-path-safe).

    Decoded channels are collapsed through grayscale (BGR→GRAY→BGR): every
    model in this repo is single-band IR trained, and the reference script
    (``inference_customn_fusion.py``) reads ``IMREAD_GRAYSCALE`` then
    ``GRAY2BGR`` before preprocessing. Customn BMPs are dual-band
    pseudo-color with B=0, so keeping raw decoded channels would corrupt the
    AWB gain estimation inside the preprocessing pipelines.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


def preprocess_for(loaded: LoadedModel, bgr: np.ndarray) -> np.ndarray:
    """Route *bgr* through the preprocessing distribution *loaded* was trained on.

    Reproduces each run's training-time input pipeline (per the ``data:``
    entry of its args.yaml):

    - ``dual``        : PreprocessingPipelineDual (AWB → tophat/CLAHE ⊕ AWB → butterworth BPF/CLAHE)
    - ``wavelet``     : dual with a WAVELET frequency stream
    - ``rpca``        : dual with an RPCA frequency stream
    - ``tophat``      : single-stream AWB → top-hat → CLAHE → unsharp
    - ``butterworth`` : single-stream AWB → Butterworth BPF → CLAHE → unsharp
    - ``raw``         : unchanged passthrough

    Input must already be 3-channel BGR (1-channel-replicated is fine);
    grayscale conversion is the caller's responsibility (see read_image_bgr).
    """
    from preprocessing_module import (
        ColorCorrectionMethod,
        EnhancementMethod,
        PreprocessingConfig,
        PreprocessingPipeline,
        PreprocessingPipelineDual,
        SeaClutterMethod,
    )

    if loaded.prep == "dual":
        return PreprocessingPipelineDual()(bgr)
    if loaded.prep == "wavelet":
        return PreprocessingPipelineDual(freq_method=SeaClutterMethod.WAVELET)(bgr)
    if loaded.prep == "rpca":
        return PreprocessingPipelineDual(freq_method=SeaClutterMethod.RPCA)(bgr)

    if loaded.prep == "tophat":
        config = PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.TOPHAT,
            tophat_kernel=31,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        )
        return PreprocessingPipeline(config)(bgr)

    if loaded.prep == "butterworth":
        config = PreprocessingConfig(
            color_correction_method=ColorCorrectionMethod.SEA_WHITE_BALANCE,
            wb_corner_fraction=0.06,
            wb_gray_target=128.0,
            sea_clutter_method=SeaClutterMethod.BUTTERWORTH_BPF,
            butterworth_cutoff_low=8.0,
            butterworth_cutoff_high=80.0,
            enhancement_method=EnhancementMethod.CLAHE,
            clahe_clip_limit=3.0,
            edge_enhance=True,
            unsharp_strength=0.5,
        )
        return PreprocessingPipeline(config)(bgr)

    # "raw" (and any unknown value): feed the decoded image unchanged.
    return bgr


def draw_detections(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw green boxes + filled label backgrounds on a copy of *bgr*.

    Exact port of yolo-training/scripts/inference_customn_fusion.py:76-112.
    """
    canvas = bgr.copy()
    color = (0, 255, 0)  # green
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.cls_name}: {det.conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - label_h - 4, 0)
        cv2.rectangle(
            canvas,
            (x1, label_y),
            (x1 + label_w + 2, label_y + label_h + 4),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 1, label_y + label_h + 2),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )
    return canvas


def predict_image(
    loaded: LoadedModel,
    bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
) -> tuple[list[Detection], np.ndarray, float]:
    """Run inference on one BGR image; returns (detections, annotated, latency_ms).

    The input is first routed through ``preprocess_for`` so it matches the
    distribution the model was trained on. Boxes are predicted on that
    preprocessed image but drawn on the ORIGINAL image.
    """
    t0 = perf_counter()
    fused = preprocess_for(loaded, bgr)
    results = loaded.model(fused, imgsz=imgsz, conf=conf, verbose=False, device=loaded.device)

    detections: list[Detection] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cls_name = loaded.model.names.get(cls_id, f"class_{cls_id}")
            detections.append(
                Detection(cls_id=cls_id, cls_name=cls_name, conf=cf, xyxy=(x1, y1, x2, y2))
            )

    latency_ms = (perf_counter() - t0) * 1000
    return detections, draw_detections(bgr, detections), latency_ms


def compute_image_metrics(
    filename: str, detections: list[Detection], latency_ms: float
) -> dict:
    """Per-image summary stats; confidence stats are 0.0 when no detections."""
    per_class: dict[str, int] = {}
    for det in detections:
        per_class[det.cls_name] = per_class.get(det.cls_name, 0) + 1

    confs = [det.conf for det in detections]
    return {
        "filename": filename,
        "latency_ms": round(latency_ms, 1),
        "num_detections": len(detections),
        "per_class": per_class,
        "conf_mean": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "conf_max": round(max(confs), 3) if confs else 0.0,
        "conf_min": round(min(confs), 3) if confs else 0.0,
    }


def aggregate_metrics(per_image: list[dict]) -> dict:
    """Fold per-image metric dicts into dataset-level totals (empty-safe)."""
    total_images = len(per_image)
    images_with_detections = sum(1 for m in per_image if m.get("num_detections", 0) > 0)
    total_detections = sum(m.get("num_detections", 0) for m in per_image)

    class_distribution: dict[str, int] = {}
    for m in per_image:
        for name, count in (m.get("per_class") or {}).items():
            class_distribution[name] = class_distribution.get(name, 0) + count

    total_latency_ms = sum(float(m.get("latency_ms", 0.0)) for m in per_image)
    mean_latency_ms = total_latency_ms / total_images if total_images else 0.0

    return {
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "class_distribution": class_distribution,
        "total_latency_ms": round(total_latency_ms, 1),
        "mean_latency_ms": round(mean_latency_ms, 1),
    }
