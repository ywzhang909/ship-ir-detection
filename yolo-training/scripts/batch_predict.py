"""
Batch prediction — sample from train/test/val splits of each dataset.

Outputs grayscale images with red bounding boxes and confidence labels.

Usage:
    uv run python scripts/batch_predict.py
    uv run python scripts/batch_predict.py --sample 20
    uv run python scripts/batch_predict.py --mode standard
    uv run python scripts/batch_predict.py --mode sahi --slice-size 640 --overlap 0.2
"""

import argparse
import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

COLOR_RED = (0, 0, 255)

# ── Dataset configurations with explicit train / test / val splits ──
# For NSLSR: use the pre-organised split directories.
# For customn / lab_photos: point both train & test at the same root; the
# script will randomly split.
DATASET_CONFIGS = [
    {
        "name": "nslsr_swir",
        "display_name": "NSLSR-SWIR",
        "splits": {
            "train": "datasets/NSLSR/data/NSLSR_train/SWIR",
            "test":  "datasets/NSLSR/data/NSLSR_test/SWIR",
            "val":   "datasets/NSLSR/data/NSLSR_val/SWIR",
        },
        "model_path": "runs/detect/ship-detection/nslsr-swir-v1/weights/best.pt",
        "class_names": ["ship"],
        "image_glob": "*.jpg",
        "from_same_root": False,          # each split dir is independent
    },
    {
        "name": "nslsr_lwir",
        "display_name": "NSLSR-LWIR",
        "splits": {
            "train": "datasets/NSLSR/data/NSLSR_train/LWIR",
            "test":  "datasets/NSLSR/data/NSLSR_test/LWIR",
            "val":   "datasets/NSLSR/data/NSLSR_val/LWIR",
        },
        "model_path": "runs/detect/ship-detection/nslsr-swir-v1/weights/best.pt",
        "class_names": ["ship"],
        "image_glob": "*.jpg",
        "from_same_root": False,
    },
    {
        "name": "customn",
        "display_name": "Customn (Real IR)",
        "splits": {
            "train": "datasets/customn",
            "test":  "datasets/customn",
        },
        "model_path": "runs/detect/ship-detection/baseline-yolo11m/weights/best.pt",
        "class_names": [
            "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
            "Independence", "Jiangkai II", "Oliver Hazard Perry",
            "Sejong Daewang", "Zumwalt",
        ],
        "image_glob": "*.bmp",
        "from_same_root": True,           # same dir → random split
        "random_test_ratio": 0.2,
    },
    {
        "name": "lab_photos",
        "display_name": "实验室照片 (Lab Photos)",
        "splits": {
            "train": "datasets/实验室照片",
            "test":  "datasets/实验室照片",
        },
        "model_path": "runs/detect/ship-detection/baseline-yolo11m/weights/best.pt",
        "class_names": [
            "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
            "Independence", "Jiangkai II", "Oliver Hazard Perry",
            "Sejong Daewang", "Zumwalt",
        ],
        "image_glob": "*.bmp",
        "from_same_root": True,
        "random_test_ratio": 0.2,
    },
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ship/


# ═══════════════════════════════════════════════════════════════════
#  Image collection & sampling
# ═══════════════════════════════════════════════════════════════════

def collect_all_images(root_dir: Path, glob_pattern: str) -> list[Path]:
    """Return every image file under *root_dir* matching *glob_pattern*."""
    if not root_dir.exists():
        return []
    ext = Path(f"x{glob_pattern}").suffix.lower()
    return sorted(p for p in root_dir.rglob("*")
                  if p.suffix.lower() == ext and p.is_file())


def sample_split_images(
    config: dict,
    split_name: str,
    sample_size: int,
    seed: int = 42,
) -> list[Path]:
    """Return a random sample of image paths for a given train/test/val split.

    For datasets where ``from_same_root`` is ``True`` (customn, lab_photos),
    the train and test splits share the same directory, so we first collect
    *all* images and then split them deterministically (80 / 20 by default).
    For pre-split datasets (NSLSR) each split directory is used independently.
    """
    split_dir = PROJECT_ROOT / config["splits"][split_name]
    if not split_dir.exists():
        logger.warning("  Split dir not found: %s — skipping", split_dir)
        return []

    if config.get("from_same_root"):
        # ── Shared-root dataset: deterministic random split ──
        all_images = collect_all_images(split_dir, config["image_glob"])
        if not all_images:
            return []
        rng = random.Random(seed)
        rng.shuffle(all_images)

        ratio = config.get("random_test_ratio", 0.2)
        n_test = max(1, int(len(all_images) * ratio))
        if split_name == "train":
            selected = all_images[n_test:]
        elif split_name == "test":
            selected = all_images[:n_test]
        else:  # val — not applicable, same as test
            selected = all_images[:n_test]

    else:
        # ── Pre-split dataset ──
        selected = collect_all_images(split_dir, config["image_glob"])

    # Random sample
    if len(selected) <= sample_size:
        return selected
    rng = random.Random(seed)
    return rng.sample(selected, sample_size)


# ═══════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════

def standard_inference(
    model_path: str,
    image_paths: list[Path],
    class_names: list[str],
    conf: float = 0.25,
    imgsz: int = 640,
) -> list:
    """Run standard YOLO inference."""
    from ultralytics import YOLO

    logger.info("  Loading model from %s ...", model_path)
    model = YOLO(model_path)

    all_detections = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            all_detections.append((img_path, []))
            continue

        results = model(img_path, imgsz=imgsz, conf=conf, verbose=False)
        img_detections = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                img_detections.append((cls_id, conf_val, x1, y1, x2, y2))
        all_detections.append((img_path, img_detections))
        if img_detections:
            logger.info("    %s — %d detections", img_path.name, len(img_detections))
    return all_detections


def sahi_inference(
    model_path: str,
    image_paths: list[Path],
    class_names: list[str],
    conf: float = 0.25,
    slice_size: int = 640,
    overlap: float = 0.2,
) -> list:
    """Run SAHI tiled inference for better small object detection."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("SAHI not installed. Run: uv add sahi")
        sys.exit(1)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("  Loading model from %s (device=%s) ...", model_path, device)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    all_detections = []
    for img_path in image_paths:
        try:
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
            )
            img_detections = []
            for pred in result.object_prediction_list:
                bbox = pred.bbox
                cls_id = int(pred.category.id)
                conf_val = float(pred.score.value)
                x1, y1, x2, y2 = bbox.to_voc_bbox()
                img_detections.append((cls_id, conf_val, x1, y1, x2, y2))
            all_detections.append((img_path, img_detections))
            if img_detections:
                logger.info("    %s — %d detections", img_path.name, len(img_detections))
        except Exception as exc:
            logger.warning("    Error processing %s: %s", img_path.name, exc)
            all_detections.append((img_path, []))
    return all_detections


# ═══════════════════════════════════════════════════════════════════
#  Drawing
# ═══════════════════════════════════════════════════════════════════

def draw_detections_grayscale(
    image_bgr: np.ndarray,
    detections: list,
    class_names: list[str],
) -> np.ndarray:
    """Convert to grayscale, then draw red bounding boxes + confidence labels."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 2
    font_thickness = 1

    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis, (x1_i, y1_i), (x2_i, y2_i), COLOR_RED, thickness)

        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{cls_name}: {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1_i - 8, th + 4)
        cv2.rectangle(
            vis,
            (x1_i, label_y - th - 4),
            (x1_i + tw + 4, label_y + 2),
            COLOR_RED,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            vis, label, (x1_i + 2, label_y - 1),
            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
        )
    return vis


# ═══════════════════════════════════════════════════════════════════
#  Save results
# ═══════════════════════════════════════════════════════════════════

def save_split_results(
    all_detections: list,
    output_dir: Path,
    class_names: list[str],
    split_name: str,
):
    """Save grayscale annotated images and CSV detection log for one split."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    detections_csv = split_dir / "detections.csv"

    total_detections = 0
    images_with_detections = 0
    total_images = len(all_detections)

    with open(detections_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "class_id", "class_name",
                          "confidence", "x1", "y1", "x2", "y2"])

        for img_path, img_detections in all_detections:
            out_img_path = split_dir / img_path.name
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                vis = draw_detections_grayscale(img_bgr, img_detections, class_names)
                cv2.imwrite(str(out_img_path), vis)

            if img_detections:
                images_with_detections += 1
            for cls_id, conf, x1, y1, x2, y2 in img_detections:
                total_detections += 1
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                writer.writerow([
                    img_path.name,
                    split_name,
                    cls_id,
                    cls_name,
                    f"{conf:.2f}",
                    int(x1), int(y1), int(x2), int(y2),
                ])

    logger.info("  [%s] %d images, %d detections → %s",
                split_name, total_images, total_detections, split_dir)
    return {
        "split": split_name,
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
    }


# ═══════════════════════════════════════════════════════════════════
#  Process one dataset (all splits)
# ═══════════════════════════════════════════════════════════════════

def process_dataset(
    config: dict,
    inference_fn: Callable,
    inference_kwargs: dict,
    sample_size: int,
) -> dict:
    """Sample images from each split, run inference, save results."""
    dataset_name = config["display_name"]
    model_path = PROJECT_ROOT / config["model_path"]
    class_names = config["class_names"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("  Model:   %s", model_path)
    logger.info("  Classes: %d — %s", len(class_names), class_names)
    logger.info("  Sample:  %d images per split", sample_size)
    logger.info("=" * 60)

    if not model_path.exists():
        logger.warning("  Model not found at %s — skipping", model_path)
        return {"name": dataset_name, "status": "skipped", "reason": "model not found"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = PROJECT_ROOT / "runs" / "batch_predict" / f"{config['name']}_{timestamp}"

    split_summaries = []
    # Process available splits (val is optional)
    available_splits = [s for s in ("train", "test", "val") if s in config.get("splits", {})]
    for split_name in available_splits:
        image_paths = sample_split_images(config, split_name, sample_size)
        if not image_paths:
            logger.info("  [%s] no images — skipped", split_name)
            continue

        logger.info("  [%s] %d images sampled", split_name, len(image_paths))

        all_detections = inference_fn(
            str(model_path),
            image_paths,
            class_names,
            **inference_kwargs,
        )

        summary = save_split_results(
            all_detections, output_root, class_names, split_name,
        )
        split_summaries.append(summary)

    total_images = sum(s["total_images"] for s in split_summaries)
    total_dets = sum(s["total_detections"] for s in split_summaries)
    return {
        "name": dataset_name,
        "status": "completed",
        "total_images": total_images,
        "total_detections": total_dets,
        "output_dir": str(output_root),
        "splits": split_summaries,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sample predictions from train/test splits of multiple datasets")
    parser.add_argument("--mode", choices=["standard", "sahi"], default="sahi",
                        help="Inference mode (default: sahi)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for standard inference")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="SAHI tile overlap ratio")
    parser.add_argument("--sample", type=int, default=20,
                        help="Images sampled per split (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=[cfg["name"] for cfg in DATASET_CONFIGS],
                        default=[cfg["name"] for cfg in DATASET_CONFIGS],
                        help="Datasets to process (default: all)")
    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH PREDICTION — SAMPLED")
    logger.info("  Mode:       %s", args.mode)
    logger.info("  Confidence: %.2f", args.conf)
    logger.info("  Sample:     %d / split", args.sample)
    logger.info("  Datasets:   %s", args.datasets)
    logger.info("=" * 60)

    configs = [cfg for cfg in DATASET_CONFIGS if cfg["name"] in args.datasets]

    if args.mode == "sahi":
        inference_fn = sahi_inference
        inference_kwargs = {
            "conf": args.conf,
            "slice_size": args.slice_size,
            "overlap": args.overlap,
        }
    else:
        inference_fn = standard_inference
        inference_kwargs = {"conf": args.conf, "imgsz": args.imgsz}

    summaries = []
    for config in configs:
        summary = process_dataset(config, inference_fn, inference_kwargs, args.sample)
        summaries.append(summary)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for s in summaries:
        if s["status"] == "completed":
            logger.info("  %-30s %3d images, %4d detections",
                        s["name"], s["total_images"], s["total_detections"])
            for sp in s.get("splits", []):
                logger.info("    └ %-8s %3d images, %4d detections",
                            sp["split"], sp["total_images"], sp["total_detections"])
            logger.info("    Output: %s", s["output_dir"])
        else:
            logger.info("  %-30s SKIPPED — %s", s["name"], s.get("reason", ""))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
