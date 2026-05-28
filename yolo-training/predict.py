"""
Inference script for ship detection with SAHI tiled inference support.

Supports two modes:
  1. Standard YOLO inference (--mode standard)
  2. SAHI tiled inference (--mode sahi) — recommended for small objects

Usage:
    # Standard inference
    uv run python predict.py --source image.jpg --weights runs/detect/train/weights/best.pt

    # SAHI tiled inference (better for small ships)
    uv run python predict.py --mode sahi --source image.jpg --weights runs/detect/train/weights/best.pt

    # Batch inference on directory
    uv run python predict.py --source ./test_images/ --mode sahi
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SAHI class names (matching configs/data.yaml)
CLASS_NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
    "Independence", "Jiangkai II", "Oliver Hazard Perry",
    "Sejong Daewang", "Zumwalt",
]

CLASS_NAMES_NSLSR = ["ship"]

# Dataset → class names mapping
CLASS_NAMES_MAP = {
    "ship": CLASS_NAMES,
    "nslsr-swir": CLASS_NAMES_NSLSR,
    "nslsr-lwir": CLASS_NAMES_NSLSR,
}


def standard_inference(
    model_path: str, source: str,
    conf: float = 0.25, imgsz: int = 1280,
    class_names: list | None = None,
):
    """Run standard YOLO inference."""
    from ultralytics import YOLO

    if class_names is None:
        class_names = CLASS_NAMES

    logger.info("Loading model from %s ...", model_path)
    model = YOLO(model_path)

    logger.info("Running inference on %s ...", source)
    results = model(source, imgsz=imgsz, conf=conf, save=True)

    for i, r in enumerate(results):
        if r.boxes is not None and len(r.boxes) > 0:
            logger.info("Result %d: %d detections", i, len(r.boxes))
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                logger.info(
                    "  %s: conf=%.3f, xyxy=%s",
                    class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                    conf_val,
                    box.xyxy[0].tolist(),
                )
        else:
            logger.info("Result %d: no detections", i)

    logger.info("Results saved in runs/detect/predict/")


def sahi_inference(
    model_path: str,
    source: str,
    conf: float = 0.3,
    slice_size: int = 512,
    overlap: float = 0.2,
    class_names: list | None = None,
):
    """Run SAHI tiled inference for improved small object detection."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.cv import visualize_object_predictions
    except ImportError:
        logger.error("SAHI not installed. Run: uv add sahi")
        sys.exit(1)

    source_path = Path(source)
    if source_path.is_dir():
        image_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.JPG"))
    else:
        image_paths = [source_path]

    logger.info("Loading detection model from %s ...", model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",  # fallback to CPU in try/except
    )

    # Fallback to CPU if CUDA not available
    import torch
    if not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU.")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            device="cpu",
        )

    output_dir = Path("runs/predict_sahi")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        logger.info("Processing %s ...", img_path.name)

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

        # Log detections
        object_preds = result.object_prediction_list
        logger.info("  Detected %d objects", len(object_preds))
        for pred in object_preds:
            bbox = pred.bbox
            cls_name = pred.category.name
            score = pred.score.value
            logger.info(
                "  %s: conf=%.3f, bbox=%s",
                cls_name, score,
                [round(v, 1) for v in bbox.to_voc_bbox()],
            )

        # Save visualization
        vis_img = visualize_object_predictions(
            np.array(Image.open(img_path).convert("RGB")),
            object_preds,
        )
        # visualize_object_predictions returns PIL Image or numpy array
        if isinstance(vis_img, Image.Image):
            vis_img = np.array(vis_img)

        out_path = output_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        logger.info("  Saved visualization to %s", out_path)

    logger.info("All SAHI predictions saved in %s", output_dir)


def save_detections_csv(results, output_path, class_names):
    """Save detection results to a CSV file.

    Args:
        results: List of (filename, [(cls_id, conf, x1, y1, x2, y2), ...]) tuples.
        output_path: Path to output CSV file.
        class_names: List of class name strings.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for filename, detections in results:
            for cls_id, conf, x1, y1, x2, y2 in detections:
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                writer.writerow([filename, cls_id, cls_name, f"{conf:.3f}", int(x1), int(y1), int(x2), int(y2)])


def main():
    parser = argparse.ArgumentParser(description="Ship detection inference")
    parser.add_argument(
        "--mode", choices=["standard", "sahi"], default="sahi",
        help="Inference mode: standard YOLO or SAHI tiled (default: sahi)",
    )
    parser.add_argument(
        "--source", "-s", type=str, required=True,
        help="Path to image, directory, or video",
    )
    parser.add_argument(
        "--weights", "-w", type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to model weights (default: runs/detect/train/weights/best.pt)",
    )
    parser.add_argument(
        "--dataset", type=str, default="ship",
        choices=["ship", "nslsr-swir", "nslsr-lwir"],
        help="Dataset type for class name mapping (default: ship)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (standard mode)")
    parser.add_argument(
        "--slice-size", type=int, default=512,
        help="SAHI slice size in pixels (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.2,
        help="SAHI slice overlap ratio (default: 0.2)",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional path to save detection CSV log",
    )
    args = parser.parse_args()

    if not Path(args.weights).exists():
        logger.warning("Weights not found at %s", args.weights)
        logger.warning("Train a model first or specify a different path.")

    class_names = CLASS_NAMES_MAP.get(args.dataset, CLASS_NAMES_MAP["ship"])
    logger.info("Using dataset: %s (%d classes)", args.dataset, len(class_names))

    if args.mode == "standard":
        standard_inference(args.weights, args.source, args.conf, args.imgsz, class_names)
    else:
        sahi_inference(args.weights, args.source, args.conf, args.slice_size, args.overlap, class_names)


if __name__ == "__main__":
    main()
