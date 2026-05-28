"""
Run inference on YOLO-format test splits and save annotated images to predict/.

Usage:
    uv run python scripts/inference_test_split.py --source data/processed_nslsr/swir/test/images
    uv run python scripts/inference_test_split.py --source data/processed_nslsr/lwir/test/images
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship"]


def run_sahi_inference(model_path, image_paths, conf, slice_size, overlap):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    logger.info("Loading model from %s ...", model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",
    )

    import torch
    if not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU.")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            device="cpu",
        )

    all_detections = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("  Skipping %s — failed to load", img_path.name)
            all_detections.append((img_path, []))
            continue

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

        # Draw detections
        for cls_id, conf, x1, y1, x2, y2 in img_detections:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y1 - lh - 4, 0)
            cv2.rectangle(img_bgr, (x1, ly), (x1 + lw + 2, ly + lh + 4), (0, 255, 0), cv2.FILLED)
            cv2.putText(img_bgr, label, (x1 + 1, ly + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        all_detections.append((img_path, img_detections))
        status = f"{len(img_detections)} detections" if img_detections else "no detections"
        logger.info("  %s — %s", img_path.name, status)

    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Inference on test splits")
    parser.add_argument("--source", required=True, help="Path to test images directory")
    parser.add_argument("--weights", default="runs/detect/ship-detection/nslsr-swir-v1/weights/best.pt",
                        help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI tile size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI tile overlap")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        logger.error("Source not found: %s", src)
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("Weights not found: %s", weights)
        sys.exit(1)

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = sorted([p for ext in exts for p in src.glob(ext)])
    if not image_paths:
        logger.error("No images found in %s", src)
        sys.exit(1)
    logger.info("Found %d images", len(image_paths))

    # Output: parent of images dir → predict/
    predict_dir = src.parent / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", predict_dir)

    # Run inference
    all_detections = run_sahi_inference(
        str(weights), image_paths, args.conf, args.slice_size, args.overlap
    )

    # Save images + CSV
    detections_csv = predict_dir / "detections.csv"
    total_det = 0
    images_with_det = 0

    with open(detections_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for img_path, img_detections in all_detections:
            # Save annotated image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                for cls_id, conf, x1, y1, x2, y2 in img_detections:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ly = max(y1 - lh - 4, 0)
                    cv2.rectangle(img_bgr, (x1, ly), (x1 + lw + 2, ly + lh + 4), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img_bgr, label, (x1 + 1, ly + lh + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(predict_dir / img_path.name), img_bgr)

            if img_detections:
                images_with_det += 1
                total_det += len(img_detections)

            for cls_id, conf, x1, y1, x2, y2 in img_detections:
                writer.writerow([
                    img_path.name, cls_id, CLASS_NAMES[cls_id],
                    f"{conf:.2f}", int(x1), int(y1), int(x2), int(y2),
                ])

    logger.info("=" * 50)
    logger.info("Inference Summary")
    logger.info("=" * 50)
    logger.info("  Total images:          %d", len(image_paths))
    logger.info("  Images with detections: %d", images_with_det)
    logger.info("  Total detections:      %d", total_det)
    logger.info("  Annotated images:      %s", predict_dir)
    logger.info("  Detection log:         %s", detections_csv)

    # Clean up: remove images without detections
    kept = 0
    removed = 0
    for img_path, img_detections in all_detections:
        out_path = predict_dir / img_path.name
        if not img_detections and out_path.exists():
            out_path.unlink()
            removed += 1
        elif img_detections:
            kept += 1

    logger.info("  Kept: %d images with detections", kept)
    logger.info("  Removed: %d images without detections", removed)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
