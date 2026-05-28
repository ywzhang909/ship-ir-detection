"""
Batch inference on the customn dataset with bounding box visualization.

Scans all .BMP images in `datasets/customn/`, runs YOLO inference
(standard or SAHI tiled), draws bounding boxes, and saves annotated
images plus a CSV detection log.

Usage:
    uv run python scripts/inference_customn.py --weights runs/detect/train/weights/best.pt
    uv run python scripts/inference_customn.py --weights runs/detect/train/weights/best.pt --mode standard
    uv run python scripts/inference_customn.py --weights runs/detect/train/weights/best.pt --dataset nslsr-swir --conf 0.25
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class names per dataset
CLASS_NAMES_MAP = {
    "nslsr-swir": ["ship"],
    "nslsr-lwir": ["ship"],
    "ship": [
        "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
        "Independence", "Jiangkai II", "Oliver Hazard Perry",
        "Sejong Daewang", "Zumwalt",
    ],
}

# Expected customn location: ship/datasets/customn/
CUSTOMN_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "customn"


def collect_bmp_files(data_dir: Path):
    """Recursively collect all .BMP files (case-insensitive)."""
    return sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() == ".bmp" and p.is_file()
    )


def draw_detections(image: np.ndarray, detections: list, class_names: list):
    """Draw bounding boxes with labels and confidence scores on the image.

    Args:
        image: BGR numpy array (OpenCV format).
        detections: List of (class_id, confidence, x1, y1, x2, y2) tuples.
        class_names: List of class name strings.
    """
    color = (0, 255, 0)  # green
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Format label
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{cls_name}: {conf:.2f}"

        # Put label text above the bbox
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1 - label_h - 4, 0)
        cv2.rectangle(
            image,
            (x1, label_y),
            (x1 + label_w + 2, label_y + label_h + 4),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image, label, (x1 + 1, label_y + label_h + 2),
            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA,
        )


def run_standard_inference(
    model_path: str,
    image_paths: list,
    class_names: list,
    conf: float = 0.25,
    imgsz: int = 640,
):
    """Run standard YOLO inference on all images."""
    from ultralytics import YOLO

    logger.info("Loading model from %s ...", model_path)
    model = YOLO(model_path)

    all_detections = []

    for img_path in image_paths:
        try:
            # Load image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("  Skipping %s — failed to load", img_path)
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

            # Draw
            draw_detections(img_bgr, img_detections, class_names)
            all_detections.append((img_path, img_detections))

            if img_detections:
                logger.info("  %s — %d detections", img_path.name, len(img_detections))
            else:
                logger.info("  %s — no detections", img_path.name)

        except Exception as exc:
            logger.warning("  Skipping %s — error: %s", img_path, exc)
            all_detections.append((img_path, []))

    return all_detections


def run_sahi_inference(
    model_path: str,
    image_paths: list,
    class_names: list,
    conf: float = 0.25,
    slice_size: int = 640,
    overlap: float = 0.2,
):
    """Run SAHI tiled inference on all images."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("SAHI is not installed. Run: uv add sahi")
        sys.exit(1)

    logger.info("Loading detection model from %s ...", model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",
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

    all_detections = []

    for img_path in image_paths:
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("  Skipping %s — failed to load", img_path)
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

            # Draw
            draw_detections(img_bgr, img_detections, class_names)
            all_detections.append((img_path, img_detections))

            if img_detections:
                logger.info("  %s — %d detections", img_path.name, len(img_detections))
            else:
                logger.info("  %s — no detections", img_path.name)

        except Exception as exc:
            logger.warning("  Skipping %s — error: %s", img_path, exc)
            all_detections.append((img_path, []))

    return all_detections


def save_results(
    all_detections: list,
    output_dir: Path,
    class_names: list,
    customn_dir: Path,
):
    """Save annotated images and CSV detection log."""
    detections_csv = output_dir / "detections.csv"
    total_detections = 0
    images_with_detections = 0
    total_images = len(all_detections)

    with open(detections_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for img_path, img_detections in all_detections:
            # Compute relative path from customn_dir
            try:
                relative = img_path.relative_to(customn_dir)
            except ValueError:
                relative = img_path.name
            rel_str = str(relative).replace("\\", "/")

            # Save annotated image preserving subfolder structure
            out_img = output_dir / relative
            out_img.parent.mkdir(parents=True, exist_ok=True)

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                # Re-draw if we have detections (we already drew, but need to read back
                # since we drew in memory earlier - but that was in the per-image loop)
                # Actually we already drew on the image during inference, but we saved
                # them in a dict. Let's re-read and re-draw to be safe.
                draw_detections(img_bgr, img_detections, class_names)
                cv2.imwrite(str(out_img), img_bgr)

            # Write CSV rows
            if img_detections:
                images_with_detections += 1
            for cls_id, conf, x1, y1, x2, y2 in img_detections:
                total_detections += 1
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                writer.writerow([
                    f"customn/{rel_str}",
                    cls_id,
                    cls_name,
                    f"{conf:.2f}",
                    int(x1), int(y1), int(x2), int(y2),
                ])

    logger.info("")
    logger.info("=" * 50)
    logger.info("Inference Summary")
    logger.info("=" * 50)
    logger.info("  Total images scanned:   %d", total_images)
    logger.info("  Images with detections: %d", images_with_detections)
    logger.info("  Total detections:       %d", total_detections)
    logger.info("  Annotated images:       %s", output_dir)
    logger.info("  Detection log:          %s", detections_csv)
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference on customn dataset with bbox visualization",
    )
    parser.add_argument(
        "--weights", "-w", type=str, required=True,
        help="Path to trained model weights (required)",
    )
    parser.add_argument(
        "--dataset", type=str, default="nslsr-swir",
        choices=["nslsr-swir", "nslsr-lwir", "ship"],
        help="Dataset type for class name mapping (default: nslsr-swir)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--mode", type=str, default="sahi",
        choices=["standard", "sahi"],
        help="Inference mode: standard YOLO or SAHI tiled (default: sahi)",
    )
    parser.add_argument(
        "--slice-size", type=int, default=640,
        help="SAHI tile size in pixels (default: 640)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.2,
        help="SAHI tile overlap ratio (default: 0.2)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for standard inference (default: 640)",
    )
    args = parser.parse_args()

    # Validate model weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Model weights not found at: %s", weights_path.resolve())
        sys.exit(1)

    # Validate customn dataset directory
    if not CUSTOMN_DIR.exists():
        logger.error("Customn dataset directory not found at: %s", CUSTOMN_DIR)
        sys.exit(1)

    # Get class names for the selected dataset
    class_names = CLASS_NAMES_MAP[args.dataset]

    # Collect all BMP files
    logger.info("Scanning %s for .BMP files ...", CUSTOMN_DIR)
    image_paths = collect_bmp_files(CUSTOMN_DIR)
    if not image_paths:
        logger.error("No .BMP files found in %s", CUSTOMN_DIR)
        sys.exit(1)
    logger.info("Found %d .BMP files", len(image_paths))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / "inference_customn" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    logger.info("Running %s inference with confidence=%.2f ...", args.mode, args.conf)
    if args.mode == "standard":
        all_detections = run_standard_inference(
            str(weights_path), image_paths, class_names,
            conf=args.conf, imgsz=args.imgsz,
        )
    else:
        all_detections = run_sahi_inference(
            str(weights_path), image_paths, class_names,
            conf=args.conf, slice_size=args.slice_size, overlap=args.overlap,
        )

    # Save results
    save_results(all_detections, output_dir, class_names, CUSTOMN_DIR)


if __name__ == "__main__":
    main()
