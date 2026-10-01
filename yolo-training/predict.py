"""
Enhanced inference script for IR ship detection.

Supports:
  1. Standard YOLO inference (--mode standard)
  2. SAHI tiled inference (--mode sahi) — recommended for small ships
  3. Integrated preprocessing (--preprocess) — same pipeline as training
  4. SAHI postprocess metrics: IOS (Intersection over Smaller) for better small object NMS
  5. Optional CSV logging with IOS scores

Usage:
    # SAHI inference with preprocessing
    uv run python predict.py --mode sahi --source image.jpg --preprocess --preprocess-method tophat_clahe

    # Standard inference
    uv run python predict.py --mode standard --source ./test_images/
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# SAHI class names (matching configs/data.yaml)
CLASS_NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
    "Independence", "Jiangkai II", "Oliver Hazard Perry",
    "Sejong Daewang", "Zumwalt",
]

CLASS_NAMES_NSLSR = ["ship"]

CLASS_NAMES_MAP = {
    "ship": CLASS_NAMES,
    "nslsr-swir": CLASS_NAMES_NSLSR,
    "nslsr-lwir": CLASS_NAMES_NSLSR,
}


def maybe_preprocess(image: np.ndarray, args):
    """Apply preprocessing pipeline if requested.

    Returns (preprocessed_image, pipeline_description_or_None).
    """
    if not args.preprocess or args.preprocess_method == "none":
        return image, None

    from preprocessing_module import get_preprocessing_pipeline

    pipeline, desc, _ = get_preprocessing_pipeline(args.preprocess_method)
    logger.info("Applying preprocessing: %s", desc)
    return pipeline(image), desc


def standard_inference(
    model_path: str,
    source: str,
    conf: float = 0.25,
    imgsz: int = 1280,
    class_names: list | None = None,
    preprocess_args=None,
):
    """Run standard YOLO inference with optional preprocessing."""
    from ultralytics import YOLO

    logger.info("Loading model from %s ...", model_path)
    model = YOLO(model_path)
    # 统一输出 "ship"：训练数据 class 0 的类名本身是标注错误（标成了 "Ada"，实际为舰船）。
    # 因此无论模型内置 9 类名（baseline/starnet）还是单类名（T5 等），一律显示 "ship"。
    class_names = ["ship"]

    source_path = Path(source)
    image_paths = []
    if source_path.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.bmp", "*.tif"):
            image_paths.extend(source_path.glob(ext))
    else:
        image_paths = [source_path]

    output_dir = Path("runs/detect/predict")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []

    for img_path in image_paths:
        logger.info("Processing %s ...", img_path.name)

        # Optional preprocessing
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("  Failed to read: %s", img_path)
            continue

        if preprocess_args and getattr(preprocess_args, "preprocess", False):
            img_bgr, _ = maybe_preprocess(img_bgr, preprocess_args)
            # Save preprocessed input
            pre_out = output_dir / f"pre_{img_path.name}"
            cv2.imwrite(str(pre_out), img_bgr)

        results = model(img_bgr, imgsz=imgsz, conf=conf, save=False)

        file_detections = []
        for i, r in enumerate(results):
            if r.boxes is not None and len(r.boxes) > 0:
                logger.info("  %d detections", len(r.boxes))
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    logger.info("  %s: conf=%.3f, xyxy=[%.0f,%.0f,%.0f,%.0f]",
                                cls_name, conf_val, x1, y1, x2, y2)
                    file_detections.append((cls_id, conf_val, x1, y1, x2, y2))
            else:
                logger.info("  no detections")

        # Draw detections on a copy of the original image for visualization
        img_vis = cv2.imread(str(img_path))
        for cls_id, conf_val, x1, y1, x2, y2 in file_detections:
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"c{cls_id}"
            color = (0, 255, 0) if conf_val > 0.5 else (0, 255, 255)
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} {conf_val:.0%}"
            # Draw filled label background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(int(y1) - 5, th + 2)
            cv2.rectangle(img_vis, (int(x1), label_y - th - 2),
                          (int(x1) + tw + 4, label_y + 2), color, -1)
            cv2.putText(img_vis, label, (int(x1) + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out_path = output_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img_vis)
        logger.info("  Saved visualization to %s", out_path)

        all_detections.append((img_path.name, file_detections))

    logger.info("Results saved in %s", output_dir)
    return all_detections


def sahi_inference(
    model_path: str,
    source: str,
    conf: float = 0.3,
    slice_size: int = 512,
    overlap: float = 0.2,
    class_names: list | None = None,
    preprocess_args=None,
    postprocess_metric: str = "IOS",
    postprocess_threshold: float = 0.5,
):
    """Run SAHI tiled inference for improved small object detection.

    Uses IOS (Intersection over Smaller) for NMS postprocessing — this is
    critical for small ships because standard IoU gives low overlaps between
    adjacent tiles, causing duplicate detections to survive NMS.

    Args:
        postprocess_metric: NMS metric ('IOS' recommended for small objects).
        postprocess_threshold: NMS threshold (0.5 default, lower = less
            aggressive merging, higher = more merging).
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.cv import visualize_object_predictions
    except ImportError:
        logger.error("SAHI not installed. Run: uv add sahi")
        sys.exit(1)

    if class_names is None:
        class_names = CLASS_NAMES

    source_path = Path(source)
    if source_path.is_dir():
        image_paths = sorted(
            list(source_path.glob("*.jpg")) +
            list(source_path.glob("*.png")) +
            list(source_path.glob("*.JPG")) +
            list(source_path.glob("*.bmp"))
        )
    else:
        image_paths = [source_path]

    # Load detection model
    logger.info("Loading detection model from %s ...", model_path)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    output_dir = Path("runs/predict_sahi")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []
    total_objects = 0

    for img_path in image_paths:
        logger.info("Processing %s ...", img_path.name)

        # Optional preprocessing before SAHI slicing
        if preprocess_args and getattr(preprocess_args, "preprocess", False):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("  Failed to read: %s", img_path)
                continue
            processed_img, desc = maybe_preprocess(img_bgr, preprocess_args)
            # Save to temp file for SAHI
            temp_path = output_dir / f"_pre_{img_path.name}"
            cv2.imwrite(str(temp_path), processed_img)
            source_for_sahi = str(temp_path)
        else:
            source_for_sahi = str(img_path)

        result = get_sliced_prediction(
            source_for_sahi,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_metric=postprocess_metric,
            postprocess_match_threshold=postprocess_threshold,
        )

        # Log detections
        object_preds = result.object_prediction_list
        file_detections = []

        if object_preds:
            logger.info("  Detected %d objects", len(object_preds))
            for pred in object_preds:
                bbox = pred.bbox
                cls_name = pred.category.name
                score = pred.score.value
                cls_id = pred.category.id
                # Get xyxy
                x1, y1, x2, y2 = bbox.to_voc_bbox()
                logger.info("  %s: conf=%.3f, bbox=[%.0f,%.0f,%.0f,%.0f]",
                            cls_name, score, x1, y1, x2, y2)
                file_detections.append((int(cls_id), float(score), x1, y1, x2, y2))
                total_objects += 1
        else:
            logger.info("  no detections")

        all_detections.append((img_path.name, file_detections))

        # Save visualization — draw detections on original (not preprocessed) image
        img_vis = cv2.imread(str(img_path))
        for cls_id, score, x1, y1, x2, y2 in file_detections:
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"c{cls_id}"
            color = (0, 255, 0) if score > 0.5 else (0, 255, 255)
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} {score:.0%}"
            # Draw filled label background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(int(y1) - 5, th + 2)
            cv2.rectangle(img_vis, (int(x1), label_y - th - 2),
                          (int(x1) + tw + 4, label_y + 2), color, -1)
            cv2.putText(img_vis, label, (int(x1) + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out_path = output_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img_vis)
        logger.info("  Saved visualization to %s", out_path)

    logger.info("All SAHI predictions: %d objects across %d images",
                total_objects, len(image_paths))
    logger.info("Results saved in %s", output_dir)

    return all_detections


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

    logger.info("CSV saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Enhanced ship detection inference")

    # Mode
    parser.add_argument("--mode", choices=["standard", "sahi"], default="sahi",
                        help="Inference mode (default: sahi)")
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Path to image or directory")

    # Model
    parser.add_argument("--weights", "-w", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Model weights path")
    parser.add_argument("--dataset", type=str, default="ship",
                        choices=["ship", "nslsr-swir", "nslsr-lwir"],
                        help="Dataset for class name mapping")

    # Inference params
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (standard mode)")

    # SAHI params
    parser.add_argument("--slice-size", type=int, default=512,
                        help="SAHI slice size (default: 512)")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="SAHI slice overlap ratio (default: 0.2)")
    parser.add_argument("--postprocess-metric", type=str, default="IOS",
                        choices=["IOU", "IOS", "IOM"],
                        help="SAHI postprocess match metric (default: IOS for small objects)")
    parser.add_argument("--postprocess-threshold", type=float, default=0.5,
                        help="SAHI NMS threshold (default: 0.5)")

    # Preprocessing
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply preprocessing pipeline")
    parser.add_argument("--preprocess-method", type=str, default="tophat_clahe",
                        choices=["none", "tophat_clahe", "tophat_detail",
                                 "dog_retinex", "retinex_gamma", "median_clahe",
                                 "bilateral_gamma", "ssl_adaptive"],
                        help="Preprocessing preset (default: tophat_clahe)")

    # Output
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Save detection results to CSV")

    args = parser.parse_args()

    # Log setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    if not Path(args.weights).exists():
        logger.warning("Weights not found at %s", args.weights)
        logger.warning("Train a model first or specify a different path.")

    class_names = CLASS_NAMES_MAP.get(args.dataset, CLASS_NAMES_MAP["ship"])
    logger.info("Dataset: %s (%d classes)", args.dataset, len(class_names))
    logger.info("Mode: %s", args.mode)
    logger.info("Preprocessing: %s", args.preprocess_method if args.preprocess else "disabled")
    if args.mode == "sahi":
        logger.info("SAHI: slice=%d, overlap=%.2f, metric=%s, threshold=%.2f",
                    args.slice_size, args.overlap,
                    args.postprocess_metric, args.postprocess_threshold)

    # Run inference
    if args.mode == "standard":
        results = standard_inference(
            args.weights, args.source, args.conf, args.imgsz,
            class_names, preprocess_args=args,
        )
    else:
        results = sahi_inference(
            args.weights, args.source, args.conf,
            args.slice_size, args.overlap,
            class_names, preprocess_args=args,
            postprocess_metric=args.postprocess_metric,
            postprocess_threshold=args.postprocess_threshold,
        )

    # Save CSV
    if args.output_csv:
        save_detections_csv(results, args.output_csv, class_names)


if __name__ == "__main__":
    main()
