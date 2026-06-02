"""
Batch inference on the customn dataset using the Dual-stream + SpatialFrequencyFusion model.

Applies awb_dual_fusion preprocessing (spatial tophat+CLAHE + frequency Butterworth BPF+CLAHE,
stacked as [orig, freq, spatial] → yields [spatial, freq, orig] after BGR→RGB) and runs
YOLO prediction using the fusion model weights.

Usage:
    uv run python scripts/inference_customn_fusion.py --weights runs/detect/ship-detection/nslsr-lwir-yolo11m-awb_dual_fusion-fusion/weights/best.pt
    uv run python scripts/inference_customn_fusion.py --weights <path-to-weights> --conf 0.25
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is in sys.path for preprocessing_module import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship"]

# Expected customn location: ship/datasets/customn/
CUSTOMN_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "customn"


def collect_bmp_files(data_dir: Path, max_images: int | None = None):
    """Recursively collect all .BMP files (case-insensitive), skipping 'predict' subdir.

    Args:
        data_dir: Root directory to scan.
        max_images: If set, sample evenly across subdirectories (up to this total).
    """
    all_bmps = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() == ".bmp" and p.is_file()
    )
    # Filter out files under 'predict/' subdirectory (previous inference results)
    filtered = [p for p in all_bmps if "predict" not in p.parts]
    skipped = len(all_bmps) - len(filtered)
    if skipped:
        logger.info("  Skipped %d files under 'predict/' subdirectory", skipped)

    if max_images is not None and len(filtered) > max_images:
        # Group by parent subdirectory for even sampling
        from collections import defaultdict
        groups = defaultdict(list)
        for p in filtered:
            groups[p.parent].append(p)
        sampled = []
        per_group = max(1, max_images // len(groups))
        for parent in sorted(groups):
            files = groups[parent]
            if len(files) <= per_group:
                sampled.extend(files)
            else:
                step = len(files) / per_group
                sampled.extend(files[::max(1, int(step))][:per_group])
        filtered = sorted(sampled)[:max_images]
        logger.info("  Sampled %d files (max_images=%d, %d subdirs)",
                     len(filtered), max_images, len(groups))

    return filtered


def draw_detections(image: np.ndarray, detections: list):
    """Draw bounding boxes with labels and confidence scores on the image.

    Args:
        image: BGR numpy array (OpenCV format) — modified in-place.
        detections: List of (class_id, confidence, x1, y1, x2, y2) tuples.
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
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
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


def build_fusion_input(
    image_bgr: np.ndarray,
    method: str = "awb_dual_fusion",
) -> np.ndarray:
    """Build the 3-channel fusion input with dual-stream preprocessing.

    Pipeline:
        AWB → Dual-stream: spatial (tophat+CLAHE) AND frequency (method-specific)
        → stacked as [orig, freq, spatial] BGR (so after BGR→RGB it becomes
        [spatial, freq, orig] — which is what SpatialFrequencyFusion expects.

    Args:
        image_bgr: 3-channel BGR image (H, W, 3) uint8.
        method: Preprocessing method — ``"awb_dual_fusion"`` (Butterworth BPF,
                default) or ``"awb_wavelet_dual_fusion"`` (wavelet DWT).

    Returns:
        3-channel BGR fused image (H, W, 3).
    """
    from preprocessing_module import PreprocessingPipelineDual, SeaClutterMethod

    if method == "awb_wavelet_dual_fusion":
        pipeline = PreprocessingPipelineDual(freq_method=SeaClutterMethod.WAVELET)
    else:
        pipeline = PreprocessingPipelineDual()  # default: Butterworth
    fused = pipeline(image_bgr)  # (H, W, 3) BGR layout: [orig, freq, spatial]
    return fused


def run_fusion_inference(
    model_path: str,
    image_paths: list,
    conf: float = 0.25,
    imgsz: int = 640,
    method: str = "awb_dual_fusion",
):
    """Run fusion model inference on all images.

    Each image is preprocessed with the specified dual-stream method before feeding to YOLO.
    Detections are drawn on the original image (not the preprocessed one).

    Args:
        model_path: Path to the trained fusion model weights (best.pt).
        image_paths: List of paths to input BMP images.
        conf: Confidence threshold.
        imgsz: Image size for YOLO inference.

    Returns:
        List of (img_path, detections_list) tuples.
    """
    from ultralytics import YOLO

    logger.info("Loading fusion model from %s ...", model_path)
    model = YOLO(model_path)

    all_detections = []
    save_interval = max(100, len(image_paths) // 10)  # save ~10 checkpoints
    checkpoint_dir = Path("runs") / "inference_fusion" / "checkpoint"

    for idx, img_path in enumerate(image_paths):
        try:
            # Read original grayscale image
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                logger.warning("  Skipping %s — failed to load", img_path)
                all_detections.append((img_path, []))
                continue

            # Convert to 3-channel BGR (PreprocessingPipelineDual expects BGR)
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            # Apply dual-stream preprocessing
            fused = build_fusion_input(img_bgr, method=method)

            # Run YOLO prediction on the preprocessed image
            results = model(fused, imgsz=imgsz, conf=conf, verbose=False)

            # Parse detections
            img_detections = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    img_detections.append((cls_id, conf_val, x1, y1, x2, y2))

            # Draw detections on the ORIGINAL image (converted to BGR for colored boxes)
            img_bgr_orig = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            draw_detections(img_bgr_orig, img_detections)
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
                relative = img_path.relative_to(CUSTOMN_DIR)
            except ValueError:
                relative = img_path.name
            rel_str = str(relative).replace("\\", "/")

            # Only keep images that have detections
            if img_detections:
                images_with_detections += 1

                # Save annotated image with bounding boxes, preserving subfolder structure
                out_img = output_dir / relative
                out_img.parent.mkdir(parents=True, exist_ok=True)

                img_bgr_orig = cv2.imread(str(img_path))
                if img_bgr_orig is not None:
                    draw_detections(img_bgr_orig, img_detections)
                    cv2.imwrite(str(out_img), img_bgr_orig)
            for cls_id, conf, x1, y1, x2, y2 in img_detections:
                total_detections += 1
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                writer.writerow([
                    f"customn/{rel_str}",
                    cls_id,
                    cls_name,
                    f"{conf:.2f}",
                    int(x1), int(y1), int(x2), int(y2),
                ])

    logger.info("")
    logger.info("=" * 50)
    logger.info("Fusion Inference Summary")
    logger.info("=" * 50)
    logger.info("  Total images scanned:   %d", total_images)
    logger.info("  Images with detections: %d", images_with_detections)
    logger.info("  Total detections:       %d", total_detections)
    logger.info("  Annotated images:       %s", output_dir)
    logger.info("  Detection log:          %s", detections_csv)
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Fusion model inference on customn dataset (Dual-stream + SpatialFrequencyFusion)",
    )
    parser.add_argument(
        "--weights", "-w", type=str, required=True,
        help="Path to trained fusion model weights (required)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for YOLO inference (default: 640)",
    )
    parser.add_argument(
        "--method", type=str, default="awb_dual_fusion",
        choices=["awb_dual_fusion", "awb_wavelet_dual_fusion"],
        help="Dual-stream preprocessing method (default: awb_dual_fusion)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit total images (evenly sampled across subdirs, default: all)",
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

    # Collect all BMP files
    logger.info("Scanning %s for .BMP files ...", CUSTOMN_DIR)
    image_paths = collect_bmp_files(CUSTOMN_DIR, max_images=args.max_images)
    if not image_paths:
        logger.error("No .BMP files found in %s", CUSTOMN_DIR)
        sys.exit(1)
    logger.info("Found %d .BMP files", len(image_paths))

    # Create output directory — name indicates fusion algorithm
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / "inference_fusion" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    logger.info("Running fusion inference with method=%s, confidence=%.2f, imgsz=%d ...",
                args.method, args.conf, args.imgsz)
    all_detections = run_fusion_inference(
        str(weights_path), image_paths,
        conf=args.conf, imgsz=args.imgsz,
        method=args.method,
    )

    # Save results
    save_results(all_detections, output_dir)

    logger.info("Done. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
