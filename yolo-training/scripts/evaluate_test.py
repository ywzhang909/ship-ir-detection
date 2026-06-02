"""
Evaluate a trained model on the NSLSR test split and report metrics.

Supports both standard YOLO and fusion models. For fusion models, the
preprocessed data config (preprocessed_{method}.yaml) should be used.

Also supports SAHI tiled inference evaluation for small-object ablation
studies (T20: SAHI + P2).

Usage:
    # Evaluate best T1 fusion model on test set
    uv run python scripts/evaluate_test.py --weights runs/detect/ship-detection/T1_fusion_h32_l3_cbam/weights/best.pt

    # Evaluate with preprocessed data (for fusion models)
    uv run python scripts/evaluate_test.py --weights runs/detect/ship-detection/T1_fusion_h32_l3_cbam/weights/best.pt --data data/_preprocessed_data_awb_dual_fusion.yaml

    # Evaluate all completed experiments
    uv run python scripts/evaluate_test.py --all

    # Evaluate with SAHI tiled inference (T20 ablation)
    uv run python scripts/evaluate_test.py --weights path/to/best.pt --sahi

    # Evaluate ALL experiments with SAHI (T20 full ablation)
    uv run python scripts/evaluate_test.py --all --sahi
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Root directory for experiment results
RunsDir = Path(__file__).resolve().parent.parent.parent / "runs" / "detect" / "ship-detection"

# Map experiment names to their weight paths and data configs
EXPERIMENTS = {
    "Run1_tophat": {
        "weights": RunsDir / "lwir-awb-tophat-clahe-yolo11m" / "weights" / "best.pt",
        "data": None,  # use original data.yaml
        "description": "Run 1: Top-hat + CLAHE (空域杂波抑制)",
    },
    "Run2_butterworth": {
        "weights": RunsDir / "nslsr-lwir-yolo11m-awb_butterworth_clahe" / "weights" / "best.pt",
        "data": None,
        "description": "Run 2: Butterworth BPF + CLAHE (频域杂波抑制)",
    },
    "Run3_fusion_base": {
        "weights": RunsDir / "nslsr-lwir-yolo11m-awb_dual_fusion-fusion" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "Run 3: Fusion baseline (h=16, l=2, SE)",
    },
    "T1_fusion_h32_cbam": {
        "weights": RunsDir / "T1_fusion_h32_l3_cbam" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T1: Fusion h=32, l=3, CBAM",
    },
    "T2_residual07": {
        "weights": RunsDir / "T2_fusion_residual07" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T2: Fusion residual=0.7",
    },
    "T4_wavelet_dual": {
        "weights": RunsDir / "T4_wavelet_dual" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_wavelet_dual_fusion.yaml",
        "description": "T4: Wavelet dual fusion",
    },
    "T8_shape_iou": {
        "weights": RunsDir / "T8_shape_iou" / "weights" / "best.pt",
        "data": None,
        "description": "T8: Shape-IoU loss",
    },
    "T5_yolo11l_fusion": {
        "weights": RunsDir / "T5_yolo11l_fusion" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T5: YOLO11l + Fusion (h=16, l=2, SE)",
    },
    "T7_augment_jitter": {
        "weights": RunsDir / "T7_augment_jitter" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T7: Augment jitter 0.3 + Fusion",
    },
    "T9_wiou_fusion": {
        "weights": RunsDir / "T9_wiou_fusion" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T9: WIoU v3 loss + YOLO11l + Fusion",
    },
    "T25_preprocess_jitter": {
        "weights": RunsDir / "T25_preprocess_jitter" / "weights" / "best.pt",
        "data": RunsDir.parent.parent.parent / "yolo-training" / "data" / "_preprocessed_data_awb_dual_fusion.yaml",
        "description": "T25: Preprocess jitter 0.3 + Fusion",
    },
    # ── SAHI + P2 ablation experiments (T20) ──────────────────────────
    "P2_standard": {
        "weights": RunsDir / "nslsr-lwir-yolo11m_p2" / "weights" / "best.pt",
        "data": None,
        "description": "P2: YOLO11m with P2 head (standard inference)",
    },
    "P2_SAHI": {
        "weights": RunsDir / "nslsr-lwir-yolo11m_p2" / "weights" / "best.pt",
        "data": None,
        "description": "P2: YOLO11m with P2 head (SAHI inference)",
    },
}


def _load_test_data(data_cfg_path: str | Path | None) -> list[tuple[Path, list]]:
    """Load test images and ground-truth labels from a data config.

    Args:
        data_cfg_path: Path to data YAML, or None to auto-detect.

    Returns:
        List of (image_path, [class_id, x1, y1, x2, y2, ...]) per image.
    """
    import yaml

    # Determine data config
    if data_cfg_path is None:
        candidates = [
            Path("configs/data_nslsr_lwir.yaml"),
            Path("configs/data_nslsr_swir.yaml"),
            Path("configs/data.yaml"),
        ]
        for c in candidates:
            if c.exists():
                data_cfg_path = c
                break

    with open(data_cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("path", ""))
    test_cfg = cfg.get("test", "test/images")
    # Resolve test path
    test_dir = data_root / test_cfg if not Path(test_cfg).is_absolute() else Path(test_cfg)
    # Also look for label dir
    label_dir = data_root / str(test_cfg).replace("images", "labels") if not Path(test_cfg).is_absolute() else Path(str(test_cfg).replace("images", "labels"))

    if not test_dir.exists():
        logger.warning("Test directory not found: %s", test_dir)
        return []

    # Collect all images
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"):
        image_paths.extend(sorted(test_dir.glob(ext)))

    # Load labels
    test_data = []
    for img_path in image_paths:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # YOLO format: class cx cy w h (normalized)
                        cx, cy, w, h = map(float, parts[1:5])
                        # Convert to absolute pixel coordinates
                        img_hw = cv2.imread(str(img_path))
                        if img_hw is None:
                            continue
                        H, W = img_hw.shape[:2]
                        # Convert YOLO format to xyxy
                        x1 = (cx - w / 2) * W
                        y1 = (cy - h / 2) * H
                        x2 = (cx + w / 2) * W
                        y2 = (cy + h / 2) * H
                        gt_boxes.append([cls_id, x1, y1, x2, y2])
        test_data.append((img_path, gt_boxes))

    logger.info("Loaded %d test images with labels from %s", len(test_data), test_dir)
    return test_data


def evaluate_with_sahi(
    weights_path: Path,
    conf: float = 0.001,
    slice_size: int = 640,
    overlap: float = 0.2,
    iou_thresh: float = 0.5,
) -> dict:
    """Evaluate a model on the test set using SAHI tiled inference.

    Computes mAP by comparing SAHI detections against ground-truth labels.

    Args:
        weights_path: Path to model weights (.pt).
        conf: Confidence threshold.
        slice_size: SAHI slice size (pixels).
        overlap: SAHI slice overlap ratio.
        iou_thresh: IoU threshold for mAP computation.

    Returns:
        Dict of metrics.
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("SAHI not installed. Run: uv add sahi")
        return {}

    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        return {}

    logger.info("Loading model: %s", weights_path)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=conf,
        device=device,
    )

    # Load test data
    test_data = _load_test_data(None)

    if not test_data:
        return {}

    # Run SAHI inference on each image
    all_pred_boxes = []  # list of (image_idx, cls_id, conf, x1, y1, x2, y2)
    all_gt_boxes = []    # list of (image_idx, cls_id, x1, y1, x2, y2)

    for img_idx, (img_path, gt_boxes) in enumerate(test_data):
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

            for pred in result.object_prediction_list:
                bbox = pred.bbox
                cls_id = int(pred.category.id)
                conf_val = float(pred.score.value)
                x1, y1, x2, y2 = bbox.to_voc_bbox()
                all_pred_boxes.append((img_idx, cls_id, conf_val, x1, y1, x2, y2))
        except Exception as e:
            logger.warning("SAHI failed for %s: %s", img_path.name, e)
            continue

        for gt in gt_boxes:
            cls_id, x1, y1, x2, y2 = gt
            all_gt_boxes.append((img_idx, cls_id, x1, y1, x2, y2))

    # Compute metrics
    if not all_pred_boxes and not all_gt_boxes:
        logger.warning("No predictions and no ground truth found.")
        return {}

    logger.info("SAHI evaluation: %d predictions, %d ground-truth boxes across %d images",
                len(all_pred_boxes), len(all_gt_boxes), len(test_data))

    # Simple mAP computation
    metrics = _compute_map(all_pred_boxes, all_gt_boxes, iou_thresh=iou_thresh)

    return metrics


def _compute_ap_class(
    cls_preds: list,
    cls_gts: list,
    iou_thresh: float,
) -> float:
    """Compute AP for a single class at a given IoU threshold."""
    if not cls_gts or not cls_preds:
        return 0.0

    # Sort predictions by confidence descending
    cls_preds_sorted = sorted(cls_preds, key=lambda x: x[0], reverse=True)

    # Match predictions to ground truth
    gt_matched = set()
    frame_tp = []

    for conf, x1, y1, x2, y2, img_idx in cls_preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gx1, gy1, gx2, gy2, g_img_idx) in enumerate(cls_gts):
            if gt_idx in gt_matched or g_img_idx != img_idx:
                continue
            iou = _compute_iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh:
            gt_matched.add(best_gt_idx)
            frame_tp.append(1)
        else:
            frame_tp.append(0)

    if not frame_tp:
        return 0.0

    # Compute precision-recall curve
    total_gt = len(cls_gts)
    acc_tp = np.cumsum(frame_tp)
    n_pred = np.arange(1, len(frame_tp) + 1)
    prec = acc_tp / n_pred.astype(float)
    rec = acc_tp / total_gt

    # AP = area under precision-recall curve (trapezoidal interpolation)
    ap = 0.0
    for i in range(len(prec)):
        if i == 0:
            ap += prec[i] * rec[i]
        else:
            ap += prec[i] * (rec[i] - rec[i - 1])

    return float(ap)


def _compute_map(
    pred_boxes: list,
    gt_boxes: list,
    iou_thresh: float = 0.5,
) -> dict:
    """Compute mean Average Precision at multiple IoU thresholds.

    Computes AP@50 (IoU=0.5) and AP@50-95 (average over IoU=0.50:0.05:0.95).

    Args:
        pred_boxes: List of (image_idx, cls_id, conf, x1, y1, x2, y2).
        gt_boxes: List of (image_idx, cls_id, x1, y1, x2, y2).
        iou_thresh: Unused (kept for backward compat); computes full AP50-95.

    Returns:
        Dict with mAP50, mAP50-95, Precision, Recall.
    """
    if not gt_boxes:
        return {}

    max_class = max(max(b[1] for b in gt_boxes), max((b[1] for b in pred_boxes), default=0)) + 1

    # Group by class
    class_preds = {}
    class_gts = {}
    for cls_id in range(max_class):
        class_preds[cls_id] = [(p[2], p[3], p[4], p[5], p[6], p[0]) for p in pred_boxes if p[1] == cls_id]
        class_gts[cls_id] = [(g[2], g[3], g[4], g[5], g[0]) for g in gt_boxes if g[1] == cls_id]

    # Per-threshold AP
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    all_aps = []

    for iou_th in iou_thresholds:
        aps_at_th = []
        for cls_id in range(max_class):
            if not class_gts[cls_id]:
                continue
            ap = _compute_ap_class(class_preds[cls_id], class_gts[cls_id], iou_th)
            aps_at_th.append(ap)
        all_aps.append(np.mean(aps_at_th) if aps_at_th else 0.0)

    # Also compute at just 0.5
    aps50 = []
    for cls_id in range(max_class):
        if not class_gts[cls_id]:
            continue
        ap50 = _compute_ap_class(class_preds[cls_id], class_gts[cls_id], 0.50)
        aps50.append(ap50)

    # Compute final precision/recall at IoU=0.5
    final_precisions = []
    final_recalls = []
    for cls_id in range(max_class):
        if not class_gts[cls_id] or not class_preds[cls_id]:
            continue
        # Compute at conf=0.5
        cls_preds_sorted = sorted(class_preds[cls_id], key=lambda x: x[0], reverse=True)
        gt_matched = set()
        total_gt = len(class_gts[cls_id])
        tp = 0
        fp = 0
        for conf, x1, y1, x2, y2, img_idx in cls_preds_sorted:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, (gx1, gy1, gx2, gy2, g_img_idx) in enumerate(class_gts[cls_id]):
                if gt_idx in gt_matched or g_img_idx != img_idx:
                    continue
                iou = _compute_iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= 0.5:
                tp += 1
                gt_matched.add(best_gt_idx)
            else:
                fp += 1
        final_precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        final_recalls.append(tp / total_gt if total_gt > 0 else 0.0)

    result = {
        "mAP50": round(np.mean(aps50) * 100, 2) if aps50 else 0.0,
        "mAP50-95": round(np.mean(all_aps) * 100, 2) if all_aps else 0.0,
        "Precision": round(np.mean(final_precisions) * 100, 2) if final_precisions else 0.0,
        "Recall": round(np.mean(final_recalls) * 100, 2) if final_recalls else 0.0,
        "num_images": 0,
    }

    return result


def _compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def evaluate_model(weights_path: Path, data_cfg: str | None = None, conf: float = 0.001) -> dict:
    """Evaluate a model on the test split using YOLO's built-in val and return metrics.

    Args:
        weights_path: Path to trained model weights (.pt).
        data_cfg: Path to data config YAML (uses model's original data if None).
        conf: Confidence threshold for evaluation (default: 0.001 to capture all).

    Returns:
        Dict with keys: mAP50-95, mAP50, Precision, Recall.
    """
    from ultralytics import YOLO

    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        return {}

    logger.info("Loading model: %s", weights_path)
    model = YOLO(str(weights_path))

    val_kwargs = {
        "split": "test",
        "conf": conf,
        "imgsz": 640,
        "batch": 16,
        "verbose": False,
    }
    if data_cfg and Path(data_cfg).exists():
        val_kwargs["data"] = data_cfg
        logger.info("Using data config: %s", data_cfg)

    logger.info("Running validation on test split ...")
    try:
        metrics = model.val(**val_kwargs)
    except Exception as e:
        logger.error("Validation failed: %s", e)
        # Fallback: try without custom data config
        if "data" in val_kwargs:
            logger.info("Retrying without custom data config ...")
            del val_kwargs["data"]
            try:
                metrics = model.val(**val_kwargs)
            except Exception as e2:
                logger.error("Fallback validation also failed: %s", e2)
                return {}
        else:
            return {}

    if metrics and hasattr(metrics, "box"):
        result = {
            "mAP50-95": round(metrics.box.map * 100, 2),
            "mAP50": round(metrics.box.map50 * 100, 2),
            "Precision": round(metrics.box.mp * 100, 2),
            "Recall": round(metrics.box.mr * 100, 2),
        }
    elif metrics and isinstance(metrics, dict):
        result = {k: round(float(v) * 100, 2) if isinstance(v, float) else v
                  for k, v in metrics.items()}
    else:
        logger.warning("Unexpected metrics format: %s", type(metrics))
        result = {"raw": str(metrics)}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on NSLSR test split"
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (required unless --all)")
    parser.add_argument("--data", type=str, default=None,
                        help="Data config YAML (default: model's original data)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all completed experiments")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold (default: 0.001)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    # SAHI evaluation options
    parser.add_argument("--sahi", action="store_true",
                        help="Use SAHI tiled inference instead of standard YOLO val")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="SAHI slice size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="SAHI slice overlap ratio (default: 0.2)")
    args = parser.parse_args()

    if not args.weights and not args.all:
        logger.error("Either --weights or --all is required")
        sys.exit(1)

    results = {}

    if args.all:
        # Evaluate all experiments
        mode = "SAHI" if args.sahi else "YOLO val"
        logger.info("=" * 60)
        logger.info("Evaluating ALL completed experiments on test set (mode: %s)", mode)
        logger.info("=" * 60)

        for exp_name, exp_info in EXPERIMENTS.items():
            logger.info("-" * 50)
            logger.info("Experiment: %s — %s", exp_name, exp_info["description"])
            logger.info("-" * 50)

            if args.sahi:
                metrics = evaluate_with_sahi(
                    exp_info["weights"],
                    conf=args.conf,
                    slice_size=args.slice_size,
                    overlap=args.overlap,
                )
            else:
                metrics = evaluate_model(
                    exp_info["weights"],
                    data_cfg=exp_info["data"],
                    conf=args.conf,
                )
            results[exp_name] = {
                "description": exp_info["description"],
                "mode": "SAHI" if args.sahi else "standard",
                "metrics": metrics,
            }

            if metrics:
                logger.info("  Test Set Results:")
                for k, v in metrics.items():
                    logger.info("    %s: %.2f" if isinstance(v, float) else "    %s: %s", k, v)
            else:
                logger.warning("  Evaluation failed for %s", exp_name)
    else:
        # Evaluate single model
        if args.sahi:
            metrics = evaluate_with_sahi(
                Path(args.weights),
                conf=args.conf,
                slice_size=args.slice_size,
                overlap=args.overlap,
            )
        else:
            metrics = evaluate_model(
                Path(args.weights),
                data_cfg=args.data,
                conf=args.conf,
            )
        results["single"] = {"weights": str(args.weights), "mode": "SAHI" if args.sahi else "standard", "metrics": metrics}

        if metrics:
            logger.info("Test Set Results:")
            for k, v in metrics.items():
                logger.info("  %s: %.2f" if isinstance(v, float) else "  %s: %s", k, v)
        else:
            logger.warning("Evaluation failed")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", output_path)

    # Print summary table
    if args.all and results:
        logger.info("")
        logger.info("=" * 90)
        logger.info("%-30s %-8s %8s %10s %10s %8s",
                    "Experiment", "Mode", "mAP50", "mAP50-95", "Precision", "Recall")
        logger.info("-" * 90)
        for exp_name, exp_data in results.items():
            m = exp_data.get("metrics", {})
            mode = exp_data.get("mode", "standard")
            logger.info("%-30s %-8s %8.2f %10.2f %10.2f %8.2f",
                        exp_name[:30],
                        mode,
                        m.get("mAP50", 0),
                        m.get("mAP50-95", 0),
                        m.get("Precision", 0),
                        m.get("Recall", 0))
        logger.info("=" * 90)

        # If both standard and SAHI were run, print ablation comparison
        std_results = {k: v for k, v in results.items() if v.get("mode") == "standard"}
        sahi_results = {k: v for k, v in results.items() if v.get("mode") == "SAHI"}
        if std_results and sahi_results:
            logger.info("")
            logger.info("SAHI+P2 ABLATION SUMMARY")
            logger.info("=" * 60)
            for exp_name in set(list(std_results.keys()) + list(sahi_results.keys())):
                std_m = std_results.get(exp_name, {}).get("metrics", {})
                sahi_m = sahi_results.get(exp_name, {}).get("metrics", {})
                if std_m and sahi_m:
                    delta_map50 = sahi_m.get("mAP50", 0) - std_m.get("mAP50", 0)
                    delta_map = sahi_m.get("mAP50-95", 0) - std_m.get("mAP50-95", 0)
                    logger.info("  %s:", exp_name)
                    logger.info("    Standard: mAP50=%.2f, mAP50-95=%.2f",
                                std_m.get("mAP50", 0), std_m.get("mAP50-95", 0))
                    logger.info("    SAHI:     mAP50=%.2f, mAP50-95=%.2f",
                                sahi_m.get("mAP50", 0), sahi_m.get("mAP50-95", 0))
                    logger.info("    Δ (SAHI - Standard): mAP50=%+.2f, mAP50-95=%+.2f",
                                delta_map50, delta_map)


if __name__ == "__main__":
    main()
