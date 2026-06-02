"""
TTA (Test-Time Augmentation) evaluation on trained models.

Evaluates with YOLO built-in TTA (multi-scale + horizontal flip)
and compares against standard (non-TTA) evaluation.

Usage:
    uv run python scripts/evaluate_tta.py --weights runs/detect/ship-detection/T16_crrp/weights/best.pt
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Map experiment names to their data configs (matching evaluate_test.py pattern)
RunsDir = Path(__file__).resolve().parent.parent.parent / "runs" / "detect" / "ship-detection"
YoloDir = RunsDir.parent.parent.parent / "yolo-training"

EXPERIMENT_DATA = {
    "T16_crrp": str(YoloDir / "configs" / "data_crrp_augmented.yaml"),
    "T5_yolo11l_fusion": str(YoloDir / "data" / "_preprocessed_data_awb_dual_fusion.yaml"),
}


def run_eval(weights_path, data_cfg, augment=False, conf=0.001):
    """Run validation with optional TTA."""
    from ultralytics import YOLO

    logger.info("Loading model: %s", weights_path)
    model = YOLO(str(weights_path))

    val_kwargs = {
        "split": "test",
        "conf": conf,
        "imgsz": 640,
        "batch": 16,
        "verbose": False,
        "augment": augment,
    }
    if data_cfg and Path(data_cfg).exists():
        val_kwargs["data"] = str(Path(data_cfg).resolve())
        logger.info("Using data config: %s", val_kwargs["data"])

    mode = "TTA (augment=True)" if augment else "Standard"
    logger.info("Running %s evaluation on test split...", mode)
    metrics = model.val(**val_kwargs)

    if metrics and hasattr(metrics, "box"):
        return {
            "mAP50-95": round(metrics.box.map * 100, 2),
            "mAP50": round(metrics.box.map50 * 100, 2),
            "Precision": round(metrics.box.mp * 100, 2),
            "Recall": round(metrics.box.mr * 100, 2),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description="TTA evaluation on test split")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--data", type=str, default=None, nargs="?",
                        help="Data config YAML (auto-detected if omitted)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights not found: %s", weights_path)
        return

    # Auto-detect data config from experiment name in path
    data_cfg = args.data
    if data_cfg is None or data_cfg == "":
        for exp_name, cfg_path in EXPERIMENT_DATA.items():
            if exp_name in str(weights_path):
                data_cfg = cfg_path
                logger.info("Auto-detected data config for %s: %s", exp_name, data_cfg)
                break
        if data_cfg is None or data_cfg == "":
            logger.warning("No data config specified or auto-detected; using model default")

    # Standard evaluation
    logger.info("=" * 60)
    std_metrics = run_eval(weights_path, data_cfg, augment=False, conf=args.conf)
    logger.info("=" * 60)

    # TTA evaluation
    tta_metrics = run_eval(weights_path, data_cfg, augment=True, conf=args.conf)
    logger.info("=" * 60)

    # Print comparison
    logger.info("")
    logger.info("=" * 60)
    logger.info("TTA COMPARISON — Test Set Results")
    logger.info("=" * 60)
    logger.info("%-20s %10s %12s %10s %8s", "Mode", "mAP50", "mAP50-95", "Precision", "Recall")
    logger.info("-" * 60)
    logger.info("%-20s %10.2f %12.2f %10.2f %8.2f",
                "Standard",
                std_metrics.get("mAP50", 0),
                std_metrics.get("mAP50-95", 0),
                std_metrics.get("Precision", 0),
                std_metrics.get("Recall", 0))
    logger.info("%-20s %10.2f %12.2f %10.2f %8.2f",
                "TTA",
                tta_metrics.get("mAP50", 0),
                tta_metrics.get("mAP50-95", 0),
                tta_metrics.get("Precision", 0),
                tta_metrics.get("Recall", 0))

    if std_metrics.get("mAP50", 0) and tta_metrics.get("mAP50", 0):
        logger.info("-" * 60)
        logger.info("%-20s %+9.2f %+11.2f %+9.2f %+7.2f",
                    "Δ (TTA - Std)",
                    tta_metrics.get("mAP50", 0) - std_metrics.get("mAP50", 0),
                    tta_metrics.get("mAP50-95", 0) - std_metrics.get("mAP50-95", 0),
                    tta_metrics.get("Precision", 0) - std_metrics.get("Precision", 0),
                    tta_metrics.get("Recall", 0) - std_metrics.get("Recall", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
