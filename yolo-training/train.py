"""
YOLO training entry point for ship identification.

Supports multiple datasets (ship, nslsr-swir, nslsr-lwir) with wandb logging.

Usage:
    uv run python train.py [--dataset ship|nslsr-swir|nslsr-lwir]
                           [--wandb] [--run-name NAME]
                           [--data CONFIG_PATH] [--model MODEL_NAME]
                           [--epochs N] [--batch N] [--imgsz N]

Examples:
    # Train Ship dataset with defaults (yolo11m, 200 epochs)
    uv run python train.py

    # Train NSLSR SWIR with wandb logging
    uv run python train.py --dataset nslsr-swir --wandb --run-name nslsr-swir-v1

    # Quick test (small model, 1 epoch)
    uv run python train.py --model yolo11n.pt --epochs 1 --batch 4

    # Resume from checkpoint
    uv run python train.py --model runs/detect/train/weights/last.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset presets: maps dataset name → (data config, train config)
DATASET_CONFIGS = {
    "ship": {"data": "configs/data.yaml", "train": "configs/train.yaml"},
    "nslsr-swir": {"data": "configs/data_nslsr_swir.yaml", "train": "configs/train_nslsr.yaml"},
    "nslsr-lwir": {"data": "configs/data_nslsr_lwir.yaml", "train": "configs/train_nslsr.yaml"},
}


def resolve_data_path(data_cfg: dict) -> dict:
    """Resolve the data.yaml path to an absolute path (Windows compatibility)."""
    cfg_path = data_cfg.get("path", "")
    if cfg_path and not Path(cfg_path).exists():
        # Try resolving relative to project root
        alt = PROJECT_ROOT / cfg_path
        if alt.exists():
            data_cfg["path"] = str(alt.resolve())
        else:
            # Try configs/data.yaml
            alt2 = PROJECT_ROOT / "configs" / "data.yaml"
            if alt2.exists():
                logger.info("Using data config from: %s", alt2)
                with open(alt2) as f:
                    data_cfg.update(yaml.safe_load(f))
    return data_cfg


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for ship identification")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to dataset YAML config (overrides --dataset)",
    )
    parser.add_argument(
        "--dataset", type=str, default="ship",
        choices=["ship", "nslsr-swir", "nslsr-lwir"],
        help="Dataset preset (default: ship)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name or path (overrides configs/train.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., 'cpu', '0')")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Wandb run name (auto-generated if not set)",
    )
    args = parser.parse_args()

    # Determine config files based on --dataset or --data
    dataset = args.dataset
    if args.data:
        # Explicit --data overrides dataset preset
        data_cfg_path = Path(args.data)
        if not data_cfg_path.exists():
            data_cfg_path = PROJECT_ROOT / args.data
        train_cfg_path = PROJECT_ROOT / "configs" / "train.yaml"
    else:
        # Use dataset preset
        dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["ship"])
        data_cfg_path = PROJECT_ROOT / dataset_config["data"]
        train_cfg_path = PROJECT_ROOT / dataset_config["train"]

    # Load training config
    if not train_cfg_path.exists():
        logger.error("Training config not found: %s", train_cfg_path)
        sys.exit(1)

    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load data config
    if not data_cfg_path.exists():
        logger.error("Data config not found: %s", data_cfg_path)
        sys.exit(1)

    # Resolve paths
    resolve_data_path(cfg)

    # Override from command line
    if args.model:
        cfg["model"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz

    # Wandb integration
    if args.wandb:
        run_name = args.run_name or f"{dataset}-{Path(cfg.get('model', 'yolo11m')).stem}"
        cfg["project"] = "ship-detection"
        cfg["name"] = run_name
        logger.info("Wandb logging enabled — project: ship-detection, run: %s", run_name)

    model_name = cfg.pop("model")

    logger.info("=" * 60)
    logger.info("SHIP IDENTIFICATION YOLO TRAINING")
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset)
    logger.info("Model: %s", model_name)
    logger.info("Data config: %s", data_cfg_path)
    logger.info("Train config: %s", train_cfg_path)
    logger.info("Epochs: %d", cfg.get("epochs", 200))
    logger.info("Batch: %d", cfg.get("batch", 16))
    logger.info("Image size: %d", cfg.get("imgsz", 1280))
    logger.info("Device: %s", args.device or "auto")
    logger.info("Wandb: %s", "enabled" if args.wandb else "disabled")
    logger.info("=" * 60)

    # Initialize model
    if args.resume:
        logger.info("Resuming from last checkpoint...")
        model = YOLO(model_name)  # should be a checkpoint path when resuming
    else:
        model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_cfg_path),
        device=args.device,
        **cfg,
    )

    # Validate
    logger.info("Training complete. Running validation...")
    metrics = model.val()
    if metrics and hasattr(metrics, "box"):
        logger.info("mAP50-95: %.4f", metrics.box.map)
        logger.info("mAP50:    %.4f", metrics.box.map50)
        logger.info("Precision: %.4f", metrics.box.mp)
        logger.info("Recall:   %.4f", metrics.box.mr)
    else:
        logger.info("Validation metrics: %s", metrics)

    # Export to ONNX
    logger.info("Exporting model to ONNX...")
    success = model.export(format="onnx", imgsz=cfg.get("imgsz", 1280))
    if success:
        logger.info("ONNX export successful: %s", success)

    logger.info("Done! Model weights saved in runs/detect/train/")


if __name__ == "__main__":
    main()
