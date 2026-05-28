"""
Wandb sweep entry point for YOLO hyperparameter search.

Called by `wandb agent <SWEEP_ID>`. Reads hyperparams from wandb.config,
overrides the base training config, and runs YOLO training.

Usage:
    uv run wandb sweep configs/sweep_nslsr.yaml        # init sweep
    uv run wandb agent <SWEEP_ID>                      # start one trial
"""

import copy
import logging
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default config paths
DATASET_CONFIGS = {
    "nslsr-swir": {"data": "configs/data_nslsr_swir.yaml", "train": "configs/train_nslsr.yaml"},
    "nslsr-lwir": {"data": "configs/data_nslsr_lwir.yaml", "train": "configs/train_nslsr.yaml"},
}


def resolve_data_path(data_cfg: dict, cfg_dir: Path) -> dict:
    """Resolve the data.yaml path to an absolute path."""
    cfg_path = data_cfg.get("path", "")
    if cfg_path and not Path(cfg_path).exists():
        alt = cfg_dir / cfg_path
        if alt.exists():
            data_cfg["path"] = str(alt.resolve())
        else:
            alt2 = cfg_dir / "configs" / "data.yaml"
            if alt2.exists():
                with open(alt2) as f:
                    data_cfg.update(yaml.safe_load(f))
    return data_cfg


def main():
    import wandb

    wandb.init()
    sweep_config = dict(wandb.config)

    dataset = sweep_config.pop("dataset", "nslsr-swir")
    model_name = sweep_config.pop("model", "yolo11m.pt")
    epochs = sweep_config.pop("epochs", 100)
    batch = sweep_config.pop("batch", 16)
    imgsz = sweep_config.pop("imgsz", 640)

    # Load base training config
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["nslsr-swir"])
    train_cfg_path = PROJECT_ROOT / dataset_config["train"]
    data_cfg_path = PROJECT_ROOT / dataset_config["data"]

    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override with sweep parameters
    cfg.update(sweep_config)
    cfg["model"] = model_name
    cfg["epochs"] = epochs
    cfg["batch"] = batch
    cfg["imgsz"] = imgsz

    # Always enable wandb project logging
    cfg["project"] = "ship-detection"
    cfg["name"] = f"sweep-{dataset}-{wandb.run.id}"
    cfg["exist_ok"] = True
    cfg["save"] = True
    cfg["save_period"] = 10

    # Resolve data path
    resolve_data_path(cfg, PROJECT_ROOT)

    logger.info("=" * 60)
    logger.info("SWEEP TRIAL: %s", wandb.run.id)
    logger.info("Dataset: %s", dataset)
    logger.info("Model: %s", model_name)
    logger.info("Epochs: %d, Batch: %d, Imgsz: %d", epochs, batch, imgsz)
    logger.info("Sweep params: %s", sweep_config)
    logger.info("Data config: %s", data_cfg_path)
    logger.info("=" * 60)

    # Initialize and train
    model = YOLO(model_name)
    results = model.train(
        data=str(data_cfg_path),
        **cfg,
    )

    # Final validation metrics
    logger.info("Training complete. Running validation...")
    metrics = model.val()
    if metrics and hasattr(metrics, "box"):
        logger.info("mAP50-95: %.4f", metrics.box.map)
        logger.info("mAP50:    %.4f", metrics.box.map50)
        # Log summary to wandb
        wandb.log({
            "epoch": cfg.get("epochs", epochs),
            "final/mAP50-95": metrics.box.map,
            "final/mAP50": metrics.box.map50,
            "final/Precision": metrics.box.mp,
            "final/Recall": metrics.box.mr,
        })

    # Export to ONNX
    logger.info("Exporting model to ONNX...")
    success = model.export(format="onnx", imgsz=cfg.get("imgsz", 640))
    if success:
        logger.info("ONNX export successful: %s", success)

    logger.info("Sweep trial %s complete.", wandb.run.id)


if __name__ == "__main__":
    main()
