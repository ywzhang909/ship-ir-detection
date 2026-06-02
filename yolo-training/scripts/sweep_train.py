"""
Wandb sweep entry point for YOLO hyperparameter search.
Updated to support preprocessing, NWD loss, and optimized configs.

Called by `wandb agent <SWEEP_ID>`. Reads hyperparams from wandb.config,
overrides the base training config, and runs YOLO training.

Usage:
    uv run wandb sweep configs/sweep_nslsr.yaml        # init sweep
    uv run wandb agent <SWEEP_ID>                      # start one trial
"""

import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset presets (use optimized configs)
DATASET_CONFIGS = {
    "nslsr-swir": {"data": "configs/data_nslsr_swir.yaml", "train": "configs/train_nslsr_optimized.yaml"},
    "nslsr-lwir": {"data": "configs/data_nslsr_lwir.yaml", "train": "configs/train_nslsr_optimized.yaml"},
    "ship":       {"data": "configs/data.yaml",             "train": "configs/train_ship_optimized.yaml"},
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


def apply_preprocessing(data_cfg: dict, method: str) -> dict:
    """If preprocessing is enabled, cache processed images and return updated data_cfg."""
    if not method or method == "none":
        return data_cfg

    try:
        from preprocessing_module import get_preprocessing_pipeline  # noqa
        
        pipeline, _, _ = get_preprocessing_pipeline(method)
        logger.info("Applying preprocessing: %s", method)

        # Cache preprocessed images
        cache_root = PROJECT_ROOT / "data" / f"preprocessed_{method}"
        cache_root.mkdir(parents=True, exist_ok=True)

        new_cfg = dict(data_cfg)
        data_root = Path(data_cfg["path"])
        out_root = cache_root
        new_cfg["path"] = str(out_root.resolve())

        for split in ["train", "val", "test"]:
            src_dir = data_root / split / "images"
            if not src_dir.exists():
                continue
            dst_dir = out_root / split / "images"
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy labels
            src_label_dir = data_root / split / "labels"
            if src_label_dir.exists():
                dst_label_dir = out_root / split / "labels"
                dst_label_dir.mkdir(parents=True, exist_ok=True)
                for lbl in src_label_dir.iterdir():
                    if lbl.suffix == ".txt":
                        import shutil
                        shutil.copy2(lbl, dst_label_dir / lbl.name)

            # Preprocess images
            import cv2
            from tqdm import tqdm

            image_paths = sorted(src_dir.glob("*"))
            logger.info("Preprocessing %d images for split '%s' ...", len(image_paths), split)
            for img_path in tqdm(image_paths, desc=f"  {split}"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                processed = pipeline(img)
                cv2.imwrite(str(dst_dir / img_path.name), processed)

            new_cfg[split] = f"{split}/images"

        logger.info("Preprocessing cache saved to: %s", out_root)
        return new_cfg
    except Exception as e:
        logger.warning("Preprocessing failed, falling back to raw: %s", e)
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
    use_nwd = sweep_config.pop("use_nwd", False)
    preprocessing = sweep_config.pop("preprocessing", "none")

    # Load base training config
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["nslsr-swir"])
    train_cfg_path = PROJECT_ROOT / dataset_config["train"]
    data_cfg_path = PROJECT_ROOT / dataset_config["data"]

    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load data config
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    # Apply preprocessing (cache images)
    if preprocessing and preprocessing != "none":
        data_cfg = apply_preprocessing(data_cfg, preprocessing)

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

    # Log preprocessing info to wandb
    if preprocessing and preprocessing != "none":
        from preprocessing_module import AVAILABLE_PREPROCESS
        desc = AVAILABLE_PREPROCESS.get(preprocessing, preprocessing)
        wandb.run.summary["preprocessing_method"] = preprocessing
        wandb.run.summary["preprocessing_description"] = desc

    # Log NWD setting
    if use_nwd:
        wandb.run.summary["use_nwd"] = True

    # Resolve data path
    resolve_data_path(data_cfg, PROJECT_ROOT)

    logger.info("=" * 60)
    logger.info("SWEEP TRIAL: %s", wandb.run.id)
    logger.info("Dataset: %s", dataset)
    logger.info("Model: %s", model_name)
    logger.info("Preprocessing: %s", preprocessing)
    logger.info("NWD loss: %s", use_nwd)
    logger.info("Epochs: %d, Batch: %d, Imgsz: %d", epochs, batch, imgsz)
    logger.info("Sweep params: %s", sweep_config)
    logger.info("=" * 60)

    # Load and optionally patch model with NWD loss
    from ultralytics import YOLO

    model = YOLO(model_name)

    if use_nwd:
        try:
            from custom_loss import TinyObjectDetectionLoss
            model._custom_loss_fn = lambda m: TinyObjectDetectionLoss(
                m, iou_ratio=0.5, nwd_constant=12.8, fl_gamma=1.5
            )
            logger.info("NWD loss patched for small object detection.")
        except ImportError:
            logger.warning("custom_loss.py not found. NWD loss disabled.")

    # Train
    results = model.train(
        data=str(data_cfg_path) if preprocessing == "none" else data_cfg,
        **cfg,
    )

    # Final validation metrics
    logger.info("Training complete. Running validation...")
    metrics = model.val()
    if metrics and hasattr(metrics, "box"):
        logger.info("mAP50-95: %.4f", metrics.box.map)
        logger.info("mAP50:    %.4f", metrics.box.map50)

        # Log to wandb
        wandb.log({
            "epoch": cfg.get("epochs", epochs),
            "final/mAP50-95": metrics.box.map,
            "final/mAP50": metrics.box.map50,
            "final/Precision": metrics.box.mp,
            "final/Recall": metrics.box.mr,
        })

    # Export to ONNX
    try:
        logger.info("Exporting model to ONNX...")
        success = model.export(format="onnx", imgsz=cfg.get("imgsz", 640))
        if success:
            logger.info("ONNX export successful: %s", success)
    except Exception as e:
        logger.warning("ONNX export skipped: %s", e)

    logger.info("Sweep trial %s complete.", wandb.run.id)


if __name__ == "__main__":
    main()
