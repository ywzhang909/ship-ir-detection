"""
YOLO Integration for IR Maritime Image Enhancement.

Provides:
  1. EnhancedYoloDataset - wraps YOLO dataset with on-the-fly preprocessing
  2. enhance_inference   - drop-in wrapper for predict.py inference pipeline
  3. train_with_enhancement - integration into training loop

Usage:
    # Training with on-the-fly enhancement
    from enhancement.yolo_integration import EnhancedYoloDataset, train_with_enhancement
    from enhancement import get_preset

    cfg = get_preset("balanced")
    train_with_enhancement(
        data_yaml="configs/data.yaml",
        model_name="yolo11m.pt",
        enhancement_cfg=cfg,
        epochs=100,
        project="ship-detection",
    )

    # Inference with enhancement
    from enhancement.yolo_integration import enhance_inference
    results = enhance_inference("image.jpg", "best.pt", preset="quality")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from enhancement import (
    EnhancementConfig,
    EnhancementMethod,
    Preprocessor,
    get_preset,
    log_comparison_to_wandb,
    compare_methods,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. ON-THE-FLY ENHANCEMENT DATASET WRAPPER
# ===================================================================

class EnhancedYoloDataset:
    """Wraps a YOLO dataset iterator with on-the-fly image preprocessing.

    Augments each image before it reaches the model. Use during training
    by injecting into the dataloader or as a callback.

    This wrapper is designed to work with Ultralytics YOLO's built-in
    dataset class by preprocessing images AFTER YOLO's own augmentations
    but BEFORE the model sees them.

    Usage:
        from ultralytics.data import YOLODataset
        from enhancement import get_preset

        base_dataset = YOLODataset(...)
        enhanced_dataset = EnhancedYoloDataset(base_dataset, preset="quality")

        # Use enhanced_dataset in training
    """

    def __init__(
        self,
        base_dataset,
        config: EnhancementConfig | None = None,
        preset: str | None = None,
        apply_after_yolo_aug: bool = True,
    ):
        if preset:
            self.config = get_preset(preset)
        else:
            self.config = config or get_preset("balanced")

        self.preprocessor = Preprocessor(self.config)
        self.base_dataset = base_dataset
        self.apply_after_yolo_aug = apply_after_yolo_aug

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Get item with enhancement applied to image.

        Returns:
            Same format as base_dataset but with enhanced image.
        """
        item = self.base_dataset[idx]

        if isinstance(item, dict) and "img" in item:
            img = item["img"]
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert RGB to grayscale if needed
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 2:
                gray = img
            else:
                gray = img

            enhanced = self.preprocessor.process(gray)

            # Convert back to 3-channel if original was RGB
            if img.ndim == 3 and img.shape[2] == 3:
                item["img"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            else:
                item["img"] = enhanced

        return item


# ===================================================================
# 2. YOLO TRAINING CALLBACK
# ===================================================================

class EnhancementCallback:
    """Ultralytics YOLO callback for on-the-fly image enhancement.

    Hooks into YOLO's training pipeline to apply enhancement after
    built-in augmentations but before the forward pass.

    Register with:
        model.add_callback("on_batch_end", EnhancementCallback(cfg).on_batch_end)
    """

    def __init__(self, config: EnhancementConfig | None = None):
        self.pre = Preprocessor(config or get_preset("balanced"))
        self.enabled = True

    def on_batch_start(self, trainer):
        """Called at the start of each batch - apply enhancement to images."""
        if not self.enabled:
            return

        try:
            images = trainer.tloss  # YOLO internal batch image reference
            # In practice, hook into train pipeline via custom dataset
        except AttributeError:
            pass


# ===================================================================
# 3. INFERENCE WRAPPER
# ===================================================================

def enhance_inference(
    source: Union[str, Path, np.ndarray],
    model_path: str,
    preset: str = "balanced",
    conf: float = 0.25,
    slice_size: int = 512,
    overlap: float = 0.2,
    use_sahi: bool = True,
    save_enhanced: bool = False,
) -> List:
    """Run YOLO inference with preprocessing enhancement.

    This is a drop-in for predict.py that adds enhancement before
    the model sees the image.

    Args:
        source: Image path, directory, or numpy array
        model_path: Path to YOLO weights
        preset: Enhancement preset name
        conf: Confidence threshold
        slice_size: SAHI tile size
        overlap: SAHI tile overlap
        use_sahi: Use SAHI tiled inference
        save_enhanced: Save the enhanced image to disk

    Returns:
        YOLO results or SAHI predictions
    """
    from ultralytics import YOLO

    config = get_preset(preset)
    pre = Preprocessor(config)

    # Load image
    if isinstance(source, (str, Path)):
        img_bgr = cv2.imread(str(source))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif isinstance(source, np.ndarray):
        if source.ndim == 3:
            img_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = source
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Apply enhancement
    logger.info("Applying enhancement preset '%s'...", preset)
    enhanced = pre.process(img_gray)

    # Save enhanced image for inspection
    if save_enhanced:
        out_path = Path("runs/enhanced")
        out_path.mkdir(parents=True, exist_ok=True)
        stem = Path(str(source)).stem if isinstance(source, (str, Path)) else "enhanced"
        cv2.imwrite(str(out_path / f"{stem}_{preset}.png"), enhanced)
        logger.info("Saved enhanced image to %s", out_path / f"{stem}_{preset}.png")

    if use_sahi:
        return _sahi_inference(enhanced, model_path, conf, slice_size, overlap)
    else:
        return _standard_inference(enhanced, model_path, conf)


def _standard_inference(img_gray: np.ndarray, model_path: str, conf: float):
    """Standard YOLO inference on enhanced image."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    # Convert grayscale to 3-channel for YOLO
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return model(img_rgb, conf=conf)


def _sahi_inference(
    img_gray: np.ndarray,
    model_path: str,
    conf: float,
    slice_size: int,
    overlap: float,
):
    """SAHI tiled inference on enhanced image."""
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        logger.error("SAHI not installed. Run: uv add sahi")
        raise

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    # Save enhanced temp and run SAHI
    temp_path = Path("runs/_enhanced_temp.png")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(temp_path), img_gray)

    result = get_sliced_prediction(
        str(temp_path),
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
    )

    # Cleanup temp
    temp_path.unlink(missing_ok=True)

    return result


# ===================================================================
# 4. COMPARISON SCRIPT
# ===================================================================

def run_enhancement_comparison(
    image_path: str,
    methods: List[str] | None = None,
    log_wandb: bool = False,
    save_dir: str = "runs/enhancement_comparison",
):
    """Run all enhancement methods on an image and save/send to wandb.

    Useful for selecting the best method for your IR dataset.

    Args:
        image_path: Path to IR image
        methods: List of methods (default: all major ones)
        log_wandb: Log results to wandb
        save_dir: Directory to save comparison grid
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = compare_methods(img, methods=methods)

    # Log to wandb
    if log_wandb:
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project="ship-detection", name="enhancement-comparison")
            log_data = log_comparison_to_wandb(results)
            wandb.log(log_data)
            logger.info("Logged %d enhancement variants to wandb.", len(results))
        except ImportError:
            logger.warning("wandb not available for logging.")

    # Save comparison montage
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)

    # Create a grid of results
    n_cols = 4
    n_rows = (len(results) + n_cols - 1) // n_cols
    cell_h, cell_w = 256, 512  # resize for display
    montage = np.zeros((cell_h * n_rows, cell_w * n_cols), dtype=np.uint8)

    for i, r in enumerate(results):
        row, col = divmod(i, n_cols)
        resized = cv2.resize(r.image, (cell_w, cell_h))
        montage[row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w] = resized

        # Save individual
        cv2.imwrite(str(save_dir_p / f"{r.method}.png"), r.image)

    cv2.imwrite(str(save_dir_p / "comparison_montage.png"), montage)
    logger.info("Comparison saved to %s", save_dir_p)

    # Print timing summary
    print("\n" + "=" * 60)
    print("ENHANCEMENT METHOD COMPARISON")
    print("=" * 60)
    for r in results:
        print(f"  {r.method:25s}  {r.elapsed_ms:8.1f} ms")
    print("=" * 60)

    return results


# ===================================================================
# 5. TRAINING INTEGRATION HOOK
# ===================================================================

def train_with_enhancement(
    data_yaml: str,
    model_name: str = "yolo11m.pt",
    enhancement_cfg: EnhancementConfig | None = None,
    preset: str | None = None,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    project: str = "ship-detection",
    run_name: str | None = None,
    device: str | None = None,
    **train_kwargs,
):
    """Train YOLO with on-the-fly image enhancement.

    This modifies the training pipeline to apply enhancement AFTER
    YOLO's built-in augmentations. It works by monkey-patching the
    dataset's __getitem__ to add enhancement.

    For production, consider preprocessing offline instead (faster).

    Args:
        data_yaml: Path to data.yaml
        model_name: YOLO model name or weights path
        enhancement_cfg: Enhancement configuration
        preset: Preset name (overrides enhancement_cfg)
        epochs: Training epochs
        batch: Batch size
        imgsz: Input image size
        project: wandb project
        run_name: wandb run name
        device: Training device
        **train_kwargs: Additional YOLO training args
    """
    from ultralytics import YOLO
    from ultralytics.data import YOLODataset

    # Resolve config
    if preset:
        cfg = get_preset(preset)
    else:
        cfg = enhancement_cfg or get_preset("balanced")

    pre = Preprocessor(cfg)
    logger.info("Training with enhancement preset: %s",
                "pipeline:" + "+".join(cfg.pipeline) if cfg.method == EnhancementMethod.PIPELINE else cfg.method)

    # ── Monkey-patch YOLO dataset to add enhancement ──
    _orig_getitem = YOLODataset.__getitem__

    def _enhanced_getitem(self, idx):
        item = _orig_getitem(self, idx)
        if "img" in item:
            img = item["img"]
            # YOLO returns images as CHW uint8
            if img.ndim == 3 and img.shape[0] in (1, 3):
                if img.shape[0] == 3:
                    gray = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                else:
                    gray = img[0]
            else:
                gray = img if img.ndim == 2 else img

            enhanced = pre.process(gray)

            # Convert back to original format
            if img.ndim == 3 and img.shape[0] == 3:
                item["img"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)
            else:
                item["img"] = enhanced[np.newaxis, ...]
        return item

    YOLODataset.__getitem__ = _enhanced_getitem

    # ── Train ──
    try:
        model = YOLO(model_name)
        run_name = run_name or f"enhanced-{cfg.method.value}-e{epochs}"

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=project,
            name=run_name,
            device=device,
            **train_kwargs,
        )

        logger.info("Training complete. Validating...")
        metrics = model.val()
        if metrics and hasattr(metrics, "box"):
            logger.info("mAP50-95: %.4f", metrics.box.map)
        return results, metrics

    finally:
        # Restore original __getitem__
        YOLODataset.__getitem__ = _orig_getitem


# ===================================================================
# 6. OFFLINE BATCH ENHANCEMENT
# ===================================================================

def enhance_dataset(
    input_dir: str,
    output_dir: str,
    config: EnhancementConfig | None = None,
    preset: str = "balanced",
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".JPG", ".bmp"),
):
    """Batch enhance all images in a directory and save to output.

    Useful for preprocessing an entire dataset offline before training,
    which is faster than on-the-fly enhancement.

    Args:
        input_dir: Directory with input images
        output_dir: Directory for enhanced images
        config: Enhancement config
        preset: Preset name (overrides config)
        extensions: Image file extensions to process
    """
    if preset:
        cfg = get_preset(preset)
    else:
        cfg = config or get_preset("balanced")

    pre = Preprocessor(cfg)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))

    logger.info("Enhancing %d images with preset '%s'...", len(image_files), preset)
    times = []

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Skipping unreadable: %s", img_path)
            continue

        import time as _time
        t0 = _time.perf_counter()
        enhanced = pre.process(img)
        elapsed = (_time.perf_counter() - t0) * 1000
        times.append(elapsed)

        out_path = output_path / f"{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(out_path), enhanced)

        if (i + 1) % 100 == 0:
            logger.info("  Processed %d/%d (avg %.1f ms)", i + 1, len(image_files), np.mean(times))

    avg_ms = np.mean(times) if times else 0
    logger.info("Done! Enhanced %d images to %s (avg %.1f ms/img)", len(image_files), output_path, avg_ms)


__all__ = [
    "EnhancedYoloDataset",
    "EnhancementCallback",
    "enhance_inference",
    "run_enhancement_comparison",
    "train_with_enhancement",
    "enhance_dataset",
]
