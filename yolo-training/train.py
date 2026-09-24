"""
Enhanced YOLO training entry point for IR ship identification.

Features:
  - Multi-dataset support (ship/nslsr-swir/nslsr-lwir)
  - Integrated IR maritime preprocessing pipeline (sea clutter, enhancement)
  - Learnable spatial–frequency fusion network (SpatialFrequencyFusion)
  - Deep wandb logging (preprocessing visuals, config, loss curves, metrics)
  - Sweep-compatible (via --config-override)

Usage:
    # Basic training
    uv run python train.py --dataset nslsr-lwir --wandb

    # With preprocessing (cache preprocessed images)
    uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method tophat_clahe --wandb

    # With learnable fusion network (requires awb_dual_fusion preprocessing)
    uv run python train.py --dataset nslsr-lwir --preprocess --preprocess-method awb_dual_fusion --use-fusion --wandb

    # Resume
    uv run python train.py --model runs/detect/train/weights/last.pt --resume

    # Sweep runner
    uv run python scripts/sweep_train.py
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import preprocessing presets from the module
from preprocessing_module import (
    AVAILABLE_PREPROCESS,
    PreprocessingPipelineDual,
    PreprocessingPipeline,
    PreprocessingConfig,
    SeaClutterMethod,
    EnhancementMethod,
    get_preprocessing_pipeline,
)
from fusion_module import (
    SpatialFrequencyFusion,
    make_fusion_trainer_class,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ship": {"data": "configs/data.yaml", "train": "configs/train_ship_optimized.yaml"},
    "nslsr-swir": {
        "data": "configs/data_nslsr_swir.yaml",
        "train": "configs/train_nslsr_optimized.yaml",
    },
    "nslsr-lwir": {
        "data": "configs/data_nslsr_lwir.yaml",
        "train": "configs/train_nslsr_optimized.yaml",
    },
}

# ===================================================================
# Preprocessing Jitter (T25: randomise preprocess params per image)
# ===================================================================

SUPPORTED_JITTER_ATTRS = [
    "clahe_clip_limit",
    "clahe_tile_grid",
    "tophat_kernel",
    "butterworth_cutoff_low",
    "butterworth_cutoff_high",
    "unsharp_strength",
    "wavelet_scale_ll",
    "wavelet_scale_lh",
    "wavelet_scale_hh",
]


def _jitter_preprocess_params(config: PreprocessingConfig, jitter_mag: float):
    """Apply random jitter to a single ``PreprocessingConfig`` in-place.

    Each parameter is independently sampled from
    ``[base*(1-jitter_mag), base*(1+jitter_mag)]``, with clamping to
    reasonable ranges so the pipeline remains stable.

    Args:
        config: Config to mutate (modified in-place).
        jitter_mag: Relative jitter magnitude (0.0 = off, 0.3 = ±30%).
    """
    # -- CLAHE clip_limit ------------------------------------------------
    base = config.clahe_clip_limit
    config.clahe_clip_limit = round(
        max(0.5, random.uniform(base * (1.0 - jitter_mag), base * (1.0 + jitter_mag))), 2
    )

    # -- CLAHE tile_grid (square, power-of-2-ish, 4-32) ------------------
    base_tile = config.clahe_tile_grid[0]
    tile = int(random.uniform(base_tile * (1.0 - jitter_mag), base_tile * (1.0 + jitter_mag)))
    tile = min(max(4, tile), 32)
    config.clahe_tile_grid = (tile, tile)

    # -- Tophat / bottomhat kernel (odd number, 3-61) --------------------
    if config.sea_clutter_method in (
        SeaClutterMethod.TOPHAT,
        SeaClutterMethod.BOTTOMHAT,
        SeaClutterMethod.TOPHAT_BOTTOMHAT,
    ):
        base = config.tophat_kernel
        kernel = int(
            random.uniform(base * (1.0 - jitter_mag), base * (1.0 + jitter_mag))
        )
        kernel = max(3, min(61, kernel | 1))  # ensure odd, clamp
        config.tophat_kernel = kernel

    # -- Butterworth BPF cutoffs -----------------------------------------
    if config.sea_clutter_method == SeaClutterMethod.BUTTERWORTH_BPF:
        low = config.butterworth_cutoff_low
        config.butterworth_cutoff_low = round(
            max(2.0, random.uniform(low * (1.0 - jitter_mag), low * (1.0 + jitter_mag))), 1
        )
        high = config.butterworth_cutoff_high
        config.butterworth_cutoff_high = round(
            min(200.0, random.uniform(high * (1.0 - jitter_mag), high * (1.0 + jitter_mag))), 1
        )

    # -- Wavelet sub-band scales -----------------------------------------
    if config.sea_clutter_method == SeaClutterMethod.WAVELET:
        for attr in ("wavelet_scale_ll", "wavelet_scale_lh", "wavelet_scale_hh"):
            base = getattr(config, attr)
            val = random.uniform(base * (1.0 - jitter_mag), base * (1.0 + jitter_mag))
            val = max(0.05, min(5.0, val))
            setattr(config, attr, round(val, 2))

    # -- Unsharp masking strength ----------------------------------------
    us = config.unsharp_strength
    config.unsharp_strength = round(
        max(0.1, min(2.0, random.uniform(us * (1.0 - jitter_mag), us * (1.0 + jitter_mag)))), 2
    )


def _make_preprocess_jitter_wrapper(pipeline, jitter_mag: float):
    """Wrap a ``PreprocessingPipeline`` or ``PreprocessingPipelineDual``
    so that each ``__call__`` applies random jitter to the internal config.

    Args:
        pipeline: ``PreprocessingPipeline`` or ``PreprocessingPipelineDual``.
        jitter_mag: Relative jitter magnitude (0.0 = off).

    Returns:
        Callable with the same interface as the original pipeline.
    """
    # Attributes that _jitter_preprocess_params touches — snapshot only these
    # to avoid deep-copy overhead (60K+ images).
    _JITTER_ATTRS = (
        "clahe_clip_limit", "clahe_tile_grid",
        "tophat_kernel", "butterworth_cutoff_low", "butterworth_cutoff_high",
        "unsharp_strength",
        "wavelet_scale_ll", "wavelet_scale_lh", "wavelet_scale_hh",
    )

    def _snapshot(cfg):
        return {a: getattr(cfg, a) for a in _JITTER_ATTRS if hasattr(cfg, a)}

    def _restore(cfg, snap):
        for a, v in snap.items():
            setattr(cfg, a, v)

    if isinstance(pipeline, PreprocessingPipelineDual):
        _spatial_snap = _snapshot(pipeline.spatial_pipeline.config)
        _freq_snap = _snapshot(pipeline.freq_pipeline.config)

        def _dual_wrapper(img):
            seed = random.randint(0, 2**31)
            rng_spatial = random.Random(seed)
            rng_freq = random.Random(seed ^ 0xA5A5A5A5)
            # Temporarily swap global random with deterministic RNGs
            _orig_rand = random.random
            _orig_uniform = random.uniform
            _orig_randint = random.randint
            try:
                random.random = rng_spatial.random
                random.uniform = rng_spatial.uniform
                random.randint = rng_spatial.randint
                _jitter_preprocess_params(pipeline.spatial_pipeline.config, jitter_mag)
                random.random = rng_freq.random
                random.uniform = rng_freq.uniform
                random.randint = rng_freq.randint
                _jitter_preprocess_params(pipeline.freq_pipeline.config, jitter_mag)
                return pipeline(img)
            finally:
                random.random = _orig_rand
                random.uniform = _orig_uniform
                random.randint = _orig_randint
                _restore(pipeline.spatial_pipeline.config, _spatial_snap)
                _restore(pipeline.freq_pipeline.config, _freq_snap)
        return _dual_wrapper

    # Single (non-dual) pipeline
    _snap = _snapshot(pipeline.config)

    def _single_wrapper(img):
        _jitter_preprocess_params(pipeline.config, jitter_mag)
        result = pipeline(img)
        _restore(pipeline.config, _snap)
        return result
    return _single_wrapper


# ===================================================================
# Preprocessing Cache
# ===================================================================

def preprocess_and_cache(
    pipeline,
    data_cfg: dict,
    output_dir: str | Path,
    method_name: str,
) -> dict:
    """Apply preprocessing to all training images and cache the results.

    Creates a new data config pointing to the preprocessed images.

    Args:
        pipeline: PreprocessingPipeline instance.
        data_cfg: Dataset config dict with 'path', 'train', 'val', 'test' keys.
        output_dir: Root directory for cached preprocessed data (e.g.
            ``data/preprocessed_awb_dual_fusion/``).
        method_name: Name of preprocessing method (for reference only, the
            caller already encodes it in ``output_dir``).

    Returns:
        New data config dict pointing to cached images.
    """
    data_root = Path(data_cfg["path"])
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    new_cfg = dict(data_cfg)
    new_cfg["path"] = str(out_root.resolve())

    for split in ["train", "val", "test"]:
        src_dir = data_root / split / "images"
        if not src_dir.exists():
            logger.warning("Split %s not found at %s, skipping", split, src_dir)
            if split in new_cfg:
                new_cfg.pop(split, None)
            continue

        dst_dir = out_root / split / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels (unchanged)
        src_label_dir = data_root / split / "labels"
        if src_label_dir.exists():
            dst_label_dir = out_root / split / "labels"
            dst_label_dir.mkdir(parents=True, exist_ok=True)
            for lbl in src_label_dir.iterdir():
                if lbl.suffix in (".txt",):
                    shutil.copy2(lbl, dst_label_dir / lbl.name)

        # Preprocess images
        image_paths = sorted(src_dir.glob("*"))
        logger.info("Preprocessing %d images for split '%s' ...", len(image_paths), split)

        for img_path in tqdm(image_paths, desc=f"  {split}"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("  Failed to read: %s", img_path)
                continue
            processed = pipeline(img)
            out_path = dst_dir / img_path.name
            cv2.imwrite(str(out_path), processed)

        # Update new config
        new_cfg[split] = f"{split}/images"

        # Log a few sample visualization grids
        sample_dir = out_root / "samples"
        sample_dir.mkdir(exist_ok=True)
        image_paths = sorted(
            (data_root / "train" / "images").glob("*"),
            key=lambda p: p.stat().st_size,
        )[:8]
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if isinstance(pipeline, PreprocessingPipelineDual):
                # Dual pipeline — save 3 components + composite separately
                spatial_gray = cv2.cvtColor(pipeline.spatial_pipeline(img), cv2.COLOR_BGR2GRAY)
                freq_gray = cv2.cvtColor(pipeline.freq_pipeline(img), cv2.COLOR_BGR2GRAY)
                orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Stack order matches preprocessing pipeline: BGR→RGB → [spatial, freq, orig]
                composite = np.stack([orig_gray, freq_gray, spatial_gray], axis=-1)

                vis_components = []
                for name, arr in [("original", orig_gray), ("freq", freq_gray),
                                  ("spatial", spatial_gray), ("composite", composite)]:
                    if arr.ndim == 2:
                        display = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    else:
                        display = arr
                    label = np.zeros((20, display.shape[1], 3), dtype=np.uint8)
                    cv2.putText(label, name, (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    vis_components.append(np.vstack([label, display]))
                grid = np.hstack(vis_components)
                cv2.imwrite(str(sample_dir / f"stages_{img_path.stem}.jpg"), grid)
            elif hasattr(pipeline, "visualize_stages"):
                stages = pipeline.visualize_stages(img)
                # Stack stages horizontally
                vis_panels = []
                for stage_name, stage_img in stages.items():
                    label = np.zeros((20, stage_img.shape[1], 3), dtype=np.uint8)
                    cv2.putText(label, stage_name, (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    # Ensure 3-channel for visualization
                    if stage_img.ndim == 2:
                        display = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
                    else:
                        display = stage_img
                    panel = np.vstack([label, display])
                    vis_panels.append(panel)
                grid = np.hstack(vis_panels) if vis_panels else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(sample_dir / f"stages_{img_path.stem}.jpg"), grid)
            else:
                # Jitter-wrapped function — skip per-stage visualization
                pass

    logger.info("Preprocessing cache saved to: %s", out_root)
    return new_cfg


# ===================================================================
# Wandb Helpers (using Ultralytics built-in integration)
# ===================================================================
# When `yolo settings wandb=True`, Ultralytics auto-initializes wandb
# and logs all training metrics, loss curves, hyperparameters, and
# validation results. We add preprocessing-specific logging via
# a YOLO callback that hooks into the auto-managed wandb run.


def _make_wandb_callback(preprocessing_info: dict, fusion_info: dict | None = None):
    """Create a YOLO callback that logs preprocessing & fusion info to Ultralytics' wandb.

    Args:
        preprocessing_info: Dict with keys 'method', 'description', 'params', 'sample_dir'.
        fusion_info: Optional dict with keys 'enabled', 'hidden_dim', 'lr_scale', 'n_params'.

    Returns a callback dict suitable for ``model.add_callback()``.
    """
    def on_train_start(trainer):
        """Log preprocessing samples and config to the auto-managed wandb run."""
        import wandb

        if wandb.run is None:
            logger.debug("No active wandb run (wandb=True not set in Ultralytics settings)")
            return

        # ── Preprocessing config as summary ──
        if preprocessing_info:
            method = preprocessing_info.get("method", "none")
            desc = preprocessing_info.get("description", "")

            wandb.run.summary["preprocessing_method"] = method
            wandb.run.summary["preprocessing_description"] = desc

            jitter = preprocessing_info.get("preprocess_jitter", 0.0)
            if jitter > 0.0:
                wandb.run.summary["preprocess_jitter"] = jitter

            params = preprocessing_info.get("params", {})
            if params:
                wandb.run.summary["preprocessing_params"] = str(params)

            # ── Preprocessing sample images ──
            sample_dir = preprocessing_info.get("sample_dir")
            if sample_dir and Path(sample_dir).exists():
                sample_images = sorted(Path(sample_dir).glob("*.jpg"))
                if sample_images:
                    media = [
                        wandb.Image(str(img), caption=img.stem)
                        for img in sample_images[:4]
                    ]
                    wandb.log({"preprocessing_samples": media}, step=0)

        # ── Fusion network config (individual queryable fields) ──
        if fusion_info and fusion_info.get("enabled"):
            wandb.run.summary["fusion/enabled"] = True
            wandb.run.summary["fusion/hidden_dim"] = fusion_info.get("hidden_dim")
            wandb.run.summary["fusion/num_layers"] = fusion_info.get("num_layers")
            wandb.run.summary["fusion/attention"] = fusion_info.get("attention")
            wandb.run.summary["fusion/lr_scale"] = fusion_info.get("lr_scale")
            wandb.run.summary["fusion/residual_target"] = fusion_info.get("residual_target")
            wandb.run.summary["fusion/n_params"] = fusion_info.get("n_params")

            # Log each field individually for queryability
            wandb.log({
                "fusion/hidden_dim": fusion_info.get("hidden_dim"),
                "fusion/num_layers": fusion_info.get("num_layers"),
                "fusion/attention": fusion_info.get("attention"),
                "fusion/lr_scale": fusion_info.get("lr_scale"),
                "fusion/residual_target": fusion_info.get("residual_target"),
                "fusion/n_params": fusion_info.get("n_params"),
            }, step=0)

        logger.debug("Preprocessing/fusion info logged to wandb via callback")

    return {"on_train_start": on_train_start}


def _finalize_wandb_safe():
    """Safely finalize any leftover wandb run, suppressing Windows path errors.

    Ultralytics' built-in ``on_train_end`` callback (``wb.py``) calls
    ``wb.run.finish()``, but on Windows the ``latest-run`` symlink created
    by ``wandb.init()`` can leave dangling file handles or raise
    ``OSError`` during artifact log and finalization.  This function
    catches those silently, so post-training val/export don't crash.
    """
    try:
        import wandb

        if wandb.run is not None:
            # On Windows, os.symlink may not be available or may fail,
            # leaving latest-run in a broken state.  Finish gracefully.
            try:
                wandb.run.finish()
            except Exception:
                pass
    except ImportError:
        pass


def _make_fusion_monitor_callback():
    """Create a YOLO callback that logs fusion module stats at each val epoch.

    Logs ``residual_alpha``, channel-wise output means, and the MSE between
    fusion output and input so we can verify the module is actually learning.
    """

    def on_val_epoch_end(trainer):
        import wandb

        if wandb.run is None:
            return

        model = trainer.model
        if model is None:
            return

        # The fusion module lives at model.model[0] wrapped in Sequential
        first = model.model[0] if hasattr(model, "model") else None
        if not isinstance(first, torch.nn.Sequential) or len(first) == 0:
            return
        fusion = first[0]
        from fusion_module import SpatialFrequencyFusion

        if not isinstance(fusion, SpatialFrequencyFusion):
            return

        # ── Log residual alpha ──
        alpha = fusion.residual_alpha.item()
        log_data: dict[str, float] = {"fusion/residual_alpha": alpha}

        # ── Output statistics ──
        device = fusion.conv1.weight.device
        try:
            loader = trainer.train_loader
            if loader is not None:
                sample_batch = next(iter(loader))["img"][:4].to(device)
                with torch.no_grad():
                    fused = fusion(sample_batch)

                log_data["fusion/output_mean_ch0"] = fused[:, 0].mean().item()
                log_data["fusion/output_mean_ch1"] = fused[:, 1].mean().item()
                log_data["fusion/output_mean_ch2"] = fused[:, 2].mean().item()
                log_data["fusion/output_std"] = fused.std().item()
                log_data["fusion/input_output_mse"] = F.mse_loss(fused, sample_batch).item()
        except Exception:
            pass

        # ── Gradient norms ──
        fusion_grad_norm = 0.0
        backbone_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.norm().item()
                if "model.0.0." in name:
                    fusion_grad_norm += g
                else:
                    backbone_grad_norm += g

        log_data["fusion/grad_norm"] = fusion_grad_norm
        log_data["fusion/backbone_grad_norm"] = backbone_grad_norm
        if backbone_grad_norm > 1e-8:
            log_data["fusion/grad_ratio"] = fusion_grad_norm / backbone_grad_norm

        wandb.log(log_data, step=trainer.epoch)

    return {"on_val_epoch_end": on_val_epoch_end}


def _make_augment_jitter_callback(jitter_magnitude: float):
    """Create a callback that randomly jitters augmentation parameters each epoch.

    Randomly samples augment params from ``[base*(1-jitter), base*(1+jitter)]``
    at each epoch start, exposing the model to varying augmentation levels.

    Args:
        jitter_magnitude: Relative jitter range (e.g., 0.3 = ±30%).

    Returns:
        Callback dict with ``on_epoch_start``.
    """
    import random

    # Augmentation attributes on the trainer that can be jittered
    JITTER_ATTRS = [
        "scale", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "shear",
    ]

    def on_epoch_start(trainer):
        if trainer.epoch is None or trainer.epoch < 3:
            return  # Skip warmup epochs

        jittered = {}
        for attr in JITTER_ATTRS:
            base = getattr(trainer, attr, None)
            if base is None or base == 0.0:
                continue
            low = base * (1.0 - jitter_magnitude)
            high = base * (1.0 + jitter_magnitude)
            new_val = random.uniform(low, high)
            # Clamp to reasonable ranges
            if attr == "scale":
                new_val = max(0.1, min(1.0, new_val))
            elif attr in ("hsv_h",):
                new_val = max(0.0, min(0.1, new_val))
            elif attr in ("hsv_s", "hsv_v"):
                new_val = max(0.0, min(0.5, new_val))
            elif attr in ("translate",):
                new_val = max(0.0, min(0.5, new_val))
            setattr(trainer, attr, new_val)
            jittered[attr] = (round(base, 4), round(new_val, 4))

        if jittered:
            logger.debug("Augment jitter [epoch %d]: %s", trainer.epoch,
                         {k: f"{v[0]}→{v[1]}" for k, v in jittered.items()})

    return {"on_epoch_start": on_epoch_start}


def _make_map95_early_stop_callback(tolerance: float = 1e-3, patience: int = 3):
    """Create an on_val_epoch_end callback that stops training when mAP50-95
    improvement stalls.

    More aggressive than YOLO's default patience=30 (which uses a combined
    fitness metric). This callback monitors mAP50-95 directly and triggers
    stop when improvement < `tolerance` for `patience` consecutive val epochs.

    Args:
        tolerance: Minimum absolute improvement in mAP50-95 to reset counter.
        patience: Consecutive non-improving val epochs before stopping.

    Returns:
        Dict with 'on_val_epoch_end' callback key.
    """
    best_map95 = -1.0
    stall_count = 0

    def on_val_epoch_end(trainer):
        nonlocal best_map95, stall_count

        # Extract current mAP50-95 from trainer metrics
        current_map95 = getattr(trainer, "map", None)
        if current_map95 is None:
            # Try trainer.metrics or trainer.validator metrics
            if hasattr(trainer, "metrics") and trainer.metrics is not None:
                current_map95 = trainer.metrics.get("metrics/mAP50-95(B)", None)
            if current_map95 is None and hasattr(trainer, "validator"):
                current_map95 = getattr(trainer.validator, "map", None)

        if current_map95 is None:
            return  # Cannot determine mAP50-95, skip

        improvement = current_map95 - best_map95
        if improvement > tolerance:
            best_map95 = current_map95
            stall_count = 0
        else:
            stall_count += 1
            logger.info(
                "EarlyStop [mAP50-95]: epoch=%d, current=%.4f, best=%.4f, "
                "improvement=%.4f (tolerance=%.4f), stall=%d/%d",
                trainer.epoch, current_map95, best_map95,
                improvement, tolerance, stall_count, patience,
            )
            if stall_count >= patience:
                logger.info(
                    "EarlyStop [mAP50-95]: Stopping at epoch %d — "
                    "mAP50-95=%.4f did not improve > %.4f for %d epochs",
                    trainer.epoch, current_map95, tolerance, patience,
                )
                trainer.stop = True

    return {"on_val_epoch_end": on_val_epoch_end}


def log_val_metrics_to_wandb(metrics):
    """Log validation metrics to an existing wandb run."""
    if metrics is None or not hasattr(metrics, "box"):
        return
    try:
        import wandb

        if wandb.run is None:
            return

        wandb.log({
            "val/mAP50-95": metrics.box.map,
            "val/mAP50": metrics.box.map50,
            "val/Precision": metrics.box.mp,
            "val/Recall": metrics.box.mr,
        })
        wandb.run.summary.update({
            "best_mAP50-95": metrics.box.map,
            "best_mAP50": metrics.box.map50,
        })
    except Exception:
        pass


# ===================================================================
# Main
# ===================================================================

def main():
    # ── cuDNN stability: disable autotuning (avoids CUDNN_STATUS_EXECUTION_FAILED
    #    on newer GPU architectures like Blackwell RTX 5080 Laptop) ──
    import torch
    torch.backends.cudnn.benchmark = False
    # Set environment variable to avoid cuDNN algorithm search failures at high res
    import os
    os.environ["CUDNN_V8_API_ENABLED"] = "1"  # Use cuDNN v8 API path (more stable)

    parser = argparse.ArgumentParser(
        description="Train YOLO for IR ship identification with preprocessing"
    )

    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset YAML (overrides --dataset)")
    parser.add_argument("--dataset", type=str, default="ship",
                        choices=["ship", "nslsr-swir", "nslsr-lwir"],
                        help="Dataset preset")

    # Model
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path or P2 YAML (overrides config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")

    # Preprocessing
    parser.add_argument("--preprocess", action="store_true",
                        help="Enable IR maritime preprocessing pipeline")
    parser.add_argument("--preprocess-method", type=str, default="tophat_clahe",
                        choices=list(AVAILABLE_PREPROCESS.keys()),
                        help="Preprocessing preset (default: tophat_clahe)")
    parser.add_argument("--preprocess-cache", type=str, default=None,
                        help="Output dir for cached preprocessed images "
                             "(default: data/preprocessed_{method}/)")

    # Fusion network (learnable fusion of spatial + frequency domain features)
    parser.add_argument("--use-fusion", action="store_true",
                        help="Insert SpatialFrequencyFusion module before YOLO first conv")
    parser.add_argument("--fusion-hidden", type=int, default=16,
                        help="Hidden dim for fusion network {16, 32, 64} (default: 16)")
    parser.add_argument("--fusion-layers", type=int, default=2, choices=[2, 3, 4],
                        help="Number of Conv blocks before attention {2, 3, 4} (default: 2)")
    parser.add_argument("--fusion-attention", type=str, default="se", choices=["se", "cbam"],
                        help="Attention type: 'se' (SE-Net) or 'cbam' (CBAM) (default: se)")
    parser.add_argument("--fusion-lr-scale", type=float, default=10.0,
                        help="LR multiplier for fusion params (default: 10.0)")
    parser.add_argument("--fusion-residual-target", type=float, default=0.5,
                        help="Target value for residual alpha after warmup {0.3, 0.5, 0.7, 1.0} (default: 0.5)")
    parser.add_argument("--fusion-residual-warmup", type=int, default=10,
                        help="Epochs to warmup residual_alpha from 0→target (default: 10)")

    # Attention module insertion into backbone
    parser.add_argument("--attention-type", type=str, default=None, choices=["eca", "ca", "feca"],
                        help="Insert lightweight attention after each backbone stage: "
                             "'eca' (Efficient Channel Attention, no dim reduction) "
                             "or 'ca' (Coordinate Attention, encodes position info). "
                             "Applied on top of fusion if both enabled.")

    # Neck architecture
    parser.add_argument("--neck-type", type=str, default=None, choices=["c2ema", "asff", "c3k3"],
                        help="Replace Neck C3k2 blocks with: "
                             "'c2ema' (C2f with Efficient Multi-Scale Attention) or "
                             "'asff' (Adaptive Spatial Feature Fusion — replaces Concat "
                             "with learnable weighted fusion). "
                             "Applied on top of all other modifications.")

    # Backbone architecture
    parser.add_argument("--backbone", type=str, default=None, choices=["starnet", "leconv"],
                        help="Replace backbone C3k2 blocks with: "
                             "'starnet' (C3k2Star: StarNet element-wise feature multiplication) "
                             "or 'leconv' (C3k2LeConv: decomposed conv with long-range scaling). "
                             "Applied before neck/head modifications.")

    # Detection head architecture
    parser.add_argument("--head-type", type=str, default=None, choices=["dynamic"],
                        help="Replace Detect head with: "
                             "'dynamic' (Dynamic Head: scale-aware + spatial-aware attention). "
                             "Applied after all other modifications.")

    # Loss function
    parser.add_argument("--iou-loss", type=str, default="ciou",
                        choices=["ciou", "shape-iou", "wiou", "nwd"],
                        help="Bounding box IoU loss type: 'ciou' (default, standard CIoU), "
                             "'shape-iou' (Shape-IoU with shape-aware distance), "
                             "'wiou' (WIoU v3 with distance-aware attention + non-monotonic focusing), "
                             "or 'nwd' (NWD Normalized Wasserstein Distance for tiny objects)")

    # Augmentation jitter
    parser.add_argument("--augment-jitter", type=float, default=0.0,
                        help="Random jitter magnitude for augmentation parameters each epoch "
                             "(0.0=disabled, 0.3=±30% jitter). When >0, augment params like "
                             "scale, hsv_h/s/v, translate, degrees are randomly sampled from "
                             "[base*(1-jitter), base*(1+jitter)] at each epoch start.")

    # Preprocess jitter  (T25)
    parser.add_argument("--preprocess-jitter", type=float, default=0.0,
                        help="Random jitter magnitude for preprocessing parameters per image "
                             "(0.0=disabled, 0.3=±30% jitter). When >0, CLAHE clip_limit / tile_grid, "
                             "tophat kernel, Butterworth cutoffs, unsharp strength, and wavelet "
                             "scales are randomly sampled per image during offline preprocessing.")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--project", type=str, default="ship-detection",
                        help="Project directory name (default: ship-detection)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (alias for --run-name)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (auto-generated if not set; --name is an alias)")

    # Override
    parser.add_argument("--config-override", type=str, default=None,
                        help="JSON string of training config overrides (for sweep)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    # -------------------------------------------------------------------
    # Resolve configuration
    # -------------------------------------------------------------------
    dataset = args.dataset
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS["ship"])

    if args.data:
        data_cfg_path = Path(args.data)
        train_cfg_path = PROJECT_ROOT / "configs" / "train.yaml"
    else:
        data_cfg_path = PROJECT_ROOT / dataset_config["data"]
        train_cfg_path = PROJECT_ROOT / dataset_config["train"]

    if not train_cfg_path.exists():
        # Fallback to original train config
        train_cfg_path = PROJECT_ROOT / "configs" / "train.yaml"

    # Load configs
    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    # Apply config overrides (for sweep)
    if args.config_override:
        try:
            overrides = json.loads(args.config_override)
            cfg.update(overrides)
            logger.info("Applied config overrides: %s", overrides)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Invalid config override JSON: %s", e)

    # Resolve data path
    cfg_path = data_cfg.get("path", "")
    if cfg_path and not Path(cfg_path).exists():
        alt = PROJECT_ROOT / cfg_path
        if alt.exists():
            data_cfg["path"] = str(alt.resolve())
        else:
            logger.warning("Data path not found: %s", cfg_path)

    # CLI overrides
    if args.model:
        cfg["model"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz

    cfg["model_name"] = cfg.get("model", "yolo11m")
    cfg["dataset_name"] = dataset

    # -------------------------------------------------------------------
    # Preprocessing (cache images before training)
    # -------------------------------------------------------------------
    preprocessing_info = None
    if args.preprocess and args.preprocess_method != "none":
        logger.info("=" * 60)
        logger.info("PREPROCESSING STAGE")
        logger.info("Method: %s", args.preprocess_method)
        logger.info("Description: %s", AVAILABLE_PREPROCESS.get(args.preprocess_method, ""))
        if args.preprocess_jitter > 0.0:
            logger.info("Preprocess jitter: enabled (magnitude=%.2f)", args.preprocess_jitter)
            logger.info("Jittered params: %s", ", ".join(SUPPORTED_JITTER_ATTRS))
        logger.info("=" * 60)

        pipeline, desc, params = get_preprocessing_pipeline(args.preprocess_method)
        preprocessing_info = {
            "method": args.preprocess_method,
            "description": desc,
            "params": params,
        }

        # Apply preprocessing jitter if enabled
        if args.preprocess_jitter > 0.0:
            pipeline = _make_preprocess_jitter_wrapper(pipeline, args.preprocess_jitter)
            preprocessing_info["preprocess_jitter"] = args.preprocess_jitter
            if isinstance(params, dict):
                params["jitter"] = f"±{args.preprocess_jitter*100:.0f}% on {', '.join(SUPPORTED_JITTER_ATTRS)}"

        cache_root = args.preprocess_cache or str(
            PROJECT_ROOT / "data" / f"preprocessed_{args.preprocess_method}"
        )
        new_data_cfg = preprocess_and_cache(
            pipeline, data_cfg, cache_root, args.preprocess_method
        )
        data_cfg = new_data_cfg
        actual_cache_root = Path(cache_root)
        preprocessing_info["sample_dir"] = str(actual_cache_root / "samples")
        logger.info("Preprocessing complete. Training will use preprocessed data.")

    preprocess_tag = f"-{args.preprocess_method}" if args.preprocess else ""
    fusion_tag = "-fusion" if args.use_fusion else ""
    run_name = args.name or args.run_name or (
        f"{dataset}-{Path(cfg['model_name']).stem}{preprocess_tag}{fusion_tag}"
    )
    cfg["project"] = args.project
    cfg["name"] = run_name

    # -------------------------------------------------------------------
    # Model initialization
    # -------------------------------------------------------------------
    model_name = cfg.pop("model")
    logger.info("=" * 60)
    logger.info("SHIP IDENTIFICATION YOLO TRAINING")
    logger.info("=" * 60)
    logger.info("Dataset:    %s", dataset)
    logger.info("Data path:  %s", data_cfg.get("path", "?"))
    logger.info("Model:      %s", model_name)
    logger.info("Config:     %s", train_cfg_path)
    logger.info("Epochs:     %d", cfg.get("epochs", 200))
    logger.info("Batch:      %d", cfg.get("batch", 16))
    logger.info("Image size: %d", cfg.get("imgsz", 640))
    logger.info("Preprocess: %s", args.preprocess_method if args.preprocess else "disabled")
    if args.preprocess_jitter > 0.0 and args.preprocess:
        logger.info("Preprocess jitter: %.2f", args.preprocess_jitter)
    logger.info("Fusion:     %s",
                f"SpatialFrequencyFusion(h={args.fusion_hidden}, layers={args.fusion_layers}, "
                f"attn={args.fusion_attention}, residual_target={args.fusion_residual_target}, "
                f"lr_scale={args.fusion_lr_scale}x)" if args.use_fusion else "disabled")
    logger.info("Attention:  %s", f"{args.attention_type.upper()} in backbone" if args.attention_type else "disabled")
    logger.info("Neck:       %s", f"{args.neck_type.upper()}" if args.neck_type else "YOLO default")
    logger.info("Backbone:   %s", f"{args.backbone.upper()}" if args.backbone else "YOLO default")
    logger.info("Head:       %s", f"{args.head_type.upper()}" if args.head_type else "YOLO default")
    logger.info("Wandb:      %s", "enabled" if args.wandb else "disabled")
    logger.info("Device:     %s", args.device or "auto")
    logger.info("=" * 60)

    try:
        model = YOLO(model_name)
        # If model is a YAML architecture (e.g. P2 head), load pretrained
        # base weights to transfer matching layers.
        if model_name.endswith(".yaml") and not args.resume:
            base_pt = Path(model_name).stem.replace("_p2", "") + ".pt"
            if Path(base_pt).exists():
                logger.info("Architecture from YAML: %s", model_name)
                logger.info("Transferring pretrained base: %s", base_pt)
                model.load(str(base_pt))
            else:
                logger.info("Architecture from YAML: %s (no base .pt found, training from scratch)", model_name)
    except Exception as e:
        logger.error("Failed to load model '%s': %s", model_name, e)
        sys.exit(1)

    # ── Input validation ─────────────────────────────────────────────
    if args.use_fusion and args.preprocess_method not in ("awb_dual_fusion", "awb_wavelet_dual_fusion", "awb_rpca_dual_fusion"):
        logger.error(
            "--use-fusion requires a dual-stream preprocessing method "
            "(awb_dual_fusion, awb_wavelet_dual_fusion, or awb_rpca_dual_fusion) "
            "(got '%s'). The fusion module expects the 3-channel "
            "[spatial, freq, original] input that only the dual pipeline produces.",
            args.preprocess_method,
        )
        sys.exit(1)

    # Register custom trainer that injects fusion into the model created
    # during model.train() (Ultralytics internally recreates the model from
    # YAML — we override get_model() to insert fusion at that point)
    fusion_info = None
    if args.use_fusion:
        from ultralytics.models.yolo.detect import DetectionTrainer

        fusion_kwargs = dict(
            in_channels=3,
            hidden=args.fusion_hidden,
            out_channels=3,
            num_layers=args.fusion_layers,
            attention=args.fusion_attention,
        )
        FusionTrainer = make_fusion_trainer_class(
            DetectionTrainer,
            fusion_kwargs,
            fusion_lr_scale=args.fusion_lr_scale,
            residual_warmup_epochs=args.fusion_residual_warmup,
            residual_alpha_target=args.fusion_residual_target,
        )
        model.trainer_class = FusionTrainer

        n_fusion_params = sum(
            p.numel()
            for p in SpatialFrequencyFusion(**fusion_kwargs).parameters()
        )
        logger.info("=" * 60)
        logger.info("FUSION NETWORK")
        logger.info("SpatialFrequencyFusion (hidden=%d, layers=%d, attn=%s, ~%d params)",
                     args.fusion_hidden, args.fusion_layers,
                     args.fusion_attention, n_fusion_params)
        logger.info("Residual α warmup: 0.0 → %.2f over %d epochs",
                     args.fusion_residual_target, args.fusion_residual_warmup)
        logger.info("Custom trainer: %s", FusionTrainer.__name__)
        logger.info("=" * 60)
        fusion_info = {
            "enabled": True,
            "hidden_dim": args.fusion_hidden,
            "num_layers": args.fusion_layers,
            "attention": args.fusion_attention,
            "lr_scale": args.fusion_lr_scale,
            "residual_target": args.fusion_residual_target,
            "n_params": n_fusion_params,
        }

    # ── Attention module insertion into backbone ────────────────────
    if args.attention_type:
        from attention_modules import insert_attention_into_backbone
        n_attn = insert_attention_into_backbone(model, args.attention_type)
        logger.info("=" * 60)
        logger.info("ATTENTION IN BACKBONE")
        logger.info("Type:      %s (%d modules inserted after C3k2 stages)",
                     args.attention_type.upper(), n_attn)
        logger.info("=" * 60)

    # ── Neck architecture replacement ───────────────────────────────
    if args.neck_type == "c2ema":
        from c2ema import replace_neck_with_c2ema
        n_replaced = replace_neck_with_c2ema(model)
        logger.info("=" * 60)
        logger.info("NECK REPLACEMENT")
        logger.info("Type:      C2EMA (%d neck blocks replaced)", n_replaced)
        logger.info("=" * 60)
    elif args.neck_type == "asff":
        from asff import replace_concat_with_asff
        n_replaced = replace_concat_with_asff(model)
        logger.info("=" * 60)
        logger.info("NECK REPLACEMENT")
        logger.info("Type:      ASFF (%d Concat layers replaced)", n_replaced)
        logger.info("=" * 60)
    elif args.neck_type == "c3k3":
        from c3k3 import replace_c3k2_with_c3k
        n_replaced = replace_c3k2_with_c3k(model, kernel_size=3)
        logger.info("=" * 60)
        logger.info("NECK REPLACEMENT")
        logger.info("Type:      C3K3 (%d C3k2 blocks replaced, k=3)", n_replaced)
        logger.info("=" * 60)

    # ── Backbone replacement ─────────────────────────────────────────
    if args.backbone == "starnet":
        from starnet_backbone import replace_backbone_with_starnet
        n_replaced = replace_backbone_with_starnet(model)
        logger.info("=" * 60)
        logger.info("BACKBONE REPLACEMENT")
        logger.info("Type:      StarNet (%d C3k2 blocks → C3k2Star)", n_replaced)
        logger.info("=" * 60)

    if args.backbone == "leconv":
        from leconv_backbone import replace_backbone_with_leconv
        n_replaced = replace_backbone_with_leconv(model)
        logger.info("=" * 60)
        logger.info("BACKBONE REPLACEMENT")
        logger.info("Type:      LeConv (%d C3k2 blocks → C3k2LeConv)", n_replaced)
        logger.info("=" * 60)

    # ── Detection head replacement ──────────────────────────────────
    if args.head_type == "dynamic":
        from dynamic_head import replace_detect_with_dynamic_head
        success = replace_detect_with_dynamic_head(model)
        logger.info("=" * 60)
        logger.info("HEAD REPLACEMENT")
        logger.info("Type:      DynamicHead (scale + spatial attention)")
        logger.info("Result:    %s", "✓ replaced" if success else "✗ failed")
        logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Wandb setup — create callback with preprocessing + fusion info
    # -------------------------------------------------------------------
    wandb_callback = None
    fusion_monitor_callback = None
    if args.wandb:
        wandb_callback = _make_wandb_callback(preprocessing_info, fusion_info)
        model.add_callback("on_train_start", wandb_callback["on_train_start"])
        if args.use_fusion:
            fusion_monitor_callback = _make_fusion_monitor_callback()
            model.add_callback("on_val_epoch_end", fusion_monitor_callback["on_val_epoch_end"])

    # ── Augment jitter callback (randomize augment params each epoch) ──
    if args.augment_jitter > 0.0:
        jitter_callback = _make_augment_jitter_callback(args.augment_jitter)
        model.add_callback("on_epoch_start", jitter_callback["on_epoch_start"])
        logger.info("Augment jitter: enabled (magnitude=%.2f)", args.augment_jitter)

    # ── mAP50-95 Early stopping callback (always active) ───────────────
    # Monitors mAP50-95 on val end; stops when improvement < 1e-3
    # for 3 consecutive epochs. More aggressive than default patience=30.
    early_stop_callbacks = _make_map95_early_stop_callback(
        tolerance=1e-3, patience=3
    )
    model.add_callback("on_val_epoch_end", early_stop_callbacks["on_val_epoch_end"])

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    # Remove keys that ultralytics doesn't expect
    for key in ["model_name", "dataset_name"]:
        cfg.pop(key, None)

    # Resolve data argument: ultralytics expects a file path, not a dict
    if args.preprocess and isinstance(data_cfg, dict):
        data_yaml_path = PROJECT_ROOT / f"data/_preprocessed_data_{args.preprocess_method}.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_cfg, f, default_flow_style=False)
        logger.info("Wrote preprocessed data config to %s", data_yaml_path)
        data_arg = str(data_yaml_path)
    else:
        data_arg = str(data_cfg_path)

    # Filter to known ultralytics args (prevent crashes on unsupported keys)
    known_keys = {
        "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf",
        "cos_lr", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
        "warmup_bias_lr", "box", "cls", "dfl", "iou",
        "mosaic", "close_mosaic", "mixup", "copy_paste", "scale", "fliplr",
        "flipud", "degrees", "translate", "shear", "perspective", "erasing",
        "hsv_h", "hsv_s", "hsv_v", "overlap_mask", "mask_ratio", "dropout",
        "save", "save_period", "exist_ok", "pretrained", "verbose",
        "project", "name", "val", "plots", "deterministic", "nbs",
        # ── added for completeness ──
        "single_cls", "amp", "cache", "fraction", "resume", "device", "workers",
    }
    filtered = {k: v for k, v in cfg.items() if k in known_keys}

    # ── Pass resume flag to model.train() so Ultralytics loads checkpoint
    #    fusion weights via resume_training() ──
    if args.resume:
        filtered["resume"] = True

    # ── Shape-IoU loss: monkey-patch v8DetectionLoss to replace BboxLoss ──
    if args.iou_loss == "shape-iou":
        from ultralytics.utils.loss import v8DetectionLoss
        from custom_loss import ShapeIoUBboxLoss

        _orig_v8_init = v8DetectionLoss.__init__

        def _shape_iou_init(self, model):
            _orig_v8_init(self, model)
            reg_max = getattr(model.model[-1], "reg_max", 16)
            self.bbox_loss = ShapeIoUBboxLoss(reg_max=reg_max)

        v8DetectionLoss.__init__ = _shape_iou_init
        logger.info("Loss: Shape-IoU (replaced BboxLoss → ShapeIoUBboxLoss)")

    # ── WIoU v3 loss: monkey-patch v8DetectionLoss to replace BboxLoss ──
    if args.iou_loss == "wiou":
        from ultralytics.utils.loss import v8DetectionLoss
        from custom_loss import WIoUBboxLoss

        _orig_v8_init = v8DetectionLoss.__init__

        def _wiou_init(self, model):
            _orig_v8_init(self, model)
            reg_max = getattr(model.model[-1], "reg_max", 16)
            self.bbox_loss = WIoUBboxLoss(reg_max=reg_max, monotonous=False)

        v8DetectionLoss.__init__ = _wiou_init
        logger.info("Loss: WIoU v3 (replaced BboxLoss → WIoUBboxLoss, non-monotonic focusing)")

    # ── NWD loss: monkey-patch v8DetectionLoss to replace BboxLoss ──
    if args.iou_loss == "nwd":
        from ultralytics.utils.loss import v8DetectionLoss
        from custom_loss import NWDBboxLoss

        _orig_v8_init = v8DetectionLoss.__init__

        def _nwd_init(self, model):
            _orig_v8_init(self, model)
            reg_max = getattr(model.model[-1], "reg_max", 16)
            self.bbox_loss = NWDBboxLoss(reg_max=reg_max, iou_ratio=0.3, nwd_constant=12.8)

        v8DetectionLoss.__init__ = _nwd_init
        logger.info("Loss: NWD (replaced BboxLoss → NWDBboxLoss, iou_ratio=0.3, nwd_constant=12.8)")

    results = model.train(
        data=data_arg,
        device=args.device,
        **filtered,
    )

    # -------------------------------------------------------------------
    # Wandb cleanup — Ultralytics' on_train_end already called
    # wb.run.finish(), but on Windows the symlink-based latest-run
    # cleanup can fail silently with OSError.  Force a final finish
    # here to catch any leftover file-handle issues before val/export.
    # -------------------------------------------------------------------
    _finalize_wandb_safe()

    # -------------------------------------------------------------------
    # Validation & logging (Ultralytics auto-logs to wandb; we also log
    # our custom metrics in case wandb=True is not set)
    # -------------------------------------------------------------------
    logger.info("Training complete. Running validation...")
    try:
        metrics = model.val()
        if metrics and hasattr(metrics, "box"):
            logger.info("mAP50-95:  %.4f", metrics.box.map)
            logger.info("mAP50:     %.4f", metrics.box.map50)
            logger.info("Precision: %.4f", metrics.box.mp)
            logger.info("Recall:    %.4f", metrics.box.mr)

            log_val_metrics_to_wandb(metrics)
        else:
            logger.info("Validation metrics: %s", metrics)
    except Exception as e:
        logger.warning("Validation failed: %s", e)

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------
    try:
        logger.info("Exporting model to ONNX...")
        success = model.export(format="onnx", imgsz=cfg.get("imgsz", 640))
        if success:
            logger.info("ONNX export successful: %s", success)
    except Exception as e:
        logger.warning("ONNX export skipped: %s", e)

    logger.info("Done! Model weights saved in runs/detect/train/")


if __name__ == "__main__":
    from ultralytics import YOLO
    main()
