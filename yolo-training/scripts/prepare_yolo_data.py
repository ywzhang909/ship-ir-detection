"""
Build YOLO-format dataset directory structure with train/val/test splits.

Pipeline:
  1. Walks extracted raw data, collects all images with metadata
  2. Generates YOLO labels using generate_labels logic
  3. Performs stratified train/val/test split (80/10/10)
  4. Copies (or hardlinks) images and labels to data/processed/
  5. Optionally includes Sanya IR images in test set (no labels)

Usage:
    uv run python scripts/prepare_yolo_data.py [--no-sanya] [--dry-run]
"""

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scripts.generate_labels import generate_label_line

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_NOR = PROJECT_ROOT / "data" / "raw" / "nor_irship"
RAW_SANYA = PROJECT_ROOT / "data" / "raw" / "sanya_ir"
PROCESSED = PROJECT_ROOT / "data" / "processed"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def collect_nor_images(raw_dir: Path) -> list[dict]:
    """Collect all images from the synthetic dataset with metadata.

    Returns list of dicts with keys: path, class_name, range_str, height_str.
    """
    records = []
    for p in raw_dir.rglob("*.png"):
        parts = p.relative_to(raw_dir).parts
        if len(parts) >= 3:
            records.append({
                "path": p,
                "class_name": parts[2],
                "range_str": parts[0],
                "height_str": parts[1],
            })
    return records


def collect_sanya_images(raw_dir: Path) -> list[Path]:
    """Collect all images from the real IR dataset (no labels)."""
    return list(raw_dir.rglob("*.JPG")) + list(raw_dir.rglob("*.jpg"))


def stratified_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by class name."""
    random.seed(seed)

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class_name"]].append(rec)

    train, val, test = [], [], []
    for cls, cls_records in by_class.items():
        random.shuffle(cls_records)
        n = len(cls_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train.extend(cls_records[:n_train])
        val.extend(cls_records[n_train:n_train + n_val])
        test.extend(cls_records[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def copy_with_hardlink(src: Path, dst: Path) -> None:
    """Copy file using hardlink if possible, fall back to shutil.copy2."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # NTFS hard links
        import os
        os.link(str(src), str(dst))
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO-format dataset")
    parser.add_argument("--no-sanya", action="store_true", help="Skip Sanya IR images in test set")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for split")
    args = parser.parse_args()

    if not RAW_NOR.exists():
        logger.error("nor_irship raw data not found at %s", RAW_NOR)
        logger.error("Run extract_datasets.py first.")
        return

    # ---------------------------------------------------------------
    # Step 1: Collect all images
    # ---------------------------------------------------------------
    logger.info("Collecting synthetic images from %s ...", RAW_NOR)
    nor_records = collect_nor_images(RAW_NOR)
    logger.info("Found %d synthetic images.", len(nor_records))

    if args.dry_run:
        by_class = defaultdict(int)
        for rec in nor_records:
            by_class[rec["class_name"]] += 1
        print("\nClass distribution:")
        for cls, count in sorted(by_class.items()):
            print(f"  {cls}: {count}")
        print(f"\nTotal: {len(nor_records)} images")

        if not args.no_sanya and RAW_SANYA.exists():
            sanya_images = collect_sanya_images(RAW_SANYA)
            print(f"Sanya IR images available: {len(sanya_images)} (will be added to test set)")

        print("\nDry run complete. Use without --dry-run to create the dataset.")
        return

    # ---------------------------------------------------------------
    # Step 2: Stratified split
    # ---------------------------------------------------------------
    logger.info("Creating stratified train/val/test split...")
    train_records, val_records, test_records = stratified_split(nor_records, seed=args.seed)

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_records), len(val_records), len(test_records),
    )

    # ---------------------------------------------------------------
    # Step 3: Copy files to processed directory
    # ---------------------------------------------------------------
    splits = {"train": [], "val": [], "test": []}

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        logger.info("Copying %s set (%d images)...", split_name, len(split_records))
        for rec in tqdm(split_records):
            src_img = rec["path"]
            img_id = src_img.stem
            class_name = rec["class_name"]
            height_str = rec["height_str"]
            range_str = rec["range_str"]

            # Preserve origin metadata in filename for traceability
            dst_name = f"{img_id}.png"
            dst_img = PROCESSED / split_name / "images" / dst_name
            copy_with_hardlink(src_img, dst_img)

            # Generate label
            label_line = generate_label_line(class_name, height_str, range_str)
            dst_label = PROCESSED / split_name / "labels" / f"{img_id}.txt"
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text(label_line + "\n")

            splits[split_name].append({
                "image": str(dst_img.relative_to(PROCESSED)),
                "label": str(dst_label.relative_to(PROCESSED)),
                "class": class_name,
                "range": range_str,
                "height": height_str,
            })

    # ---------------------------------------------------------------
    # Step 4: Optionally add Sanya IR images to test set (no labels)
    # ---------------------------------------------------------------
    if not args.no_sanya and RAW_SANYA.exists():
        sanya_images = collect_sanya_images(RAW_SANYA)
        logger.info("Adding %d Sanya IR images to test set (unlabeled)...", len(sanya_images))

        for src_img in tqdm(sanya_images):
            # Preserve relative path structure to avoid name collisions
            rel = src_img.relative_to(RAW_SANYA)
            dst_name = f"sanya_{rel.parent.stem}_{rel.stem}.jpg"
            dst_img = PROCESSED / "test" / "images" / dst_name
            copy_with_hardlink(src_img, dst_img)

            splits["test"].append({
                "image": str(dst_img.relative_to(PROCESSED)),
                "class": "unknown",
            })

    # ---------------------------------------------------------------
    # Step 5: Save splits metadata
    # ---------------------------------------------------------------
    splits_path = PROJECT_ROOT / "data" / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2))
    logger.info("Saved split metadata to %s", splits_path)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    for split_name in ("train", "val", "test"):
        img_dir = PROCESSED / split_name / "images"
        label_dir = PROCESSED / split_name / "labels"
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_labels = len(list(label_dir.glob("*"))) if label_dir.exists() else 0
        logger.info("  %s: %d images, %d labels", split_name, n_imgs, n_labels)

    logger.info("Data path: %s", PROCESSED)
    logger.info("Run training with: uv run python train.py")


if __name__ == "__main__":
    main()
