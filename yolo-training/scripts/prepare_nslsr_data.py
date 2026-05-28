"""
Build YOLO-format dataset directory from raw NSLSR data with properly disjoint
train/val/test splits.

The NSLSR dataset has a pre-split structure but the test set is a SUBSET of the
validation set. This script fixes the overlap by removing test images from val.

Pipeline:
  1. Scans train/val/test label files to build filename sets
  2. Computes clean val = original_val - test (removes 118 overlapping images)
  3. Hardlinks images and labels to data/processed_nslsr/{swir,lwir}/
  4. Generates data YAML configs for each modality

Usage:
    uv run python scripts/prepare_nslsr_data.py
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_NSLSR = PROJECT_ROOT / "datasets" / "NSLSR" / "data"
PROCESSED = PROJECT_ROOT / "yolo-training" / "data" / "processed_nslsr"
CONFIGS_DIR = PROJECT_ROOT / "yolo-training" / "configs"

SPLITS = ("train", "val", "test")
MODALITIES = ("SWIR", "LWIR")

# Source directory names per split
SRC_DIRS = {
    "train": "NSLSR_train",
    "val": "NSLSR_val",
    "test": "NSLSR_test",
}


def get_stems(label_dir: Path) -> set[str]:
    """Return set of filename stems (without extension) from a label directory."""
    return {p.stem for p in label_dir.glob("*.txt")}


def hardlink(src: Path, dst: Path) -> None:
    """Hardlink *src* to *dst*, creating parent directories as needed.

    Silently skips if *dst* already exists (idempotent).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
    except FileExistsError:
        pass  # already linked — idempotent
    except OSError as e:
        print(f"  [WARN] Failed to hardlink {src.name}: {e}", file=sys.stderr)


def main():
    # ---------------------------------------------------------------
    # Step 1: Scan label files from each source split
    # ---------------------------------------------------------------
    print("=" * 60)
    print("NSLSR DATA PREPARATION")
    print("=" * 60)

    split_stems: dict[str, set[str]] = {}
    for split in SPLITS:
        label_dir = RAW_NSLSR / SRC_DIRS[split] / "labels"
        if not label_dir.exists():
            print(f"[ERROR] Label directory not found: {label_dir}", file=sys.stderr)
            sys.exit(1)
        stems = get_stems(label_dir)
        split_stems[split] = stems
        print(f"  {SRC_DIRS[split]}: {len(stems)} label files")

    # ---------------------------------------------------------------
    # Step 2: Fix val/test overlap
    # ---------------------------------------------------------------
    test_stems = split_stems["test"]
    val_stems = split_stems["val"]
    overlap = test_stems & val_stems
    clean_val_stems = val_stems - test_stems

    print(f"\n  Overlap (test ⊆ val): {len(overlap)} images will be removed from val")
    print(f"  Clean val (val \\ test): {len(clean_val_stems)} images")

    if overlap != test_stems:
        missing_in_val = test_stems - val_stems
        print(
            f"  [WARN] {len(missing_in_val)} test images not found in val:",
            file=sys.stderr,
        )
        for s in sorted(missing_in_val):
            print(f"    {s}.txt", file=sys.stderr)

    # Build destination stems per split
    dest_stems = {
        "train": split_stems["train"],
        "val": clean_val_stems,
        "test": test_stems,
    }

    # ---------------------------------------------------------------
    # Step 3: Hardlink images and labels
    # ---------------------------------------------------------------
    for modality in MODALITIES:
        modality_lower = modality.lower()
        print(f"\n  Processing {modality} ...")

        for split in SPLITS:
            stems = dest_stems[split]
            img_out = PROCESSED / modality_lower / split / "images"
            lbl_out = PROCESSED / modality_lower / split / "labels"

            src_img_dir = RAW_NSLSR / SRC_DIRS[split] / modality
            src_lbl_dir = RAW_NSLSR / SRC_DIRS[split] / "labels"

            linked = 0
            missing_img = 0
            missing_lbl = 0

            for stem in sorted(stems):
                src_img = src_img_dir / f"{stem}.jpg"
                src_lbl = src_lbl_dir / f"{stem}.txt"
                dst_img = img_out / f"{stem}.jpg"
                dst_lbl = lbl_out / f"{stem}.txt"

                # Verify both exist before linking either (all-or-nothing)
                if not src_img.exists():
                    print(f"  [WARN] Missing image: {src_img}", file=sys.stderr)
                    missing_img += 1
                    continue
                if not src_lbl.exists():
                    print(f"  [WARN] Missing label: {src_lbl}", file=sys.stderr)
                    missing_lbl += 1
                    continue

                hardlink(src_img, dst_img)
                hardlink(src_lbl, dst_lbl)
                linked += 1

            print(
                f"    {split}: {linked} linked"
                + (f", {missing_img} missing images" if missing_img else "")
                + (f", {missing_lbl} missing labels" if missing_lbl else "")
            )

    # ---------------------------------------------------------------
    # Step 4: Generate data YAML files
    # ---------------------------------------------------------------
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    for modality in MODALITIES:
        modality_lower = modality.lower()
        yaml_path = CONFIGS_DIR / f"data_nslsr_{modality_lower}.yaml"
        path_str = str((PROCESSED / modality_lower).resolve()).replace("\\", "/")

        yaml_content = (
            f"# YOLO Dataset Configuration \u2014 NSLSR {modality}\n"
            f"# Generated by scripts/prepare_nslsr_data.py\n"
            f"\n"
            f"# Dataset root path\n"
            f"path: {path_str}\n"
            f"\n"
            f"# Train/val/test image directories (relative to path)\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"\n"
            f"# Number of classes\n"
            f"nc: 1\n"
            f"\n"
            f"# Class names (index 0)\n"
            f"names:\n"
            f"  0: ship\n"
        )
        yaml_path.write_text(yaml_content)
        print(f"\n  Saved config: {yaml_path}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    for modality in MODALITIES:
        modality_lower = modality.lower()
        print(f"\n  {modality}:")
        for split in SPLITS:
            img_dir = PROCESSED / modality_lower / split / "images"
            lbl_dir = PROCESSED / modality_lower / split / "labels"
            n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
            n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
            print(f"    {split}: {n_imgs} images, {n_lbls} labels")

    print(f"\n  Overlap removed from val: {len(overlap)} images")
    print(f"\n  Data path:   {PROCESSED}")
    print(f"  Configs:     {CONFIGS_DIR}")
    print(f"  Usage:       uv run python train.py --data configs/data_nslsr_swir.yaml")
    print()


if __name__ == "__main__":
    main()
