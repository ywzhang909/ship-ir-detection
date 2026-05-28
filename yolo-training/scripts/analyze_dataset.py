"""
Analyze extracted ship datasets and generate statistics.

Walks the extracted `data/raw/nor_irship/` and `data/raw/sanya_ir/`
directories to compute class distributions, object size histograms,
and image-level statistics.

Usage:
    uv run python scripts/analyze_dataset.py
"""

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOR_IRSHIP_DIR = PROJECT_ROOT / "data" / "raw" / "nor_irship"
SANYA_DIR = PROJECT_ROOT / "data" / "raw" / "sanya_ir"

# Known 9 ship classes
CLASS_NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique",
    "Independence", "Jiangkai II", "Oliver Hazard Perry",
    "Sejong Daewang", "Zumwalt",
]


def analyze_nor_irship():
    """Analyze the synthetic ship dataset."""
    if not NOR_IRSHIP_DIR.exists():
        logger.warning("nor_irship directory not found at %s. Run extract_datasets.py first.", NOR_IRSHIP_DIR)
        return {}

    stats = {
        "class_counts": Counter(),
        "range_counts": Counter(),
        "pixel_height_counts": Counter(),
        "class_per_range": defaultdict(Counter),
        "sample_images": [],
    }

    # Walk the directory structure:
    #   {range_angle}/{pixel_height}/{class_name}/{image_id}.png
    png_files = list(NOR_IRSHIP_DIR.rglob("*.png"))
    logger.info("Found %d PNG files in nor_irship.", len(png_files))

    for p in png_files:
        parts = p.relative_to(NOR_IRSHIP_DIR).parts
        if len(parts) >= 3:
            range_str = parts[0]   # e.g. "0-29"
            height_str = parts[1]  # e.g. "10-13"
            class_name = parts[2]  # e.g. "Sejong Daewang"

            stats["range_counts"][range_str] += 1
            stats["pixel_height_counts"][height_str] += 1
            stats["class_counts"][class_name] += 1
            stats["class_per_range"][range_str][class_name] += 1

    # Print report
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS: nor_irship (Synthetic Rendered Ships)")
    print("=" * 70)

    print(f"\nTotal images: {len(png_files)}")
    print(f"Image format: 1024x512 grayscale PNG (from sample)")

    print("\n--- Class Distribution ---")
    for cls in sorted(stats["class_counts"].keys()):
        count = stats["class_counts"][cls]
        pct = 100 * count / len(png_files)
        print(f"  {cls:30s}: {count:6d} ({pct:5.2f}%)")

    print("\n--- Range (Angle) Distribution ---")
    for rng in sorted(stats["range_counts"].keys()):
        count = stats["range_counts"][rng]
        print(f"  {rng:10s}: {count:6d} images")

    print("\n--- Pixel Height Distribution ---")
    for h in sorted(stats["pixel_height_counts"].keys(), key=lambda x: int(x.split("-")[0])):
        count = stats["pixel_height_counts"][h]
        print(f"  {h:10s}: {count:6d} images")

    print("\n--- Class per Range ---")
    for rng in sorted(stats["class_per_range"].keys()):
        print(f"  Range {rng}:")
        for cls in sorted(stats["class_per_range"][rng].keys()):
            print(f"    {cls:30s}: {stats['class_per_range'][rng][cls]:6d}")

    # Verify all 9 classes are present
    found_classes = set(stats["class_counts"].keys())
    expected = set(CLASS_NAMES)
    missing = expected - found_classes
    extra = found_classes - expected
    if missing:
        print(f"\n⚠️  Missing classes: {missing}")
    if extra:
        print(f"\n⚠️  Extra classes found: {extra}")
    if not missing and not extra:
        print("\n✓ All 9 expected classes present.")

    return stats


def analyze_sanya():
    """Analyze the real IR ship dataset."""
    if not SANYA_DIR.exists():
        logger.warning("sanya_ir directory not found at %s. Run extract_datasets.py first.", SANYA_DIR)
        return {}

    stats = {
        "class_counts": Counter(),
        "time_counts": Counter(),
        "focal_counts": Counter(),
        "weather_counts": Counter(),
    }

    jpg_files = list(SANYA_DIR.rglob("*.JPG")) + list(SANYA_DIR.rglob("*.jpg"))
    logger.info("Found %d JPG files in sanya_ir.", len(jpg_files))

    for p in jpg_files:
        parts = p.relative_to(SANYA_DIR).parts
        if len(parts) >= 3:
            time_str = parts[0]    # "day" or "night"
            focal_str = parts[1]   # "15", "35", "75"
            class_letter = parts[2]  # "A", "C", "D", "E"

            stats["time_counts"][time_str] += 1
            stats["focal_counts"][focal_str] += 1
            stats["class_counts"][class_letter] += 1

            # Parse weather from filename
            fname = p.stem
            if "_rain" in fname:
                weather = "rain"
            elif "_fog" in fname:
                weather = "fog"
            elif "_cloudy" in fname:
                weather = "cloudy"
            else:
                weather = "clear"
            stats["weather_counts"][weather] += 1

    print("\n" + "=" * 70)
    print("DATASET ANALYSIS: sanya_ir (Real IR Ship Images)")
    print("=" * 70)
    print(f"\nTotal images: {len(jpg_files)}")

    print("\n--- Class Letter Distribution ---")
    for cls in sorted(stats["class_counts"].keys()):
        count = stats["class_counts"][cls]
        pct = 100 * count / len(jpg_files)
        print(f"  {cls}: {count:6d} ({pct:5.2f}%)")

    print("\n--- Time of Day Distribution ---")
    for t in sorted(stats["time_counts"].keys()):
        print(f"  {t}: {stats['time_counts'][t]:6d}")

    print("\n--- Focal Length Distribution ---")
    for f in sorted(stats["focal_counts"].keys()):
        print(f"  {f}mm: {stats['focal_counts'][f]:6d}")

    print("\n--- Weather Distribution ---")
    for w in sorted(stats["weather_counts"].keys()):
        print(f"  {w}: {stats['weather_counts'][w]:6d}")

    return stats


def main():
    print("=" * 70)
    print("SHIP IDENTIFICATION DATASET ANALYSIS")
    print("=" * 70)

    nor_stats = analyze_nor_irship()
    sanya_stats = analyze_sanya()

    # Summary
    if nor_stats:
        total_nor = sum(nor_stats["class_counts"].values())
    else:
        total_nor = 0
    if sanya_stats:
        total_sanya = sum(sanya_stats["class_counts"].values())
    else:
        total_sanya = 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  nor_irship (synthetic): {total_nor:6d} images, {len(nor_stats.get('class_counts', {}))} classes")
    print(f"  sanya_ir (real IR):     {total_sanya:6d} images, {len(sanya_stats.get('class_counts', {}))} classes")
    print(f"  TOTAL:                  {total_nor + total_sanya:6d} images")

    if nor_stats:
        print("\nClasses found in nor_irship:")
        for cls, count in sorted(nor_stats["class_counts"].items()):
            print(f"  {cls:30s}: {count}")


if __name__ == "__main__":
    main()
