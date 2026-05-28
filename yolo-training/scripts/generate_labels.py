"""
Generate YOLO-format pseudo-labels for the synthetic ship dataset.

Each synthetic image has a known ship class (from the directory name)
and a known pixel height range (e.g., "10-13"). The ship is centered
in the 1024×512 image. This script produces .txt label files with
approximate bounding boxes based on these assumptions.

Label format (YOLO):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    All values in [0, 1], normalized by image dimensions.

Aspect ratio estimation:
  - Range 0-29 (broadside-like):    width = height × 4.0
  - Range 30-59 (angled):           width = height × 3.0
  - Range 60-89 (head-on-like):     width = height × 2.0

Usage:
    uv run python scripts/generate_labels.py [--raw-dir PATH] [--output-dir PATH]
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_WIDTH = 1024
IMG_HEIGHT = 512
CX_NORM = 0.5  # ship is centered horizontally
CY_NORM = 0.5  # ship is centered vertically

CLASS_MAP = {
    "Ada": 0,
    "Akizuki": 1,
    "Alvaro De Bazan": 2,
    "Armourique": 3,
    "Independence": 4,
    "Jiangkai II": 5,
    "Oliver Hazard Perry": 6,
    "Sejong Daewang": 7,
    "Zumwalt": 8,
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Aspect ratio multiplier per range prefix
# The range encodes the viewing angle:
#   0-xx = broadside (ship is long)
#   xx-xx = angled (medium)
#   xx-89 = head-on (ship appears short)
ASPECT_MAP = {
    0: 4.0,   # range starts with 0 → broadside
    30: 3.0,  # range starts with 30 → angled
    60: 2.0,  # range starts with 60 → head-on
}


def get_aspect_ratio(range_str: str) -> float:
    """Determine aspect ratio from the range prefix."""
    try:
        prefix = int(range_str.split("-")[0])
    except (ValueError, IndexError):
        logger.warning("Could not parse range '%s', defaulting to 3.0", range_str)
        return 3.0

    for threshold in sorted(ASPECT_MAP.keys(), reverse=True):
        if prefix >= threshold:
            return ASPECT_MAP[threshold]
    return 3.0


def generate_label_line(
    class_name: str,
    height_range: str,
    range_str: str,
) -> str:
    """Generate a single YOLO-format label line.

    Args:
        class_name: Ship class name (e.g., "Sejong Daewang")
        height_range: Pixel height range (e.g., "10-13")
        range_str: Range/angle prefix (e.g., "30-59")

    Returns:
        YOLO label string: "<class_id> <cx> <cy> <w> <h>"
    """
    class_id = CLASS_MAP.get(class_name)
    if class_id is None:
        raise ValueError(f"Unknown class: {class_name}")

    # Pixel height: use midpoint of the range
    try:
        h_min, h_max = map(int, height_range.split("-"))
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid height range: {height_range}")  # noqa: B904

    box_h_px = (h_min + h_max) / 2.0
    aspect = get_aspect_ratio(range_str)
    box_w_px = box_h_px * aspect

    # Normalize
    box_h_norm = box_h_px / IMG_HEIGHT
    box_w_norm = box_w_px / IMG_WIDTH

    # Validate bounds
    assert 0 < box_h_norm <= 1, f"box_h_norm={box_h_norm} out of range"
    assert 0 < box_w_norm <= 1, f"box_w_norm={box_w_norm} out of range"

    return f"{class_id} {CX_NORM:.6f} {CY_NORM:.6f} {box_w_norm:.6f} {box_h_norm:.6f}"


def generate_labels_for_dataset(
    raw_dir: Path,
    output_dir: Path | None = None,
) -> int:
    """Walk raw_dir and generate YOLO labels for all synthetic images.

    Args:
        raw_dir: Path to extracted nor_irship dataset (data/raw/nor_irship)
        output_dir: Where to save .txt files. If None, save alongside images.

    Returns:
        Number of label files generated.
    """
    if output_dir is None:
        output_dir = raw_dir

    png_files = list(raw_dir.rglob("*.png"))
    logger.info("Found %d PNG files in %s", len(png_files), raw_dir)

    count = 0
    errors = 0

    for p in png_files:
        parts = p.relative_to(raw_dir).parts
        if len(parts) < 3:
            errors += 1
            continue

        range_str = parts[0]
        height_str = parts[1]
        class_name = parts[2]

        try:
            label_line = generate_label_line(class_name, height_str, range_str)
        except (ValueError, AssertionError) as e:
            logger.warning("Error generating label for %s: %s", p, e)
            errors += 1
            continue

        # Save .txt alongside image (or in output_dir with same relative path)
        if output_dir == raw_dir:
            label_path = p.with_suffix(".txt")
        else:
            rel_path = p.relative_to(raw_dir)
            label_path = output_dir / rel_path.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)

        label_path.write_text(label_line + "\n")
        count += 1

    logger.info("Generated %d label files (%d errors)", count, errors)
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo-labels for synthetic ship dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nor_irship",
        help="Path to extracted nor_irship dataset (default: data/raw/nor_irship)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .txt labels. Default: same as raw-dir (alongside images)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing labels",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        logger.error("Raw directory not found: %s", args.raw_dir)
        logger.error("Run extract_datasets.py first.")
        return

    logger.info("Using raw dir: %s", args.raw_dir)
    logger.info("Output dir: %s", args.output_dir or args.raw_dir)

    # Quick pre-flight check
    png_count = len(list(args.raw_dir.rglob("*.png")))
    logger.info("Found %d PNG images to process.", png_count)

    if args.dry_run:
        # Just show a few examples
        sample_dirs = sorted(args.raw_dir.iterdir())[:2]
        for d in sample_dirs:
            if d.is_dir():
                sample_class = list(d.rglob("*.png"))[:1]
                for p in sample_class:
                    parts = p.relative_to(args.raw_dir).parts
                    if len(parts) >= 3:
                        line = generate_label_line(parts[2], parts[1], parts[0])
                        print(f"  {p.relative_to(args.raw_dir)} → {line}")
        print(f"\nDry run: would generate {png_count} label files.")
        return

    count = generate_labels_for_dataset(args.raw_dir, args.output_dir)
    logger.info("Done. Generated %d label files.", count)


if __name__ == "__main__":
    main()
