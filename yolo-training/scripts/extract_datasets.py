"""
Extract both ship datasets from compressed archives.

Dataset 1: nor_irship_with_background_opensource (RAR, 5.1 GB)
  - 60,714 synthetic ship images, 1024x512 grayscale PNG
  - 9 ship classes, organized by range_angle / pixel_height / class_name

Dataset 2: sanya_ir_nodel_opensource_weather (tar.xz, 647 MB)
  - ~3,500 real IR ship images, JPG
  - 4 ship letter-classes, organized by day|night / focal_length / class_letter

Usage:
    uv run python scripts/extract_datasets.py [--dataset all|nor_irship|sanya] [--force]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source archives (relative to the Ship dataset repo, which is a sibling of this project)
SHIP_DATASET_DIR = PROJECT_ROOT.parent / "Ship"

ARCHIVES = {
    "nor_irship": {
        "src": SHIP_DATASET_DIR / "nor_irship_with_background_opensource" / "large_001.rar",
        "dst": PROJECT_ROOT / "data" / "raw" / "nor_irship",
        "expected_count": 60714,
    },
    "sanya": {
        "src": SHIP_DATASET_DIR / "sanya_ir_nodel_opensource_weather.tar.xz",
        "dst": PROJECT_ROOT / "data" / "raw" / "sanya_ir",
        "expected_count": None,  # unknown exact count
    },
}


def find_7z() -> Path:
    """Locate 7-Zip executable."""
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "7-Zip" / "7z.exe",
        Path(os.environ.get("PROGRAMW6432", "C:\\Program Files")) / "7-Zip" / "7z.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")) / "7-Zip" / "7z.exe",
    ]
    # Also check PATH
    for p in os.environ.get("PATH", "").split(";"):
        candidate = Path(p) / "7z.exe"
        if candidate.exists():
            return candidate

    for c in candidates:
        if c.exists():
            return c

    # Check scoop shim
    scoop_shim = Path.home() / "scoop" / "shims" / "7z.exe"
    if scoop_shim.exists():
        return scoop_shim

    raise FileNotFoundError(
        "7-Zip (7z.exe) not found. Install it from https://7-zip.org/ or via 'winget install 7zip.7zip'"
    )


def extract_nor_irship(seven_zip: Path, force: bool = False) -> None:
    """Extract the large RAR archive containing synthetic ship images."""
    cfg = ARCHIVES["nor_irship"]
    dst = cfg["dst"]

    if dst.exists() and not force:
        # Quick sanity: count extracted files
        existing = sum(1 for _ in dst.rglob("*.png"))
        if existing >= cfg["expected_count"]:
            logger.info(
                "nor_irship already extracted at %s (%d files found, expected %d). "
                "Use --force to re-extract.",
                dst, existing, cfg["expected_count"],
            )
            return
        logger.warning(
            "nor_irship directory exists but only %d/%d files found. Re-extracting...",
            existing, cfg["expected_count"],
        )

    src = cfg["src"]
    if not src.exists():
        logger.error("Source archive not found: %s", src)
        logger.error(
            "Make sure the Ship dataset repo is at: %s", SHIP_DATASET_DIR
        )
        sys.exit(1)

    logger.info("Extracting %s to %s ...", src.name, dst)
    logger.info("This may take 10-30 minutes depending on disk speed.")

    dst.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(seven_zip), "x", str(src), f"-o{str(dst)}", "-y"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error("7-Zip extraction failed with code %d", result.returncode)
        logger.error("stdout: %s", result.stdout[-500:])
        logger.error("stderr: %s", result.stderr[-500:])
        logger.error(
            "If 7-Zip cannot handle RAR5 format, try installing WinRAR or use:\n"
            "  winget install RARLab.WinRAR"
        )
        sys.exit(1)

    # Count results
    extracted = sum(1 for _ in dst.rglob("*.png"))
    logger.info("Extraction complete: %d PNG files extracted.", extracted)

    if extracted != cfg["expected_count"]:
        logger.warning(
            "Expected %d files but found %d. Some files may be missing.",
            cfg["expected_count"], extracted,
        )


def extract_sanya(seven_zip: Path, force: bool = False) -> None:
    """Extract the tar.xz archive containing real IR ship images."""
    cfg = ARCHIVES["sanya"]
    dst = cfg["dst"]

    if dst.exists() and not force:
        existing = sum(1 for _ in dst.rglob("*.JPG"))
        if existing > 0:
            logger.info(
                "sanya_ir already extracted at %s (%d JPG files found). Use --force to re-extract.",
                dst, existing,
            )
            return

    src = cfg["src"]
    if not src.exists():
        logger.error("Source archive not found: %s", src)
        sys.exit(1)

    logger.info("Extracting %s to %s ...", src.name, dst)

    # Step 1: .tar.xz → .tar (7-Zip handles this in one step when we extract to a temp dir)
    tmp_dir = PROJECT_ROOT / "data" / "raw" / "_tmp_sanya"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        # 7-Zip can extract tar.xz directly
        result = subprocess.run(
            [str(seven_zip), "x", str(src), f"-o{str(tmp_dir)}", "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("7-Zip extraction failed for tar.xz: code %d", result.returncode)
            logger.error("stderr: %s", result.stderr[-500:])
            sys.exit(1)

        # Find the extracted directory (should be sanya_ir_nodel_opensource_weather/)
        extracted_dirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
        if not extracted_dirs:
            logger.error("No directory found after extraction. Contents: %s", list(tmp_dir.iterdir()))
            sys.exit(1)

        src_dir = extracted_dirs[0]
        logger.info("Extracted to %s, moving contents to %s ...", src_dir, dst)

        # Move to final destination
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src_dir), str(dst))

        extracted = sum(1 for _ in dst.rglob("*.JPG")) + sum(1 for _ in dst.rglob("*.jpg"))
        logger.info("Extraction complete: %d JPG files.", extracted)

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract ship identification datasets")
    parser.add_argument(
        "--dataset", choices=["all", "nor_irship", "sanya"], default="all",
        help="Which dataset(s) to extract (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-extraction even if target exists",
    )
    args = parser.parse_args()

    seven_zip = find_7z()
    logger.info("Using 7-Zip at: %s", seven_zip)

    if args.dataset in ("all", "nor_irship"):
        extract_nor_irship(seven_zip, force=args.force)

    if args.dataset in ("all", "sanya"):
        extract_sanya(seven_zip, force=args.force)

    logger.info("All extractions complete.")


if __name__ == "__main__":
    main()
