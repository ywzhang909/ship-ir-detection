"""
CRRP — Copy-Rotate-Resize-Paste Data Augmentation for IR Ship Detection.

Generates augmented training images by:
1. Extracting ship instances from existing training images
2. Applying random rotations and resizing
3. Pasting onto background images (or other training images)
4. Updating labels accordingly

This is OFFLINE augmentation — it creates new files on disk,
complementing YOLO's online augmentation.

Usage:
    # Basic: 3 augmented copies per image
    uv run python scripts/crrp_augment.py --input-dir data/nslsr-lwir --output-dir data/crrp_augmented

    # Aggressive: 5 copies, 3 ships per paste
    uv run python scripts/crrp_augment.py --input-dir data/nslsr-lwir --output-dir data/crrp_augmented --copies 5 --max-instances 3

    # Mixed: 2 copies + control set ratio
    uv run python scripts/crrp_augment.py --input-dir data/nslsr-lwir --output-dir data/crrp_augmented --copies 2 --control-ratio 0.3
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Ship aspect ratios from pseudo-label generation rules
# broadside (0-29°): w/h ≈ 4.0, angled (30-59°): w/h ≈ 3.0, head-on (60-89°): w/h ≈ 2.0
SHIP_ASPECT_RATIOS = [4.0, 3.0, 2.0]


def parse_yolo_label(label_path: Path) -> list[list[float]]:
    """Parse a YOLO-format label file.

    Args:
        label_path: Path to .txt file with lines: class_id cx cy w h (normalized).

    Returns:
        List of [class_id, cx, cy, w, h] entries.
    """
    labels = []
    if not label_path.exists():
        return labels
    text = label_path.read_text().strip()
    if not text:
        return labels
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            labels.append([float(p) for p in parts[:5]])
    return labels


def write_yolo_label(label_path: Path, labels: list[list[float]]) -> None:
    """Write YOLO-format labels to file."""
    lines = []
    for lbl in labels:
        lines.append(f"{int(lbl[0]):d} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def denormalize_bbox(bbox_norm: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalized YOLO bbox [cx, cy, w, h] to pixel coords [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox_norm[1:]
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def normalize_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> list[float]:
    """Convert pixel bbox to normalized YOLO format [cx, cy, w, h]."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return [cx, cy, w, h]


def extract_ship_patches(
    image: np.ndarray,
    labels: list[list[float]],
    min_size: int = 8,
    max_size: int = 60,
) -> list[dict]:
    """Extract ship instance patches from an image.

    Args:
        image: BGR image (H, W, 3).
        labels: YOLO labels for this image.
        min_size: Minimum patch dimension (px) to extract.
        max_size: Maximum patch dimension (px) to extract.

    Returns:
        List of dicts with 'patch' (np.ndarray), 'class_id', 'width', 'height', 'aspect_ratio'.
    """
    img_h, img_w = image.shape[:2]
    patches = []
    for label in labels:
        x1, y1, x2, y2 = denormalize_bbox(label, img_w, img_h)
        pw, ph = x2 - x1, y2 - y1
        if pw < min_size or ph < min_size or pw > max_size or ph > max_size:
            continue
        patch = image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        if patch.size == 0:
            continue
        patches.append({
            "patch": patch,
            "class_id": int(label[0]),
            "width": pw,
            "height": ph,
            "aspect_ratio": pw / max(ph, 1),
        })
    return patches


def create_ship_library(
    image_dir: Path,
    label_dir: Path,
    max_samples: int = 1000,
) -> list[dict]:
    """Build a library of ship patches from the training dataset.

    Args:
        image_dir: Directory containing training images.
        label_dir: Directory containing training labels.
        max_samples: Maximum patches to collect (to control RAM).

    Returns:
        List of ship patch dicts.
    """
    image_paths = sorted(image_dir.glob("*"))[:max_samples * 3]  # ample source
    library = []
    for img_path in tqdm(image_paths, desc="Building ship library"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        label_path = label_dir / f"{img_path.stem}.txt"
        labels = parse_yolo_label(label_path)
        if not labels:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        patches = extract_ship_patches(image, labels)
        library.extend(patches)
        if len(library) >= max_samples:
            break

    logger.info("Ship library built: %d patches from %d images", len(library), len(image_paths))
    return library


def paste_ship_instance(
    bg_image: np.ndarray,
    patch: np.ndarray,
    target_bbox_size: tuple[int, int],
    angle: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rotate, resize, and paste a ship patch onto a background image.

    Args:
        bg_image: Background image (H, W, 3).
        patch: Ship patch to paste.
        target_bbox_size: Desired (w, h) in pixels.
        angle: Rotation angle in degrees.

    Returns:
        (modified_bg_image, bbox_pixel) where bbox_pixel = (x1, y1, x2, y2).
    """
    target_w, target_h = target_bbox_size

    # Resize patch
    patch_resized = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Rotate
    center = (target_w // 2, target_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    patch_rotated = cv2.warpAffine(
        patch_resized, M, (target_w, target_h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Random pasting position (ensure within image bounds)
    bg_h, bg_w = bg_image.shape[:2]
    max_x = bg_w - target_w
    max_y = bg_h - target_h
    if max_x <= 0 or max_y <= 0:
        return bg_image, (0, 0, 0, 0)

    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)
    x1, y1 = paste_x, paste_y
    x2, y2 = paste_x + target_w, paste_y + target_h

    # Paste with alpha blending at edges for realism
    # Simple paste for now — can be improved with Poisson blending
    roi = bg_image[y1:y2, x1:x2]
    mask = np.ones_like(patch_rotated, dtype=np.float32)
    blended = (patch_rotated.astype(np.float32) * 0.85 +
               roi.astype(np.float32) * 0.15)
    bg_image[y1:y2, x1:x2] = blended.astype(np.uint8)

    return bg_image, (x1, y1, x2, y2)


def generate_crrp_dataset(
    input_dir: Path,
    output_dir: Path,
    copies: int = 3,
    max_instances: int = 2,
    min_instance_size: tuple[int, int] = (10, 5),
    max_instance_size: tuple[int, int] = (50, 20),
    control_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Generate CRRP-augmented dataset.

    For each training image, creates ``copies`` augmented versions
    with ship instances copied, rotated, resized, and pasted from a
    pre-built library.

    The output dataset includes:
    - All original images (as control set)
    - ``copies`` augmented versions per image

    Args:
        input_dir: Input dataset directory (must have train/{images,labels}/).
        output_dir: Output dataset directory.
        copies: Number of augmented copies per image.
        max_instances: Maximum ships to paste per augmented image.
        min_instance_size: Min (w, h) of pasted instances in pixels.
        max_instance_size: Max (w, h) of pasted instances in pixels.
        control_ratio: Fraction of original images to keep as control set
            (0.0 = no originals, 1.0 = keep all originals).
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_img_dir = input_dir / "train" / "images"
    train_label_dir = input_dir / "train" / "labels"
    val_img_dir = input_dir / "val" / "images"
    val_label_dir = input_dir / "val" / "labels"
    test_img_dir = input_dir / "test" / "images"
    test_label_dir = input_dir / "test" / "labels"

    if not train_img_dir.exists():
        logger.error("Training images not found at %s", train_img_dir)
        return

    # ── Build ship library from training set ──
    logger.info("=" * 60)
    logger.info("CRRP DATA AUGMENTATION")
    logger.info("Input:    %s", input_dir)
    logger.info("Output:   %s", output_dir)
    logger.info("Copies:   %d per image", copies)
    logger.info("Ships:    max %d per augmented image", max_instances)
    logger.info("Size:     %d-%d px wide, %d-%d px tall",
                min_instance_size[0], max_instance_size[0],
                min_instance_size[1], max_instance_size[1])
    logger.info("=" * 60)

    library = create_ship_library(train_img_dir, train_label_dir)
    if not library:
        logger.error("No ship instances found in training set!")
        return
    logger.info("Ship library: %d instances available", len(library))

    # ── Create output structure ──
    out_train_img = output_dir / "train" / "images"
    out_train_label = output_dir / "train" / "labels"
    out_val_img = output_dir / "val" / "images"
    out_val_label = output_dir / "val" / "labels"
    out_test_img = output_dir / "test" / "images"
    out_test_label = output_dir / "test" / "labels"

    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_label.mkdir(parents=True, exist_ok=True)
    out_val_img.mkdir(parents=True, exist_ok=True)
    out_val_label.mkdir(parents=True, exist_ok=True)
    out_test_img.mkdir(parents=True, exist_ok=True)
    out_test_label.mkdir(parents=True, exist_ok=True)

    # ── Copy validation and test sets verbatim ──
    logger.info("Copying validation set...")
    for img_path in tqdm(list(val_img_dir.glob("*")), desc="  val"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        shutil.copy2(img_path, out_val_img / img_path.name)
        lbl_src = val_label_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out_val_label / lbl_src.name)

    logger.info("Copying test set...")
    for img_path in tqdm(list(test_img_dir.glob("*")), desc="  test"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        shutil.copy2(img_path, out_test_img / img_path.name)
        lbl_src = test_label_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, out_test_label / lbl_src.name)

    # ── Process training images ──
    train_images = sorted(train_img_dir.glob("*"))
    logger.info("Generating CRRP-augmented training set (%d source images)...",
                len(train_images))

    for img_idx, img_path in enumerate(tqdm(train_images, desc="  train")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue

        label_path = train_label_dir / f"{img_path.stem}.txt"
        orig_labels = parse_yolo_label(label_path)

        # ── Optionally copy original as control set ──
        if random.random() < control_ratio:
            shutil.copy2(img_path, out_train_img / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, out_train_label / label_path.name)

        # ── Generate augmented copies ──
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        for copy_idx in range(copies):
            aug_img = image.copy()
            new_labels = list(orig_labels)  # keep original instances
            img_h, img_w = aug_img.shape[:2]

            n_instances = random.randint(1, max_instances)
            selected = random.choices(library, k=n_instances)

            for ship in selected:
                target_w = random.randint(min_instance_size[0], max_instance_size[0])
                target_h = random.randint(min_instance_size[1], max_instance_size[1])
                angle = random.uniform(-30, 30)

                aug_img, (x1, y1, x2, y2) = paste_ship_instance(
                    aug_img, ship["patch"], (target_w, target_h), angle,
                )
                if x1 == x2 or y1 == y2:
                    continue

                # Add pasted ship to labels
                norm_bbox = normalize_bbox(x1, y1, x2, y2, img_w, img_h)
                new_labels.append([float(ship["class_id"]), *norm_bbox])

            # Save augmented image and labels
            stem = f"{img_path.stem}_crrp_{copy_idx:02d}"
            cv2.imwrite(str(out_train_img / f"{stem}.jpg"), aug_img)
            write_yolo_label(out_train_label / f"{stem}.txt", new_labels)

    # ── Log statistics ──
    n_orig = len(list(out_train_img.glob("*.[jJpPbBtT]*")))
    n_aug = n_orig - int(len(train_images) * control_ratio)
    logger.info("=" * 60)
    logger.info("CRRP COMPLETE")
    logger.info("Output:      %s", output_dir)
    logger.info("Train orig:  %d images", int(len(train_images) * control_ratio))
    logger.info("Train aug:   %d images", len(train_images) * copies)
    logger.info("Total train: %d images", n_orig)
    logger.info("Val:         %d images", len(list(out_val_img.glob("*"))))
    logger.info("Test:        %d images", len(list(out_test_img.glob("*"))))
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CRRP offline data augmentation for IR ship detection",
    )
    parser.add_argument("--input-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "nslsr-lwir"),
                        help="Input dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "crrp_augmented"),
                        help="Output dataset directory")
    parser.add_argument("--copies", type=int, default=3,
                        help="Number of augmented copies per image (default: 3)")
    parser.add_argument("--max-instances", type=int, default=2,
                        help="Max ship instances to paste per augmented image (default: 2)")
    parser.add_argument("--min-w", type=int, default=10,
                        help="Minimum pasted ship width in pixels (default: 10)")
    parser.add_argument("--max-w", type=int, default=50,
                        help="Maximum pasted ship width in pixels (default: 50)")
    parser.add_argument("--min-h", type=int, default=5,
                        help="Minimum pasted ship height in pixels (default: 5)")
    parser.add_argument("--max-h", type=int, default=20,
                        help="Maximum pasted ship height in pixels (default: 20)")
    parser.add_argument("--control-ratio", type=float, default=0.2,
                        help="Fraction of original images to keep as control (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    generate_crrp_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        copies=args.copies,
        max_instances=args.max_instances,
        min_instance_size=(args.min_w, args.min_h),
        max_instance_size=(args.max_w, args.max_h),
        control_ratio=args.control_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
