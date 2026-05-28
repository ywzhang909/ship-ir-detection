"""
Tests for YOLO dataset integrity.

Verifies that the prepared dataset (data/processed/) is valid:
  - Image-label pairing
  - Split disjointness
  - Split ratio
  - Label format correctness

Usage:
    uv run pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not PROCESSED_DIR.exists(),
    reason=f"Processed data directory not found at {PROCESSED_DIR}. Run prepare_yolo_data.py first.",
)

SPLITS = ["train", "val", "test"]
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG"}
TOLERANCE = 0.03  # ±3% tolerance for split ratios


def get_image_paths(split: str) -> list[Path]:
    """Return all image paths for a given split."""
    img_dir = PROCESSED_DIR / split / "images"
    if not img_dir.exists():
        return []
    return [p for p in img_dir.iterdir() if p.suffix in ALLOWED_IMAGE_EXTS]


def get_label_paths(split: str) -> list[Path]:
    """Return all label paths for a given split."""
    label_dir = PROCESSED_DIR / split / "labels"
    if not label_dir.exists():
        return []
    return list(label_dir.glob("*.txt"))


# ---------------------------------------------------------------------------
# Tests: Image-label pairing
# ---------------------------------------------------------------------------

class TestImageLabelPairing:
    """Every image must have a corresponding label file and vice versa."""

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_every_image_has_label(self, split):
        """Each image in train/val must have a matching .txt label."""
        images = get_image_paths(split)
        labels = get_label_paths(split)

        label_names = {p.stem for p in labels}
        missing = []
        for img in images:
            if img.stem not in label_names:
                missing.append(img.name)

        assert not missing, (
            f"{split}: {len(missing)} images missing labels: {missing[:10]}"
        )

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_every_label_has_image(self, split):
        """Each .txt label must have a matching image."""
        images = get_image_paths(split)
        labels = get_label_paths(split)

        image_names = {p.stem for p in images}
        missing = []
        for lbl in labels:
            if lbl.stem not in image_names:
                missing.append(lbl.name)

        assert not missing, (
            f"{split}: {len(missing)} labels missing images: {missing[:10]}"
        )

    def test_test_set_may_have_unlabeled_images(self):
        """Test set may contain unlabeled images (e.g., Sanya IR)."""
        images = get_image_paths("test")
        labels = get_label_paths("test")
        # This should not raise - test set can have unlabeled images
        assert len(images) >= len(labels), (
            f"test: {len(images)} images but {len(labels)} labels "
            "(labels should not exceed images)"
        )


# ---------------------------------------------------------------------------
# Tests: Split disjointness
# ---------------------------------------------------------------------------

class TestSplitDisjointness:
    """No image should appear in more than one split."""

    def test_no_overlap_train_val(self):
        """Train and val must be disjoint."""
        train_imgs = {p.name for p in get_image_paths("train")}
        val_imgs = {p.name for p in get_image_paths("val")}
        overlap = train_imgs & val_imgs
        assert not overlap, f"{len(overlap)} images appear in both train and val"

    def test_no_overlap_train_test(self):
        """Train and test must be disjoint."""
        train_imgs = {p.name for p in get_image_paths("train")}
        test_imgs = {p.name for p in get_image_paths("test")}
        overlap = train_imgs & test_imgs
        assert not overlap, f"{len(overlap)} images appear in both train and test"

    def test_no_overlap_val_test(self):
        """Val and test must be disjoint."""
        val_imgs = {p.name for p in get_image_paths("val")}
        test_imgs = {p.name for p in get_image_paths("test")}
        overlap = val_imgs & test_imgs
        assert not overlap, f"{len(overlap)} images appear in both val and test"


# ---------------------------------------------------------------------------
# Tests: Split ratio
# ---------------------------------------------------------------------------

class TestSplitRatio:
    """Split ratios should be approximately 80/10/10."""

    def test_split_ratios(self):
        """Train/val/test ratios must be within ±3% of 80/10/10."""
        n_train = len(get_image_paths("train"))
        n_val = len(get_image_paths("val"))
        n_test = len(get_image_paths("test"))
        total = n_train + n_val + n_test

        if total == 0:
            pytest.skip("No images found in any split")

        actual_train = n_train / total
        actual_val = n_val / total
        actual_test = n_test / total

        errors = []
        if abs(actual_train - 0.80) > TOLERANCE:
            errors.append(f"train: expected ~0.80, got {actual_train:.4f}")
        if abs(actual_val - 0.10) > TOLERANCE:
            errors.append(f"val: expected ~0.10, got {actual_val:.4f}")
        if abs(actual_test - 0.10) > TOLERANCE:
            errors.append(f"test: expected ~0.10, got {actual_test:.4f}")

        assert not errors, f"Split ratio deviation: {'; '.join(errors)} (n={total})"


# ---------------------------------------------------------------------------
# Tests: Label format correctness
# ---------------------------------------------------------------------------

class TestLabelFormat:
    """Each label file must contain valid YOLO-format annotations."""

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_label_format(self, split):
        """Each label line must be: class_id cx cy w h, all in [0,1]."""
        labels = get_label_paths(split)
        errors = []

        for lbl in labels:
            lines = lbl.read_text().strip().splitlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"{lbl.name}:{i+1}: expected 5 values, got {len(parts)}")
                    continue

                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    errors.append(f"{lbl.name}:{i+1}: non-numeric values")
                    continue

                cls_id = int(vals[0])
                cx, cy, w, h = vals[1:]

                if not (0 <= cls_id <= 8):
                    errors.append(f"{lbl.name}:{i+1}: class_id={cls_id} out of range [0,8]")
                if not (0 <= cx <= 1):
                    errors.append(f"{lbl.name}:{i+1}: cx={cx} out of range")
                if not (0 <= cy <= 1):
                    errors.append(f"{lbl.name}:{i+1}: cy={cy} out of range")
                if not (0 < w <= 1):
                    errors.append(f"{lbl.name}:{i+1}: w={w} out of range")
                if not (0 < h <= 1):
                    errors.append(f"{lbl.name}:{i+1}: h={h} out of range")

        assert not errors, f"{split}: {len(errors)} label errors:\n" + "\n".join(errors[:20])

    def test_test_labels_if_exist(self):
        """Test set labels (if any) must use valid format too."""
        labels = get_label_paths("test")
        if not labels:
            pytest.skip("No test labels to check")

        errors = []
        for lbl in labels:
            lines = lbl.read_text().strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip metadata lines if any
                try:
                    cls_id = int(float(parts[0]))
                    vals = [float(x) for x in parts[1:]]
                    assert 0 <= cls_id <= 8, f"class_id={cls_id}"
                    assert all(0 <= v <= 1 for v in vals), f"values out of range: {vals}"
                except (ValueError, AssertionError) as e:
                    errors.append(f"{lbl.name}: {e}")

        assert not errors, f"test: {len(errors)} label errors:\n" + "\n".join(errors[:10])
