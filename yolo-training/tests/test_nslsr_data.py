"""
Tests for NSLSR dataset integrity.

Verifies that the processed NSLSR dataset (data/processed_nslsr/) is valid:
  - Image-label pairing per split and modality
  - Split disjointness within each modality
  - Exact expected image counts per split
  - YOLO label format correctness (single class, normalized coords)
  - Modality count parity (SWIR == LWIR per split)

Usage:
    uv run pytest tests/test_nslsr_data.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NSLSR_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_nslsr"

pytestmark = pytest.mark.skipif(
    not NSLSR_DIR.exists(),
    reason=f"NSLSR data directory not found at {NSLSR_DIR}. Run prepare_nslsr_data.py first.",
)

MODALITIES = ["swir", "lwir"]
SPLITS = ["train", "val", "test"]
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG"}
EXPECTED_COUNTS: dict[str, int] = {
    "train": 803,
    "val": 243,
    "test": 118,
}
VALID_CLASS_IDS = {0}  # Single class: ship


def _nslsr_image_dir(modality: str, split: str) -> Path:
    """Return the image directory for the given modality and split."""
    return NSLSR_DIR / modality / split / "images"


def _nslsr_label_dir(modality: str, split: str) -> Path:
    """Return the label directory for the given modality and split."""
    return NSLSR_DIR / modality / split / "labels"


def get_image_paths(modality: str, split: str) -> list[Path]:
    """Return all image paths for a given modality and split."""
    img_dir = _nslsr_image_dir(modality, split)
    if not img_dir.exists():
        return []
    return [p for p in img_dir.iterdir() if p.suffix in ALLOWED_IMAGE_EXTS]


def get_label_paths(modality: str, split: str) -> list[Path]:
    """Return all label paths for a given modality and split."""
    label_dir = _nslsr_label_dir(modality, split)
    if not label_dir.exists():
        return []
    return list(label_dir.glob("*.txt"))


def _modality_split_pair_params():
    """Generate (modality, split) parameter tuples for tests that run on all combos."""
    return [(m, s) for m in MODALITIES for s in SPLITS]


# ---------------------------------------------------------------------------
# Tests: Data existence
# ---------------------------------------------------------------------------

class TestNslsrDataExists:
    """The processed NSLSR directory and expected modality/split subdirs exist."""

    def test_nslsr_root_exists(self):
        """Root processed_nslsr directory must exist."""
        assert NSLSR_DIR.is_dir(), f"NSLSR directory not found: {NSLSR_DIR}"

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_modality_dir_exists(self, modality):
        """Each modality directory must exist."""
        mod_dir = NSLSR_DIR / modality
        assert mod_dir.is_dir(), f"Modality directory not found: {mod_dir}"

    @pytest.mark.parametrize("modality,split", _modality_split_pair_params())
    def test_split_dir_exists(self, modality, split):
        """Each split directory must exist within each modality."""
        split_dir = NSLSR_DIR / modality / split
        assert split_dir.is_dir(), f"Split directory not found: {split_dir}"

    @pytest.mark.parametrize("modality,split", _modality_split_pair_params())
    def test_images_dir_exists(self, modality, split):
        """Images subdirectory must exist for each modality/split."""
        img_dir = _nslsr_image_dir(modality, split)
        assert img_dir.is_dir(), f"Images directory not found: {img_dir}"

    @pytest.mark.parametrize("modality,split", _modality_split_pair_params())
    def test_labels_dir_exists(self, modality, split):
        """Labels subdirectory must exist for each modality/split."""
        lbl_dir = _nslsr_label_dir(modality, split)
        assert lbl_dir.is_dir(), f"Labels directory not found: {lbl_dir}"


# ---------------------------------------------------------------------------
# Tests: Image-label pairing
# ---------------------------------------------------------------------------

class TestImageLabelPairing:
    """Every image must have a corresponding label file and vice versa."""

    @pytest.mark.parametrize("modality,split", _modality_split_pair_params())
    def test_every_image_has_label(self, modality, split):
        """Each image must have a matching .txt label."""
        images = get_image_paths(modality, split)
        labels = get_label_paths(modality, split)

        label_stems = {p.stem for p in labels}
        missing = [img.name for img in images if img.stem not in label_stems]

        assert not missing, (
            f"{modality}/{split}: {len(missing)} images missing labels: {missing[:10]}"
        )

    @pytest.mark.parametrize("modality,split", _modality_split_pair_params())
    def test_every_label_has_image(self, modality, split):
        """Each .txt label must have a matching image."""
        images = get_image_paths(modality, split)
        labels = get_label_paths(modality, split)

        image_stems = {p.stem for p in images}
        missing = [lbl.name for lbl in labels if lbl.stem not in image_stems]

        assert not missing, (
            f"{modality}/{split}: {len(missing)} labels missing images: {missing[:10]}"
        )


# ---------------------------------------------------------------------------
# Tests: Split disjointness
# ---------------------------------------------------------------------------

class TestSplitDisjointness:
    """No image filename should appear in more than one split within a modality."""

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_no_overlap_train_val(self, modality):
        """Train and val must be disjoint."""
        train_stems = {p.stem for p in get_image_paths(modality, "train")}
        val_stems = {p.stem for p in get_image_paths(modality, "val")}
        overlap = train_stems & val_stems
        assert not overlap, (
            f"{modality}: {len(overlap)} images appear in both train and val"
        )

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_no_overlap_train_test(self, modality):
        """Train and test must be disjoint."""
        train_stems = {p.stem for p in get_image_paths(modality, "train")}
        test_stems = {p.stem for p in get_image_paths(modality, "test")}
        overlap = train_stems & test_stems
        assert not overlap, (
            f"{modality}: {len(overlap)} images appear in both train and test"
        )

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_no_overlap_val_test(self, modality):
        """Val and test must be disjoint."""
        val_stems = {p.stem for p in get_image_paths(modality, "val")}
        test_stems = {p.stem for p in get_image_paths(modality, "test")}
        overlap = val_stems & test_stems
        assert not overlap, (
            f"{modality}: {len(overlap)} images appear in both val and test"
        )


# ---------------------------------------------------------------------------
# Tests: Expected counts
# ---------------------------------------------------------------------------

class TestSplitCounts:
    """Each split must contain the expected number of images per modality."""

    @pytest.mark.parametrize("modality", MODALITIES)
    @pytest.mark.parametrize("split", SPLITS)
    def test_image_count(self, modality, split):
        """Image count must match the expected value."""
        images = get_image_paths(modality, split)
        expected = EXPECTED_COUNTS[split]
        actual = len(images)
        assert actual == expected, (
            f"{modality}/{split}: expected {expected} images, got {actual}"
        )

    @pytest.mark.parametrize("modality", MODALITIES)
    @pytest.mark.parametrize("split", SPLITS)
    def test_label_count(self, modality, split):
        """Label count must match the expected value."""
        labels = get_label_paths(modality, split)
        expected = EXPECTED_COUNTS[split]
        actual = len(labels)
        assert actual == expected, (
            f"{modality}/{split}: expected {expected} labels, got {actual}"
        )


# ---------------------------------------------------------------------------
# Tests: Modality count parity
# ---------------------------------------------------------------------------

class TestModalityCountsMatch:
    """SWIR and LWIR must have identical image counts per split."""

    @pytest.mark.parametrize("split", SPLITS)
    def test_modality_counts_match(self, split):
        """Image counts for SWIR and LWIR must be identical."""
        swir_count = len(get_image_paths("swir", split))
        lwir_count = len(get_image_paths("lwir", split))
        assert swir_count == lwir_count, (
            f"{split}: SWIR has {swir_count} images but LWIR has {lwir_count}"
        )


# ---------------------------------------------------------------------------
# Tests: Label format correctness
# ---------------------------------------------------------------------------

class TestLabelFormat:
    """Each label file must contain valid YOLO-format annotations."""

    @pytest.mark.parametrize("modality", MODALITIES)
    @pytest.mark.parametrize("split", SPLITS)
    def test_label_format(self, modality, split):
        """Each label line must be: class_id cx cy w h, all in [0,1]."""
        labels = get_label_paths(modality, split)
        errors = []

        for lbl in labels:
            lines = lbl.read_text().strip().splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: "
                        f"expected 5 values, got {len(parts)}"
                    )
                    continue

                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: non-numeric values"
                    )
                    continue

                cls_id = int(vals[0])
                cx, cy, w, h = vals[1:]

                if cls_id not in VALID_CLASS_IDS:
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: "
                        f"class_id={cls_id} out of range {VALID_CLASS_IDS}"
                    )
                if not (0 <= cx <= 1):
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: cx={cx} out of range [0,1]"
                    )
                if not (0 <= cy <= 1):
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: cy={cy} out of range [0,1]"
                    )
                if not (0 < w <= 1):
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: w={w} out of range (0,1]"
                    )
                if not (0 < h <= 1):
                    errors.append(
                        f"{modality}/{split}/{lbl.name}:{i+1}: h={h} out of range (0,1]"
                    )

        assert not errors, (
            f"{modality}/{split}: {len(errors)} label errors:\n"
            + "\n".join(errors[:20])
        )
