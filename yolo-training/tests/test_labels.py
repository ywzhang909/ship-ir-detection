"""
Tests for YOLO pseudo-label generation correctness.

Verifies that generated labels meet YOLO format requirements:
  - Values in [0, 1]
  - Center at (0.5, 0.5)
  - Correct class IDs
  - Consistent aspect ratios
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_labels import (
    CLASS_MAP,
    CLASS_NAMES,
    IMG_WIDTH,
    IMG_HEIGHT,
    CX_NORM,
    CY_NORM,
    generate_label_line,
    get_aspect_ratio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_CLASSES = sorted(CLASS_MAP.keys())
ALL_HEIGHT_RANGES = ["5-9", "10-13", "14-17", "18-21", "22-25", "26-30"]
ALL_RANGE_STRS = ["0-29", "30-59", "60-89"]

# Expected aspect ratios
EXPECTED_ASPECTS = {
    "0-29": 4.0,
    "30-59": 3.0,
    "60-89": 2.0,
}


# ---------------------------------------------------------------------------
# Tests: Center constraint
# ---------------------------------------------------------------------------

class TestCenter:
    """The ship is always centered in synthetic images."""

    def test_center_is_always_05(self):
        """CX_NORM and CY_NORM must both be 0.5."""
        assert CX_NORM == 0.5, f"Expected CX_NORM=0.5, got {CX_NORM}"
        assert CY_NORM == 0.5, f"Expected CY_NORM=0.5, got {CY_NORM}"

    @pytest.mark.parametrize("cls_name", ALL_CLASSES)
    @pytest.mark.parametrize("height", ALL_HEIGHT_RANGES)
    @pytest.mark.parametrize("range_str", ALL_RANGE_STRS)
    def test_label_center_is_05(self, cls_name, height, range_str):
        """Every generated label must have cx=0.5, cy=0.5."""
        line = generate_label_line(cls_name, height, range_str)
        parts = line.strip().split()
        cx, cy = float(parts[1]), float(parts[2])
        assert cx == 0.5, f"{cls_name}/{height}/{range_str}: cx={cx}"
        assert cy == 0.5, f"{cls_name}/{height}/{range_str}: cy={cy}"


# ---------------------------------------------------------------------------
# Tests: Normalization bounds
# ---------------------------------------------------------------------------

class TestNormalizationBounds:
    """All YOLO values must be in [0, 1]."""

    @pytest.mark.parametrize("cls_name", ALL_CLASSES)
    @pytest.mark.parametrize("height", ALL_HEIGHT_RANGES)
    @pytest.mark.parametrize("range_str", ALL_RANGE_STRS)
    def test_all_values_in_01(self, cls_name, height, range_str):
        """class_id, cx, cy, w, h must all be in [0, 1]."""
        line = generate_label_line(cls_name, height, range_str)
        parts = line.strip().split()
        assert len(parts) == 5, f"Expected 5 values, got {len(parts)}"

        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        assert 0 <= class_id <= 8, f"class_id={class_id} out of range"
        assert 0 <= cx <= 1, f"cx={cx} out of range"
        assert 0 <= cy <= 1, f"cy={cy} out of range"
        assert 0 < w <= 1, f"w={w} out of range"
        assert 0 < h <= 1, f"h={h} out of range"


# ---------------------------------------------------------------------------
# Tests: Class ID mapping
# ---------------------------------------------------------------------------

class TestClassIDs:
    """All 9 classes must map to correct IDs."""

    def test_all_classes_mapped(self):
        """All 9 classes must be in CLASS_MAP."""
        assert len(CLASS_MAP) == 9, f"Expected 9 classes, got {len(CLASS_MAP)}"

    @pytest.mark.parametrize("cls_name, expected_id", [
        ("Ada", 0),
        ("Akizuki", 1),
        ("Alvaro De Bazan", 2),
        ("Armourique", 3),
        ("Independence", 4),
        ("Jiangkai II", 5),
        ("Oliver Hazard Perry", 6),
        ("Sejong Daewang", 7),
        ("Zumwalt", 8),
    ])
    def test_class_id_mapping(self, cls_name, expected_id):
        """Each class name must map to the correct ID."""
        assert CLASS_MAP[cls_name] == expected_id

    def test_reverse_mapping(self):
        """CLASS_NAMES must correctly reverse CLASS_MAP."""
        assert len(CLASS_NAMES) == 9
        for cls_id, name in CLASS_NAMES.items():
            assert CLASS_MAP[name] == cls_id


# ---------------------------------------------------------------------------
# Tests: Aspect ratio consistency
# ---------------------------------------------------------------------------

class TestAspectRatio:
    """Width/height ratio must match expected value for each range."""

    @pytest.mark.parametrize("range_str, expected_aspect", EXPECTED_ASPECTS.items())
    def test_aspect_ratio_function(self, range_str, expected_aspect):
        """get_aspect_ratio must return expected values."""
        assert get_aspect_ratio(range_str) == expected_aspect

    @pytest.mark.parametrize("cls_name", ALL_CLASSES)
    def test_aspect_ratio_broadside(self, cls_name):
        """0-29 range: aspect ratio must be ~4.0."""
        line = generate_label_line(cls_name, "14-17", "0-29")
        parts = line.strip().split()
        w, h = float(parts[3]), float(parts[4])
        actual_aspect = (w * IMG_WIDTH) / (h * IMG_HEIGHT)
        assert abs(actual_aspect - 4.0) < 0.01, f"Expected aspect ~4.0, got {actual_aspect}"

    @pytest.mark.parametrize("cls_name", ALL_CLASSES)
    def test_aspect_ratio_angled(self, cls_name):
        """30-59 range: aspect ratio must be ~3.0."""
        line = generate_label_line(cls_name, "14-17", "30-59")
        parts = line.strip().split()
        w, h = float(parts[3]), float(parts[4])
        actual_aspect = (w * IMG_WIDTH) / (h * IMG_HEIGHT)
        assert abs(actual_aspect - 3.0) < 0.01, f"Expected aspect ~3.0, got {actual_aspect}"

    @pytest.mark.parametrize("cls_name", ALL_CLASSES)
    def test_aspect_ratio_headon(self, cls_name):
        """60-89 range: aspect ratio must be ~2.0."""
        line = generate_label_line(cls_name, "14-17", "60-89")
        parts = line.strip().split()
        w, h = float(parts[3]), float(parts[4])
        actual_aspect = (w * IMG_WIDTH) / (h * IMG_HEIGHT)
        assert abs(actual_aspect - 2.0) < 0.01, f"Expected aspect ~2.0, got {actual_aspect}"


# ---------------------------------------------------------------------------
# Tests: Midpoint height
# ---------------------------------------------------------------------------

class TestHeightMidpoint:
    """Box height must be midpoint of the pixel height range."""

    @pytest.mark.parametrize("height_range, expected_midpoint", [
        ("5-9", 7.0),
        ("10-13", 11.5),
        ("14-17", 15.5),
        ("18-21", 19.5),
        ("22-25", 23.5),
        ("26-30", 28.0),
    ])
    def test_height_midpoint(self, height_range, expected_midpoint):
        """Box height (in pixels) must match the midpoint."""
        line = generate_label_line("Zumwalt", height_range, "30-59")
        parts = line.strip().split()
        h_norm = float(parts[4])
        h_px = h_norm * IMG_HEIGHT
        assert abs(h_px - expected_midpoint) < 0.01, (
            f"Expected height ~{expected_midpoint}px, got {h_px}px"
        )


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions must produce valid labels."""

    def test_smallest_height_smallest_class(self):
        """Minimum height (5-9) with max aspect (4.0) must be valid."""
        line = generate_label_line("Ada", "5-9", "0-29")
        parts = line.strip().split()
        w, h = float(parts[3]), float(parts[4])
        assert 0 < w <= 1, f"w={w} out of range"
        assert 0 < h <= 1, f"h={h} out of range"

    def test_largest_height_largest_class(self):
        """Maximum height (26-30) with min aspect (2.0) must be valid."""
        line = generate_label_line("Zumwalt", "26-30", "60-89")
        parts = line.strip().split()
        w, h = float(parts[3]), float(parts[4])
        assert 0 < w <= 1, f"w={w} out of range"
        assert 0 < h <= 1, f"h={h} out of range"

    def test_unknown_class_raises_error(self):
        """Unknown class name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown class"):
            generate_label_line("NonExistentShip", "10-13", "30-59")

    def test_invalid_height_range(self):
        """Invalid height range must raise ValueError."""
        with pytest.raises(ValueError):
            generate_label_line("Ada", "invalid", "30-59")


# ---------------------------------------------------------------------------
# Tests: Specific example verification
# ---------------------------------------------------------------------------

class TestExampleVerification:
    """Manually computed examples must match exactly."""

    def test_example_sejong_18_21_30_59(self):
        """Sejong Daewang, height=18-21, range=30-59 (aspect 3.0).
        Expected: h_px = (18+21)/2 = 19.5, w_px = 19.5*3 = 58.5
        h_norm = 19.5/512 = 0.038086, w_norm = 58.5/1024 = 0.057129
        class_id=7, cx=0.5, cy=0.5
        """
        line = generate_label_line("Sejong Daewang", "18-21", "30-59")
        parts = line.strip().split()
        assert parts[0] == "7", f"Expected class_id=7, got {parts[0]}"
        assert parts[1] == "0.500000", f"Expected cx=0.5, got {parts[1]}"
        assert parts[2] == "0.500000", f"Expected cy=0.5, got {parts[2]}"

        w = float(parts[3])
        h = float(parts[4])
        h_expected = 19.5 / IMG_HEIGHT
        w_expected = (19.5 * 3.0) / IMG_WIDTH
        assert abs(h - h_expected) < 1e-5, f"h: expected {h_expected}, got {h}"
        assert abs(w - w_expected) < 1e-5, f"w: expected {w_expected}, got {w}"

    def test_example_ada_5_9_0_29(self):
        """Ada, height=5-9, range=0-29 (aspect 4.0).
        h_px = 7, w_px = 28
        h_norm = 7/512 = 0.013672, w_norm = 28/1024 = 0.027344
        class_id=0
        """
        line = generate_label_line("Ada", "5-9", "0-29")
        parts = line.strip().split()
        assert parts[0] == "0"
        w, h = float(parts[3]), float(parts[4])
        assert abs(w - 28.0 / 1024) < 1e-5
        assert abs(h - 7.0 / 512) < 1e-5
