"""AppTest suite for app.py.

Covers boot health and default widget state only. Streamlit's AppTest cannot
inject values into ``st.file_uploader``, so the upload/inference flow is
deliberately NOT exercised here (it is covered by unit tests against
``detector`` elsewhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"

# Defensive mirror of tests/conftest.py so ``import detector`` resolves even
# if this module is ever collected without the conftest bootstrap.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch ships with the project venv
    _HAS_CUDA = False

_requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


def _boot_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    return at


def test_app_boots_without_exception() -> None:
    at = _boot_app()
    assert not at.exception


def test_model_selectbox_defaults_to_best_model() -> None:
    at = _boot_app()
    model_box = at.sidebar.selectbox[0]
    assert "T5_yolo11l_fusion" in str(model_box.value)


@_requires_cuda
def test_gpu_badge_shows_cuda() -> None:
    at = _boot_app()
    texts = [el.value for el in at.sidebar.success] + [el.value for el in at.success]
    assert "cuda" in "\n".join(texts).lower()


def test_conf_and_imgsz_defaults() -> None:
    at = _boot_app()
    assert at.sidebar.slider[0].value == pytest.approx(0.25)

    # AppTest serializes selectbox .options to their formatted (string) form,
    # so match on str(option); .value stays the raw option (int 640).
    imgsz_boxes = [
        sb for sb in at.sidebar.selectbox if any(str(opt) == "640" for opt in sb.options)
    ]
    assert imgsz_boxes, "expected an image-size selectbox offering 640"
    assert imgsz_boxes[0].value == 640


def test_rerun_after_model_switch() -> None:
    target = next(
        (m for m in detector.discover_models() if "smoke_shape_iou" in str(m.path)),
        None,
    )
    if target is None:
        pytest.skip("smoke_shape_iou model is not present under runs/detect")
    at = _boot_app()
    at.sidebar.selectbox[0].set_value(str(target.path))
    at.run()
    assert not at.exception
