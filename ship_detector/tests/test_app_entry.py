"""Contract tests for the application entry path (main.py's setup).

`main.py` does more than construct `MainWindow`: it enables the Fusion style and
loads `resources/styles/main.qss`. Nothing else in the suite covers that path, so a
broken stylesheet path or a missing toolbar action would ship unnoticed.
"""

from __future__ import annotations

import pytest

from config import PROJECT_ROOT


def test_main_qss_stylesheet_exists_and_is_non_trivial():
    qss = PROJECT_ROOT / "resources" / "styles" / "main.qss"
    assert qss.exists(), f"stylesheet missing: {qss}"
    assert qss.stat().st_size > 200, "stylesheet looks empty/stubbed"


def test_entry_path_builds_a_window_with_the_expected_toolbar(qapp):
    """Mirror main.py: set Fusion style, apply the QSS, build the window."""
    from config import PROJECT_ROOT as root
    from ui.main_window import MainWindow

    qss = root / "resources" / "styles" / "main.qss"
    qapp.setStyle("Fusion")
    qapp.setStyleSheet(qss.read_text(encoding="utf-8"))

    win = MainWindow()
    try:
        labels = [a.text() for a in win.toolbar.actions() if a.text()]
        for expected in ("打开", "开始检测", "连续检测", "暂停", "导出结果"):
            assert expected in labels, f"toolbar missing {expected!r}: {labels}"

        assert win.detect_worker is not None
        assert win.detect_worker.isRunning() is False, "worker must not auto-start"
        assert win.detector.preprocess == "raw"
    finally:
        win.close()


def test_closing_a_window_with_a_running_worker_does_not_hang(qapp, tmp_path):
    """closeEvent must shut the continuous worker down promptly."""
    import time

    import numpy as np

    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector._is_loaded = False  # avoid touching the real model
    win.set_continuous_detection(True)
    win.canvas.frame_captured.emit(np.zeros((16, 16, 3), np.uint8), 1)

    start = time.monotonic()
    win.close()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"closeEvent took {elapsed:.2f}s — worker shutdown hung"
    assert win.detect_worker.isRunning() is False
