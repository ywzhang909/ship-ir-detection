"""Pytest bootstrap for the ship_detector GUI package.

``core``/``ui``/``config`` are imported as TOP-LEVEL modules (matching how
``main.py`` runs), so ``ship_detector/`` itself must be on ``sys.path`` —
not the repo root.
"""

import os
import sys
from pathlib import Path

_SHIP_DETECTOR = Path(__file__).resolve().parents[1]
if str(_SHIP_DETECTOR) not in sys.path:
    sys.path.insert(0, str(_SHIP_DETECTOR))

# Qt must not try to reach a display from the test runner.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_DEVICE_PIXEL_RATIO", "1")