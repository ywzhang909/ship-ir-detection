"""Pytest bootstrap: make the repo root importable so ``import detector`` works.

pytest runs with the repo root as CWD, but ``sys.path`` may not include it
depending on invocation. Prepending it here keeps every test module simple.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
