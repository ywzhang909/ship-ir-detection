"""Evaluate each ship-dataset model on BOTH val and test through one identical
code path, to test whether the val metrics in results.csv are comparable to the
test metrics from scripts/evaluate_test.py.

If the splits are i.i.d. (verified by compare_splits.py), then for a given model
val and test metrics must agree within sampling noise. Any large disagreement
localises the problem to the evaluation code path, not to the data.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

from ultralytics import YOLO

RUNS = Path(r"D:\Projects\AUVDect\data\ship\runs\detect\ship-detection")
DATA = r"D:\Projects\AUVDect\data\ship\yolo-training\configs\data.yaml"
OUT = Path(r"D:\Projects\AUVDect\data\ship\yolo-training\results\val_test_parity.json")

MODELS = ["baseline-yolo11m", "starnet-s1", "leconv-t"]


def main() -> None:
    results = {}
    for name in MODELS:
        w = RUNS / name / "weights" / "best.pt"
        if not w.exists():
            print(f"SKIP {name}: no best.pt", flush=True)
            continue
        model = YOLO(str(w))
        results[name] = {}
        for split in ("val", "test"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m = model.val(
                    data=DATA, split=split, conf=0.001, imgsz=640,
                    batch=16, workers=0, verbose=False, plots=False,
                )
            results[name][split] = {
                "mAP50-95": round(m.box.map * 100, 2),
                "mAP50": round(m.box.map50 * 100, 2),
                "Precision": round(m.box.mp * 100, 2),
                "Recall": round(m.box.mr * 100, 2),
            }
            print(f"{name:20s} {split:5s} "
                  f"mAP50-95={results[name][split]['mAP50-95']:6.2f}  "
                  f"mAP50={results[name][split]['mAP50']:6.2f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'model':<20}{'val mAP50-95':>14}{'test mAP50-95':>15}{'delta':>12}")
    print("-" * 74)
    for name, d in results.items():
        if "val" in d and "test" in d:
            delta = d["test"]["mAP50-95"] - d["val"]["mAP50-95"]
            print(f"{name:<20}{d['val']['mAP50-95']:>14.2f}{d['test']['mAP50-95']:>15.2f}"
                  f"{delta:>+12.2f}")
    print("=" * 74)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
