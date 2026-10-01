"""Compare the val and test splits of the ship dataset to explain the val->test gap.

Reports, per split: image count, instance count, per-class counts, bbox area
distribution (absolute px and normalized), bbox aspect ratio, and image sizes.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

DATA_CFG = Path(r"D:\Projects\AUVDect\data\ship\yolo-training\configs\data.yaml")
NAMES = [
    "Ada", "Akizuki", "Alvaro De Bazan", "Armourique", "Independence",
    "Jiangkai II", "Oliver Hazard Perry", "Sejong Daewang", "Zumwalt",
]


def collect(split: str, root: Path) -> dict:
    split_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
        for p in split_dir.glob(ext)
    )

    class_counts: Counter = Counter()
    areas: list[float] = []          # bbox area in px^2
    norm_areas: list[float] = []     # bbox area as fraction of image
    widths: list[float] = []
    heights: list[float] = []
    ars: list[float] = []
    n_labels: list[int] = []

    for img in images:
        lbl = label_dir / f"{img.stem}.txt"
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append((int(parts[0]), *[float(x) for x in parts[1:5]]))
        n_labels.append(len(boxes))
        for cls, cx, cy, w, h in boxes:
            class_counts[cls] += 1
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            norm_areas.append(w * h)
            ars.append(w / h if h > 0 else 0.0)

    return {
        "n_images": len(images),
        "n_boxes": len(areas),
        "boxes_per_image": float(np.mean(n_labels)) if n_labels else 0.0,
        "multi_label_imgs": int(sum(1 for n in n_labels if n > 1)),
        "class_counts": class_counts,
        "norm_area": np.array(norm_areas) if norm_areas else np.array([]),
        "norm_w": np.array(widths) if widths else np.array([]),
        "norm_h": np.array(heights) if heights else np.array([]),
        "ar": np.array(ars) if ars else np.array([]),
    }


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def main() -> None:
    cfg = yaml.safe_load(DATA_CFG.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", ""))
    if not root.is_absolute():
        root = (DATA_CFG.parent / root).resolve()
    print(f"data.yaml      : {DATA_CFG}")
    print(f"dataset root   : {root}")
    print(f"declared splits: {[k for k in cfg if k in ('train', 'val', 'test')]}")
    for k in ("train", "val", "test"):
        if k in cfg:
            print(f"  {k:5s} -> {cfg[k]}")
    print()

    stats = {s: collect(s, root) for s in ("val", "test") if (root / s / "images").exists()}

    print("=" * 78)
    print(f"{'metric':<34}{'val':>20}{'test':>20}")
    print("-" * 78)
    for key, label, fmt in [
        ("n_images", "images", "{:.0f}"),
        ("n_boxes", "instances", "{:.0f}"),
        ("boxes_per_image", "boxes / image", "{:.4f}"),
        ("multi_label_imgs", "images with >1 box", "{:.0f}"),
    ]:
        row = f"{label:<34}"
        for s in ("val", "test"):
            row += fmt.format(stats[s][key]).rjust(20)
        print(row)
    print("-" * 78)

    # bbox size distribution (normalized side lengths, i.e. fraction of image side)
    for key, label in [
        ("norm_w", "bbox width  (norm)"),
        ("norm_h", "bbox height (norm)"),
    ]:
        row = f"{label:<34}"
        for s in ("val", "test"):
            a = stats[s][key]
            row += f"{pct(a,50):.4f}/{pct(a,10):.4f}".rjust(20)
        print(row)
    print(f"{'  (median/p10)':<34}")
    print("-" * 78)

    for q in (10, 25, 50, 75, 90, 99):
        row = f"bbox area p{q} (norm)".ljust(34)
        for s in ("val", "test"):
            row += f"{pct(stats[s]['norm_area'], q):.6f}".rjust(20)
        print(row)
    print("-" * 78)

    for q in (10, 50, 90):
        row = f"aspect ratio p{q} (w/h)".ljust(34)
        for s in ("val", "test"):
            row += f"{pct(stats[s]['ar'], q):.3f}".rjust(20)
        print(row)
    print("-" * 78)

    # implied pixel size at imgsz=640 letterbox
    print("implied bbox size at imgsz=640 (median norm side * 640, px):")
    for s in ("val", "test"):
        w = pct(stats[s]["norm_w"], 50) * 640
        h = pct(stats[s]["norm_h"], 50) * 640
        small = float(np.mean(stats[s]["norm_w"] * 640 < 32)) * 100
        print(f"  {s:5s} median {w:5.1f} x {h:5.1f} px   |  frac width<32px: {small:5.2f}%")
    print()

    print("=" * 78)
    print(f"{'class':<24}{'val':>12}{'test':>12}{'delta':>12}{'test share':>14}")
    print("-" * 78)
    tot_v = sum(stats["val"]["class_counts"].values()) or 1
    tot_t = sum(stats["test"]["class_counts"].values()) or 1
    for i, name in enumerate(NAMES):
        v = stats["val"]["class_counts"].get(i, 0)
        t = stats["test"]["class_counts"].get(i, 0)
        print(f"{name:<24}{v:>12}{t:>12}{t - v:>+12}{t / tot_t * 100:>13.2f}%")
    print("-" * 78)
    print(f"{'TOTAL':<24}{tot_v:>12}{tot_t:>12}{tot_t - tot_v:>+12}")


if __name__ == "__main__":
    sys.exit(main())
