"""Generate detection figures: 10 models x 10 NSLSR samples + fusion comparison.

Outputs:
  fig15_{model}_{waveband}_{sample}.png  - individual (LWIR + SWIR, 5 samples each)
  fig15_{model}_mosaic.png               - 10-cell mosaic per model
  fig17_fusion_module_comparison.png     - 5 fusion variants x 5 LWIR samples
"""
import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/home/ws/code/ship")
sys.path.insert(0, str(REPO / "yolo-training"))
from ultralytics import YOLO

W = REPO / "data" / "weights_staging" / "weights"
DS = REPO / "dataset" / "simple"
OUT = REPO / "docs" / "论文" / "figures" / "detection"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["0030", "0250", "0500", "0700", "1000"]
WAVEBANDS = ["LWIR", "SWIR"]

MODELS = {
    "baseline_yolo11m": "baseline",
    "starnet_s1": "starnet",
    "T1_fusion_h32_l3_cbam": "T1_fusion",
    "T2_fusion_residual07": "T2_fusion",
    "T4_wavelet_dual": "T4_wavelet",
    "T5_yolo11l_fusion": "T5_yolo11l",
    "T7_augment_jitter": "T7_jitter",
    "T16_crrp": "T16_crrp",
    "T20_dynamic_head": "T20_dyn",
    "T22_large": "T22_large",
}
FUSION_VARIANTS = ["baseline_yolo11m", "T1_fusion_h32_l3_cbam",
                   "T2_fusion_residual07", "T4_wavelet_dual", "T5_yolo11l_fusion"]
FUSION_LABELS = {
    "baseline_yolo11m": "Baseline(h16,SE)",
    "T1_fusion_h32_l3_cbam": "T1(h32,CBAM)",
    "T2_fusion_residual07": "T2(res=0.7)",
    "T4_wavelet_dual": "T4(wavelet)",
    "T5_yolo11l_fusion": "T5(YOLO11l,SE)",
}


def load_gt(sample, w, h):
    p = DS / "labels" / f"{sample}.txt"
    boxes = []
    if p.exists():
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                          int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw_on(img, preds, gts):
    for x1, y1, x2, y2 in gts:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "GT", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for x1, y1, x2, y2, c in preds:
        col = (0, 255, 255) if c >= 0.5 else (0, 0, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), max(int(y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def make_mosaic(cells, ncols, title, cellw=640, cellh=512):
    nrows = (len(cells) + ncols - 1) // ncols
    canvas = np.full((nrows * cellh + 44, ncols * cellw, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    for i, (im, label) in enumerate(cells):
        r, c = divmod(i, ncols)
        y0, x0 = 44 + r * cellh, c * cellw
        im2 = cv2.resize(im, (cellw, cellh)) if im.shape[:2] != (cellh, cellw) else im
        canvas[y0:y0 + cellh, x0:x0 + cellw] = im2
        cv2.putText(canvas, label, (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return canvas


def main():
    data = {}
    for wb in WAVEBANDS:
        for s in SAMPLES:
            p = DS / wb / f"{s}.jpg"
            img = cv2.imread(str(p))
            if img is None:
                print(f"MISSING {p}", flush=True)
                continue
            h, w = img.shape[:2]
            data[(wb, s)] = (img, load_gt(s, w, h))
            print(f"loaded {wb}/{s} {w}x{h} gt={len(data[(wb, s)][1])}", flush=True)

    models = {}
    for key in MODELS:
        pt = W / f"{key}_best.pt"
        if not pt.exists():
            print(f"MISSING weight {pt.name}", flush=True)
            continue
        print(f"Loading {pt.name} ...", flush=True)
        models[key] = YOLO(str(pt))

    cache = {k: {} for k in models}
    stats = {k: {"dets": 0, "confs": [], "hits": 0, "gt": 0} for k in models}

    for key, mdl in models.items():
        for wb in WAVEBANDS:
            for s in SAMPLES:
                if (wb, s) not in data:
                    continue
                img, gt = data[(wb, s)]
                res = mdl(img, conf=0.05, imgsz=1280, verbose=False)
                preds = []
                if res[0].boxes is not None:
                    for b in res[0].boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        preds.append((x1, y1, x2, y2, float(b.conf[0])))
                cache[key][(wb, s)] = preds
                st = stats[key]
                st["dets"] += len(preds)
                st["confs"].extend([p[4] for p in preds])
                st["gt"] += len(gt)
                st["hits"] += sum(
                    1 for g in gt
                    if max((iou(p[:4], g) for p in preds), default=0.0) > 0.3)
                print(f"  {MODELS[key]:12s} {wb}/{s}: {len(preds)} dets", flush=True)

    for key, prefix in MODELS.items():
        if key not in cache:
            continue
        cells = []
        for wb in WAVEBANDS:
            for s in SAMPLES:
                if (wb, s) not in data:
                    continue
                img, gt = data[(wb, s)]
                img2 = img.copy()
                draw_on(img2, cache[key][(wb, s)], gt)
                cells.append((img2, f"{wb}/{s}"))
        m = make_mosaic(cells, ncols=5, title=f"{prefix}")
        fp = OUT / f"fig15_{prefix}_mosaic.png"
        cv2.imwrite(str(fp), m)
        print(f"saved {fp.name} ({len(cells)} cells)", flush=True)

    cw, ch = 480, 384
    n_c = len(SAMPLES)
    n_r = len(FUSION_VARIANTS)
    canvas = np.full(((n_r + 1) * ch, n_c * cw, 3), 40, dtype=np.uint8)
    for j, s in enumerate(SAMPLES):
        cv2.putText(canvas, f"LWIR/{s}", (j * cw + 10, ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, k in enumerate(FUSION_VARIANTS):
        if k not in cache:
            continue
        y0 = (i + 1) * ch
        for j, s in enumerate(SAMPLES):
            if ("LWIR", s) not in data:
                continue
            img, gt = data[("LWIR", s)]
            img2 = img.copy()
            draw_on(img2, cache[k][("LWIR", s)], gt)
            im2 = cv2.resize(img2, (cw, ch))
            canvas[y0:y0 + ch, j * cw:j * cw + cw] = im2
        cv2.putText(canvas, FUSION_LABELS[k], (8, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "Fusion ablation (GT=green, pred=red/yellow, conf=0.05)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    fp = OUT / "fig17_fusion_module_comparison.png"
    cv2.imwrite(str(fp), canvas)
    print(f"saved {fp.name}", flush=True)

    print("\n===== Summary (10 models x 10 samples) =====")
    print(f"{'Model':12s} {'Dets':>5s} {'Avg%':>6s} {'Max%':>6s} {'GT hit':>9s}")
    for key, prefix in MODELS.items():
        if key not in stats:
            continue
        st = stats[key]
        avg = float(np.mean(st["confs"])) if st["confs"] else 0.0
        mx = max(st["confs"]) if st["confs"] else 0.0
        print(f"{prefix:12s} {st['dets']:5d} {avg * 100:5.1f}% {mx * 100:5.1f}% "
              f"{st['hits']}/{st['gt']:>3d}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
