"""Regenerate all det_*.png detection figures with correct class labels.

The "Ada" bug: single-class models (T1/T2/T4/T5/T7/T16/T20/T22) have
{0: 'ship'} embedded, but the old code mapped class_id 0 → config.SHIP_CLASS_NAMES[0]
= "Ada". This script uses model.names (the model's own embedded class names) as
the single source of truth, so single-class models show "ship" and 9-class models
(baseline/starnet) show the real 9 names.

Usage (from repo root):
    .venv/bin/python yolo-training/scripts/regenerate_detection_figures.py
"""
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_MODELS_W = _REPO / "data" / "weights_staging" / "weights"
_FRAMES = [
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch0_pred_101_clean.jpg",
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch1_pred_101_clean.jpg",
    _REPO / "data" / "weights_staging" / "frames" / "val" / "val_batch2_pred_101_clean.jpg",
]
_OUT_DIR = _REPO / "docs" / "论文" / "figures" / "detection"
_CONF = 0.05
_IMGSZ = 1280  # must be 1280 — matches inference_stats.json (640 gives wrong counts)

MODEL_PREFIX = {
    "baseline_yolo11m": "det_baseline",
    "starnet_s1": "det_starnet",
    "T1_fusion_h32_l3_cbam": "det_T1_fusion",
    "T2_fusion_residual07": "det_T2_fusion",
    "T4_wavelet_dual": "det_T4_wavelet",
    "T5_yolo11l_fusion": "det_T5_yolo11l",
    "T7_augment_jitter": "det_T7_jitter",
    "T16_crrp": "det_T16_crrp",
    "T20_dynamic_head": "det_T20_dyn",
    "T22_large": "det_T22_large",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("regen")


def _model_names_list(model) -> list:
    names = model.names
    if not names:
        return ["class_0"]
    if isinstance(names, dict):
        max_id = max(names.keys())
        return [names.get(i, f"class_{i}") for i in range(max_id + 1)]
    return list(names)


def _draw(image, dets, class_names):
    for cls_id, conf, x1, y1, x2, y2 in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        name = "ship"
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - th - 4, 0)
        cv2.rectangle(image, (x1, ly), (x1 + tw + 2, ly + th + 4), color, -1)
        cv2.putText(image, label, (x1 + 1, ly + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    from ultralytics import YOLO

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, prefix in MODEL_PREFIX.items():
        pt = _MODELS_W / f"{key}_best.pt"
        if not pt.exists():
            log.warning("missing %s — skipping", pt.name)
            continue
        log.info("loading %s", pt.name)
        model = YOLO(str(pt))
        class_names = ["ship"]  # 数据标注纠错：class0 被误标 "Ada"，实为舰船，全部统一 "ship"
        log.info("  class_names = %s", class_names)

        for fi, fpath in enumerate(_FRAMES):
            if not fpath.exists():
                log.warning("missing frame %s", fpath)
                continue
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            results = model(img, conf=_CONF, imgsz=_IMGSZ, verbose=False)
            dets = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        dets.append((cls_id, conf, x1, y1, x2, y2))
            _draw(img, dets, class_names)
            out = _OUT_DIR / f"{prefix}_f{fi}.png"
            cv2.imwrite(str(out), img)
            # 帧0额外写出无后缀副本, 覆盖09-21遗留的含Ada孤儿文件(论文/manifest已改用_fN版)
            if fi == 0:
                cv2.imwrite(str(_OUT_DIR / f"{prefix}.png"), img.copy())
            log.info("  %s → %d detections", out.name, len(dets))
            total += 1
    log.info("done — regenerated %d figures", total)


if __name__ == "__main__":
    main()
