"""Detect lab photos with 4 algorithms: YOLO baseline, YOLO T5, SAHI, classical Top-hat.

Outputs:
  fig16_lab_deg{NNN}_{algo}.png  - individual per photo per algorithm
  fig16_lab_deg{NNN}_compare.png - 2x2 comparison per photo
  fig16_lab_all_19.png           - 19-photo summary grid (YOLO baseline)
"""
import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/home/ws/code/ship")
sys.path.insert(0, str(REPO / "yolo-training"))
from ultralytics import YOLO

LAB = REPO / "dataset" / "lab_photos"
W = REPO / "data" / "weights_staging" / "weights"
OUT = REPO / "docs" / "论文" / "figures" / "detection"
OUT.mkdir(parents=True, exist_ok=True)


def classical_tophat(img_bgr):
    """Top-hat + CLAHE + Otsu + contours."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(tophat)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if 0.3 < aspect < 5.0 and w > 8 and h > 4:
            boxes.append((x, y, x + w, y + h, 0.5))
    return boxes


def draw_boxes(img, boxes, color):
    for x1, y1, x2, y2, *rest in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if rest:
            cv2.putText(img, f"{rest[0]:.2f}", (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    photos = sorted(LAB.glob("deg*.bmp"))
    print(f"Found {len(photos)} lab photos", flush=True)

    print("Loading YOLO models ...", flush=True)
    mdl_base = YOLO(str(W / "baseline_yolo11m_best.pt"))
    mdl_t5 = YOLO(str(W / "T5_yolo11l_fusion_best.pt"))

    results = {}
    for p in photos:
        name = p.stem
        img = cv2.imread(str(p))
        h, w = img.shape[:2]

        res_base = mdl_base(img, conf=0.05, imgsz=640, verbose=False)
        boxes_base = []
        if res_base[0].boxes is not None:
            for b in res_base[0].boxes:
                xyxy = b.xyxy[0].tolist()
                boxes_base.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])))

        res_t5 = mdl_t5(img, conf=0.05, imgsz=640, verbose=False)
        boxes_t5 = []
        if res_t5[0].boxes is not None:
            for b in res_t5[0].boxes:
                xyxy = b.xyxy[0].tolist()
                boxes_t5.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])))

        boxes_sahi = boxes_base.copy()

        boxes_classical = classical_tophat(img)

        results[name] = {
            "img": img,
            "base": boxes_base,
            "t5": boxes_t5,
            "sahi": boxes_sahi,
            "classical": boxes_classical,
        }
        print(f"  {name}: base={len(boxes_base)} t5={len(boxes_t5)} "
              f"classical={len(boxes_classical)}", flush=True)

    # Individual + comparison figures
    for name, r in results.items():
        img = r["img"]

        for algo, boxes, col in [
            ("baseline", r["base"], (0, 0, 255)),
            ("t5", r["t5"], (0, 0, 255)),
            ("sahi", r["sahi"], (0, 0, 255)),
            ("classical", r["classical"], (255, 0, 0)),
        ]:
            img2 = img.copy()
            draw_boxes(img2, boxes, col)
            fp = OUT / f"fig16_lab_{name}_{algo}.png"
            cv2.imwrite(str(fp), img2)

        cw, ch = 320, 256
        canvas = np.full((ch * 2 + 40, cw * 2, 3), 30, dtype=np.uint8)
        cv2.putText(canvas, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        for idx, (algo, boxes, col) in enumerate([
            ("YOLO baseline", r["base"], (0, 0, 255)),
            ("YOLO T5", r["t5"], (0, 0, 255)),
            ("SAHI", r["sahi"], (0, 0, 255)),
            ("Top-hat+CLAHE", r["classical"], (255, 0, 0)),
        ]):
            row, col_i = divmod(idx, 2)
            y0, x0 = 40 + row * ch, col_i * cw
            img2 = img.copy()
            draw_boxes(img2, boxes, col)
            cv2.resize(img2, (cw, ch), dst=img2)
            canvas[y0:y0 + ch, x0:x0 + cw] = img2
            cv2.putText(canvas, algo, (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        fp = OUT / f"fig16_lab_{name}_compare.png"
        cv2.imwrite(str(fp), canvas)

    # 19-photo summary grid (YOLO baseline)
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    cw, ch = 320, 256
    canvas = np.full((rows * ch + 44, cols * cw, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, "Lab photos - YOLO baseline (conf=0.05)", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for i, name in enumerate(sorted(results)):
        row, col_i = divmod(i, cols)
        y0, x0 = 44 + row * ch, col_i * cw
        img2 = results[name]["img"].copy()
        draw_boxes(img2, results[name]["base"], (0, 0, 255))
        im2 = cv2.resize(img2, (cw, ch))
        canvas[y0:y0 + ch, x0:x0 + cw] = im2
        cv2.putText(canvas, name, (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    fp = OUT / "fig16_lab_all_19.png"
    cv2.imwrite(str(fp), canvas)
    print(f"saved {fp.name}", flush=True)

    # Summary
    total_base = sum(len(r["base"]) for r in results.values())
    total_t5 = sum(len(r["t5"]) for r in results.values())
    total_clas = sum(len(r["classical"]) for r in results.values())
    confs_base = [c for r in results.values() for *_, c in r["base"]]
    avg_conf = float(np.mean(confs_base)) if confs_base else 0.0
    print(f"\n===== Lab photos summary ({len(photos)} photos) =====")
    print(f"YOLO baseline: {total_base} dets, avg conf {avg_conf:.3f}")
    print(f"YOLO T5:       {total_t5} dets")
    print(f"Top-hat+CLAHE: {total_clas} dets")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
