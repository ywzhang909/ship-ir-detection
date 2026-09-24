#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建干净底帧 v2: 基于颜色匹配擦除 Ultralytics 旧标签块 (橙红底+黑字)。"""
import cv2
import numpy as np
import pathlib

FRAMES = pathlib.Path("/home/ws/code/ship/data/weights_staging/frames/val")
NAMES = ["val_batch0_pred_101.jpg", "val_batch1_pred_101.jpg", "val_batch2_pred_101.jpg"]


def clean_frame(src, dst):
    img = cv2.imread(str(src))
    if img is None:
        return -1
    b, g, r = cv2.split(img)
    orange = (np.abs(b.astype(int) - 200) < 70) & (g < 140) & (r > 40)
    orange = orange.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(orange, 8)
    mask = np.zeros_like(b)
    erased = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (200 <= area <= 15000 and 6 <= bh <= 110 and 20 <= bw <= 260):
            continue
        mask[max(y - 8, 0):y + bh + 10, max(x - 8, 0):x + bw + 10] = 255
        erased += 1
    print("  " + src.name + ": 匹配 " + str(n - 1) + " 块, 擦除 " + str(erased), end="")
    if mask.sum() > 0:
        clean = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        cv2.imwrite(str(dst), clean)
        print(" -> " + dst.name)
    else:
        cv2.imwrite(str(dst), img)
        print(" -> 无块, 原样复制")
    return erased


def main():
    for name in NAMES:
        src = FRAMES / name
        if src.exists():
            clean_frame(src, FRAMES / name.replace(".jpg", "_clean.jpg"))


if __name__ == "__main__":
    main()
