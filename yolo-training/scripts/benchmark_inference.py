# -*- coding: utf-8 -*-
"""10 模型推理效率基准（真实测量，反捏造）。

对 data/weights_staging/weights/ 下的 10 个训练权重，在真实验证帧上测量：
  - 参数量 (M)、文件大小 (MB)
  - 平均 GPU 推理延迟 (ms/img) @ imgsz=640 / 1280, batch=1, conf=0.05
  - 吞吐 (FPS, batch=1)
  - 各帧检出数 / 平均置信度 / 最高置信度

输出: docs/论文/figures/inference/inference_efficiency.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]          # ship/
WEIGHTS = REPO_ROOT / "data" / "weights_staging" / "weights"
FRAMES = REPO_ROOT / "data" / "weights_staging" / "frames" / "val"
OUT = REPO_ROOT / "docs" / "论文" / "figures" / "inference" / "inference_efficiency.json"

# 模型显示名（论文用语，无 run 码）
MODEL_DISPLAY = {
    "baseline_yolo11m": "YOLO11m + 空频融合 基线",
    "starnet_s1": "StarNet S1",
    "T1_fusion_h32_l3_cbam": "YOLO11m + 大空频融合架构（过拟合）",
    "T2_fusion_residual07": "YOLO11m + 空频残差搜索架构（过拟合）",
    "T4_wavelet_dual": "YOLO11m + 小波双流融合",
    "T5_yolo11l_fusion": "YOLO11l + 空频融合",
    "T7_augment_jitter": "YOLO11m + 空频融合 + 增强双随机化",
    "T16_crrp": "YOLO11l + 空频融合 + CRRP",
    "T20_dynamic_head": "YOLO11l + 动态检测头",
    "T22_large": "YOLO11l + 大容量融合",
}

IMG_SIZES = (640, 1280)
WARMUP = 3
REPEATS = 8


def _base_name(p: Path) -> str:
    s = p.stem
    return s[: -len("_best")] if s.endswith("_best") else s


def bench_model(weight_path: Path, frames: list[Path]) -> dict:
    m = YOLO(str(weight_path))
    m.to("cuda:0")
    m.model.eval()
    n_params = sum(p.numel() for p in m.model.parameters())

    result: dict = {
        "display_name": MODEL_DISPLAY.get(_base_name(weight_path), weight_path.stem),
        "params_m": round(n_params / 1e6, 2),
        "file_mb": round(weight_path.stat().st_size / 1e6, 2),
        "latency_ms_640": None,
        "latency_ms_1280": None,
        "fps_640": None,
        "detections_per_frame": {},
    }
    all_confs: list[float] = []

    for imgsz in IMG_SIZES:
        # 预热
        for _ in range(WARMUP):
            m(str(frames[0]), imgsz=imgsz, conf=0.05, device="cuda:0", verbose=False)
        torch.cuda.synchronize()

        total_ms = 0.0
        n = 0
        for f in frames:
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                r = m(str(f), imgsz=imgsz, conf=0.05, device="cuda:0", verbose=False)[0]
                torch.cuda.synchronize()
                total_ms += (time.perf_counter() - t0) * 1000.0
                n += 1
            # 记录该帧检出/置信度（最后一次推理结果）
            dets = r.boxes
            confs = [float(c) for c in dets.conf]
            all_confs.extend(confs)
            result["detections_per_frame"][f.name] = {
                "detections": int(len(dets)),
                "avg_conf": round(float(sum(confs) / len(confs)), 4) if confs else 0.0,
                "max_conf": round(float(max(confs)), 4) if confs else 0.0,
            }
        del r
        avg = total_ms / n
        result[f"latency_ms_{imgsz}"] = round(avg, 2)
        result[f"fps_{imgsz}"] = round(1000.0 / avg, 2)

    if all_confs:
        result["avg_conf"] = round(sum(all_confs) / len(all_confs), 4)
        result["max_conf"] = round(max(all_confs), 4)

    del m
    torch.cuda.empty_cache()
    return result


def main() -> None:
    frames = sorted(FRAMES.glob("val_batch*_pred_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {FRAMES}")
    weights = sorted(WEIGHTS.glob("*_best.pt"))
    if not weights:
        raise SystemExit(f"no weights in {WEIGHTS}")

    print(f"frames: {[f.name for f in frames]}")
    print(f"weights: {[w.stem for w in weights]}")
    print("=" * 72)

    out: dict = {"frames": [f.name for f in frames], "models": {}}
    for w in weights:
        if _base_name(w) not in MODEL_DISPLAY:
            print(f"  [skip] unknown model {w.stem}")
            continue
        print(f"  bench {w.stem} ...", end=" ", flush=True)
        t0 = time.time()
        rec = bench_model(w, frames)
        print(f"done ({time.time() - t0:.1f}s)  params={rec['params_m']}M "
              f"lat640={rec['latency_ms_640']}ms lat1280={rec['latency_ms_1280']}ms "
              f"fps640={rec['fps_640']}")
        out["models"][w.stem] = rec

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
