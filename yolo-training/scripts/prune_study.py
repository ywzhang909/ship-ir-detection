# -*- coding: utf-8 -*-
"""对最佳 large 模型（YOLO11l + 空频融合）尝试不同剪枝算法。零-shot（不微调）。
算法: 幅值/L1/随机(非结构化) + 通道(结构化), 稀疏度 30/50/70%。
每个变体测: 参数/文件大小/延迟@640,1280/FPS/3 帧检出与置信度。
输出: pruning/{key}.pt + pruning_study.json
"""
from __future__ import annotations
import copy, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import torch
import torch.nn.utils.prune as P
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/weights_staging/weights/T5_yolo11l_fusion_best.pt"
FRAMES_DIR = REPO / "data/weights_staging/frames/val"
OUT_DIR = REPO / "docs/论文/figures/inference/pruning"
OUT_JSON = REPO / "docs/论文/figures/inference/pruning_study.json"
SPARSITY = (0.30, 0.50, 0.70)
METHODS = ("unstructured_magnitude", "unstructured_l1", "unstructured_random", "structured_channel")
WARMUP = 3


def weight_params(net):
    out = []
    for m in net.modules():
        for pname in ("weight", "bias"):
            w = getattr(m, pname, None)
            if w is None or not isinstance(w, torch.nn.Parameter) or w.dim() not in (2, 3, 4):
                continue
            out.append(w)
    return out


def nnz_total(net):
    total = 0
    for w in weight_params(net):
        total += int((w != 0).sum().item())
    return total


def apply_magnitude(net, pct):
    for w in weight_params(net):
        thr = torch.quantile(w.float().abs().flatten(), pct)
        with torch.no_grad():
            w.copy_(w.masked_fill(w.float().abs() < thr, 0.0))


def apply_torchprune_unstructured(net, pct, kind):
    fn = P.l1_unstructured if kind == "l1" else P.random_unstructured
    for m in list(net.modules()):
        for pname in ("weight", "bias"):
            w = getattr(m, pname, None)
            if w is None or not isinstance(w, torch.nn.Parameter) or w.dim() not in (2, 3, 4):
                continue
            fn(m, name=pname, amount=pct)


def apply_structured(net, pct):
    for m in list(net.modules()):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and m.weight is not None:
            P.random_structured(m, name="weight", amount=pct, dim=0)


def prune_perm(net):
    """Permanently prune: for every module with a torchprune mask, replace the
    original weight with the pruned (masked) weight and delete the mask. Required
    because a still-masked model breaks YOLO's Conv-BN fusion (device mismatch on
    the pruned weight vs. the mask) and yields zero detections."""
    for m in list(net.modules()):
        for pname in ("weight", "bias"):
            try:
                P.remove(m, pname)
            except Exception:
                pass


def bench(path, frames, device="cuda:0"):
    m = YOLO(str(path)); m.to(device); m.model.eval()
    rec = {"params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
           "file_mb": round(path.stat().st_size / 1e6, 2)}
    all_confs = []
    for imgsz in (640, 1280):
        for _ in range(WARMUP):
            m(str(frames[0]), imgsz=imgsz, conf=0.05, device=device, verbose=False)
        torch.cuda.synchronize()
        total, n = 0.0, 0
        for f in frames:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = m(str(f), imgsz=imgsz, conf=0.05, device=device, verbose=False)[0]
            torch.cuda.synchronize()
            total += (time.perf_counter() - t0) * 1000.0
            n += 1
            if imgsz == 640:
                confs = [float(c) for c in r.boxes.conf]
                all_confs.extend(confs)
                rec.setdefault("per_frame", {})[f.name] = {
                    "detections": int(len(r.boxes)),
                    "avg_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
                    "max_conf": round(max(confs), 4) if confs else 0.0}
        avg = total / n
        rec[f"latency_ms_{imgsz}"] = round(avg, 2)
        rec[f"fps_{imgsz}"] = round(1000.0 / avg, 2)
    rec["avg_conf"] = round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0
    rec["max_conf"] = round(max(all_confs), 4) if all_confs else 0.0
    del m
    torch.cuda.empty_cache()
    return rec


def export_pruned(net, src, path):
    """Export a pruned nn.Module to a loadable .pt via a manual checkpoint dict.
    YOLO(net) is NOT accepted for raw nn.Module (str() -> filename), so we save a
    proper ckpt and reload it by path (confirmed working, also yields a real file size)."""
    d = {"model": net,
         "train_args": src.model.args if hasattr(src.model, "args") else (src.train_args if hasattr(src, "train_args") else None),
         "train_metrics": getattr(src, "metrics", None),
         "epoch": -1, "name": "pruned", "version": getattr(src.model, "version", "unknown")}
    torch.save(d, str(path))
    return Path(path)


def main():
    frames = sorted(FRAMES_DIR.glob("val_batch*_pred_*.jpg"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = YOLO(str(SRC))
    src.model.eval()
    base_params = sum(p.numel() for p in src.model.parameters())
    base_nnz = nnz_total(src.model)
    print(f"source: params={base_params / 1e6:.2f}M nnz={base_nnz / 1e6:.2f}M")
    results = {
        "source_model": "YOLO11l + 空频融合（最佳 large 模型）",
        "source_params_m": round(base_params / 1e6, 2),
        "source_nnz_m": round(base_nnz / 1e6, 3),
        "frames": [f.name for f in frames],
        "variants": {},
    }
    brec = bench(SRC, frames)
    results["variants"]["unpruned"] = {"sparsity_pct": 0, **brec}
    print(f"baseline params={brec['params_m']}M lat640={brec['latency_ms_640']}ms "
          f"lat1280={brec['latency_ms_1280']}ms fps640={brec['fps_640']} "
          f"avg_conf={brec['avg_conf']} max_conf={brec['max_conf']}")
    for method in METHODS:
        for pct in SPARSITY:
            key = f"{method}_{int(pct * 100)}"
            print(f"== {key} ==", flush=True)
            net = copy.deepcopy(src.model).to("cpu"); net.eval()
            if method == "unstructured_magnitude":
                apply_magnitude(net, pct)
            elif method == "unstructured_l1":
                apply_torchprune_unstructured(net, pct, "l1")
            elif method == "unstructured_random":
                apply_torchprune_unstructured(net, pct, "random")
            else:
                apply_structured(net, pct)
            prune_perm(net)  # bake in masks, strip PruningContainer (fixes BN-fusion device crash)
            nnz = nnz_total(net)
            achieved = round(100.0 * (1 - nnz / base_nnz), 2)
            out_pt = OUT_DIR / f"{key}.pt"
            net.to("cpu")
            export_pruned(net, src, out_pt)
            torch.cuda.empty_cache()
            r = bench(out_pt, frames)
            r["method"] = method
            r["target_sparsity_pct"] = int(pct * 100)
            r["achieved_sparsity_pct"] = achieved
            r["nnz_m"] = round(nnz / 1e6, 3)
            results["variants"][key] = r
            print(f"  achieved={achieved}% nnz={r['nnz_m']}M lat640={r['latency_ms_640']}ms "
                  f"lat1280={r['latency_ms_1280']}ms fps640={r['fps_640']} "
                  f"avg_conf={r['avg_conf']} max_conf={r['max_conf']}")
            del net
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
