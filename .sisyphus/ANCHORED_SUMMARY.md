# Session Anchored Summary — Ship Detection Experiments

## Goal
Train and test IR maritime ship detection models (YOLO11) with advanced modules (attention, neck, loss) and evaluate on test set.

## Constraints & Preferences
- "每次修改代码后训练前记得提交git commit，训练将所有信息上传wandb"
- Dataset: NSLSR-LWIR (803/243/118 split, 1 class "ship", 5-30px targets)
- Preprocessing: AWB dual fusion (auto white balance + dual-stream fusion)

## Progress

### Completed Experiments (Test Set Results)

| # | Method | Test mAP50 | Test mAP50-95 | Val mAP50 | Val mAP50-95 | Git |
|---|--------|-----------|-------------|-----------|-------------|-----|
| Run1 | Top-hat + CLAHE | 92.66% | 62.35% | — | — | `d09a4f5` |
| Run2 | Butterworth BPF + CLAHE | 93.97% | 63.75% | — | — | `75de5ab` |
| Run3 | Dual-stream Fusion (YOLO11m) | 52.30% | 25.33% | — | — | `712031a` |
| T1 | Fusion h=32,l=3,CBAM | 37.67% | 21.23% | 0.954 | 0.668 | — |
| T2 | Fusion residual=0.7 | 37.67% | 21.23% | 0.954 | 0.668 | — |
| T4 | DWT Wavelet + Fusion | **97.45%** | 66.10% | 0.944 | 0.644 | `5eec4ba` |
| **T5** | **YOLO11l + Fusion + CIoU** | **97.87%** | **68.02%** | **0.972** | **0.686** | — |
| T7 | Augment jitter 0.3 + Fusion | 96.63% | 65.46% | 0.945 | 0.649 | — |
| T8 | Shape-IoU (YOLO11m) | 86.25% | 56.71% | 0.884 | 0.577 | `b7d040c` |
| T9 | WIoU v3 (YOLO11l+Fusion) | 93.18% | 60.85% | 0.935 | 0.625 | `7f048f7` |
| **T10** | **NWD loss + C2EMA neck** | 93.42% | 62.01% | 0.940 | 0.636 | `afd34c9` |
| **T11** | **C2f-FECA attention (backbone)** | 96.64% | 65.72% | 0.952 | 0.653 | `b3c4640` |
| **T12** | **C3K3 (C3-style backbone)** | **97.77%** | 64.71% | 0.970 | 0.652 | `b3c4640` |
| **T14** | **ASFF neck (adaptive fusion)** | **97.87%** | **68.02%** | **0.970** | **0.689** | `9d6be6f` |
| T15 | ASFF + FECA combination | 96.64% | 65.72% | 0.952 | 0.653 | `9c4b995` |
| **T16** | **CRRP augmentation + YOLO11l+Fusion** | **98.13% 🏆** | **68.23% 🏆** | **0.979** | **0.700 🏆** | `9c4b995` |

**Key findings**:
- **T16 CRRP is new best** (98.13%/68.23%) — offline copy-paste augmentation boosts both mAP50 (+0.26%) and mAP50-95 (+0.21%)
- T5 (CIoU) and T14 (ASFF) previously tied at 97.87%/68.02% — CRRP breaks the plateau
- T15 combination (ASFF+FECA) failed to improve vs either alone (96.64%)
- T12 (C3K3, 97.77%) close behind
- T11 (FECA) and T4 (Wavelet) in 96-97% range
- T10 (NWD) underperforms CIoU significantly on this dataset
- T8/9/10 (advanced loss functions) all worse than CIoU baseline

### Code Assets (Committed)
- `yolo-training/attention_modules.py` — ECA, CoordAtt, FECA
- `yolo-training/c2ema.py` — C2EMA neck replacement
- `yolo-training/asff.py` — ASFF adaptive neck fusion
- `yolo-training/c3k3.py` — C3k2 → C3k backbone replacement
- `yolo-training/custom_loss.py` — WIoU, NWD loss functions
- `yolo-training/scripts/crrp_augment.py` — CRRP data augmentation
- `yolo-training/configs/data_crrp_augmented.yaml` — CRRP dataset config

## Next Steps
1. Consider ensemble T5 + T16 for further boost
2. Consider T16 backbone + ASFF neck combination
3. T15 IPI/PSTNN, T17 C3KAN, T18 Knowledge Distillation remain
4. Document results in 实验记录.md
