# AGENTS.md — Ship IR Detection (YOLO)

## Quick start

```bash
# Install (uv workspace, runs from repo root)
uv sync

# Run from yolo-training/ directory
cd yolo-training

# Verify label format (fast, no data needed)
uv run pytest tests/test_labels.py -v

# Smoke test training (1 epoch, tiny model, ~2 min)
uv run python train.py --model yolo11n.pt --epochs 1 --batch 4

# Full training (YOLO11m, 200 epochs, imgsz=1280)
uv run python train.py

# Inference (SAHI, recommended for small ships)
uv run python predict.py --mode sahi --source ./test_images/
```

## Project layout

```
ship/
├── pyproject.toml              # Root workspace — PyTorch cu128 index only
├── yolo-training/              # Main package (uv workspace member)
│   ├── train.py                # Training entry point — ALL flags here
│   ├── predict.py              # Inference (standard + SAHI)
│   ├── preprocessing_module.py # 1700+ lines, sea clutter + enhancement
│   ├── fusion_module.py        # Learnable spatial-frequency fusion
│   ├── attention_modules.py    # ECA, CA, FECA backbone insertion
│   ├── custom_loss.py          # NWD, WIoU v3, Shape-IoU losses
│   ├── c2ema.py / asff.py / c3k3.py / dynamic_head.py  # Neck/head swaps
│   ├── configs/                # data.yaml, train.yaml, YOLO P2 YAML
│   ├── scripts/                # Dataset extraction, label gen, preprocessing
│   ├── tests/                  # pytest tests
│   └── run_t{1,2,3}.bat       # Windows batch for ablation sweeps
├── docs/                       # 39 research papers on techniques used
└── 红外波段海面舰艇识别系统核心方案_v3.md  # 2800-line domain spec (Chinese)
```

## Key commands

| Task | Command |
|------|---------|
| Install deps | `uv sync` (from repo root) |
| Run label tests | `uv run pytest tests/test_labels.py -v` |
| Run dataset tests | `uv run pytest tests/test_dataset.py -v` (requires `data/processed/`) |
| Smoke train | `uv run python train.py --model yolo11n.pt --epochs 1 --batch 4` |
| Full train | `uv run python train.py` |
| SAHI infer | `uv run python predict.py --mode sahi --source ./images/` |
| Extract data | `uv run python scripts/extract_datasets.py` |
| Generate pseudo-labels | `uv run python scripts/generate_labels.py` |
| Prepare YOLO data | `uv run python scripts/prepare_yolo_data.py` |

## Non-obvious facts

- **Run everything from `yolo-training/`**. The root workspace only manages the cu128 PyTorch index. All scripts, configs, and entry points live in `yolo-training/`.
- **Python 3.13+ required**. `requires-python = ">=3.13"` in both `pyproject.toml` files.
- **PyTorch from cu128 index**. Root `pyproject.toml` pins torch/torchvision to `https://download.pytorch.org/whl/cu128` (RTX 5080 / Blackwell). This must be the workspace root index.
- **No linter, formatter, or type checker configured**. Code style is whatever the author writes. Do not introduce `ruff`, `black`, or `mypy` without asking.
- **Windows-only tooling for sweeps**. All batch scripts (`run_t{1,2,3}.bat`) use Windows syntax. There are no equivalent shell scripts.
- **`configs/data.yaml` has a hardcoded Windows path** (`D:/Projects/...`). The `train.py` data path resolver tries `PROJECT_ROOT / path` as a fallback, but agents editing this file should update it.
- **Pseudo-labels are estimates**, not ground truth. Ships are always centered `(0.5, 0.5)`. Width estimated from viewing angle directory names. Bounding boxes are approximate.
- **Preprocessing cache is per-method**. `--preprocess` creates `data/preprocessed_{method}/` — one directory per method. Labels are copied, only images are transformed.
- **`--use-fusion` requires dual-stream preprocessing**. Only `awb_dual_fusion`, `awb_wavelet_dual_fusion`, or `awb_rpca_dual_fusion` produce the 3-channel input the fusion module expects.
- **SAHI is recommended for inference**. Ships are 5-30px tall in 1024x512 images. Standard YOLO inference at 1280px misses many targets. SAHI slices at 512x512 with 20% overlap.
- **Dataset presets** in `train.py`: `ship` (9 classes, ~60K synthetic), `nslsr-swir` (1 class), `nslsr-lwir` (1 class). Use `--dataset` flag.
- **Loss replacement is monkey-patching**. `--iou-loss shape-iou`, `wiou`, or `nwd` patches `v8DetectionLoss.__init__` at runtime.
- **No `main` entry point at root**. The root `pyproject.toml` has no scripts. All CLI entry points are in `yolo-training/pyproject.toml` (`train` → `train:main`, `predict` → `predict:main`), but the project uses `uv run python train.py` directly.
- **Results go to `runs/detect/train/`** in `yolo-training/`, not the repo root.
- **Domain documentation is in Chinese**. `红外波段海面舰艇识别系统核心方案_v3.md` is the main 2800-line technical spec. `docs/` has 39 short papers on each technique (CLAHE, SAHI, WIoU, etc.).

## Testing

- `pytest` testpaths = `tests/` (configured in `yolo-training/pyproject.toml`).
- `test_labels.py` — label format verification. Runs without extracted data.
- `test_dataset.py` — dataset integrity. Requires `data/processed/` to exist.
- `test_nslsr_data.py` — NSLSR dataset validation.
- `test_shape_iou_loss.py` — Shape-IoU loss unit tests.

## Architecture notes

- **train.py** is the single orchestrator. It imports preprocessing, fusion, attention, neck, head, and loss modules and wires them into Ultralytics' YOLO trainer via callbacks and monkey-patching.
- **Model modifications are additive** — attention insertion, neck replacement, head replacement, and fusion can all be combined in one run.
- **Augment jitter** (`--augment-jitter`) randomizes augmentation params each epoch. **Preprocess jitter** (`--preprocess-jitter`) randomizes preprocessing params per image during offline caching.
- **W&B** integrates via Ultralytics' built-in `wandb=True`. Custom callbacks log preprocessing samples, fusion stats, and gradient norms.
