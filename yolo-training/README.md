# Ship Identification — YOLO Training Project

Infrared ship detection and classification using YOLO11, trained on synthetic
rendered data with pseudo-labels and validated on real IR imagery.

## Dataset

Two datasets are included (both in `../Ship/`):

| Dataset | Type | Count | Classes | Format |
|---------|------|-------|---------|--------|
| **nor_irship** | Synthetic renders | ~60,700 PNG | 9 ship types | 1024×512 grayscale |
| **sanya_ir** | Real IR captures | ~3,500 JPG | 4 letter codes | Various sizes |

### Ship Classes (9)

| ID | Class | Description |
|----|-------|-------------|
| 0 | Ada | Corvette |
| 1 | Akizuki | Destroyer |
| 2 | Alvaro De Bazan | Frigate |
| 3 | Armourique | Frigate |
| 4 | Independence | Littoral combat ship |
| 5 | Jiangkai II | Frigate (Chinese) |
| 6 | Oliver Hazard Perry | Frigate |
| 7 | Sejong Daewang | Destroyer (Korean) |
| 8 | Zumwalt | Destroyer |

## Setup

```powershell
# Install dependencies
uv sync

# Activate environment (optional — uv run handles this)
.venv\Scripts\activate
```

## Pipeline

### 1. Extract datasets

```powershell
# Extract all datasets (requires 7-Zip, ~12 GB free space)
uv run python scripts/extract_datasets.py

# Extract only one dataset
uv run python scripts/extract_datasets.py --dataset nor_irship
uv run python scripts/extract_datasets.py --dataset sanya
```

The synthetic dataset takes 5-30 minutes to extract depending on disk speed.

### 2. Analyze dataset

```powershell
# Generate statistics about the extracted data
uv run python scripts/analyze_dataset.py
```

### 3. Generate pseudo-labels

The synthetic dataset has class labels (from directory names) but no bounding
boxes. This script generates approximate YOLO-format labels:

- **Center**: The ship is always centered in the image → `(0.5, 0.5)`
- **Height**: Known from directory pixel-height range (e.g., `10-13` → 11.5px)
- **Width**: Estimated from viewing angle:
  - `0-29°` (broadside): width = height × 4.0
  - `30-59°` (angled):   width = height × 3.0
  - `60-89°` (head-on):  width = height × 2.0

```powershell
# Generate labels alongside images
uv run python scripts/generate_labels.py

# Dry-run to preview without writing
uv run python scripts/generate_labels.py --dry-run
```

### 4. Prepare YOLO-format data

Creates the `data/processed/` directory with train/val/test split (80/10/10),
using NTFS hardlinks to save disk space.

```powershell
uv run python scripts/prepare_yolo_data.py
```

### 5. Train

```powershell
# Full training (YOLO11m, 200 epochs)
uv run python train.py

# Quick test (tiny model, 1 epoch)
uv run python train.py --model yolo11n.pt --epochs 1 --batch 4

# Custom config
uv run python train.py --model yolo11s.pt --epochs 100 --batch 16 --imgsz 640
```

### 6. Test

```powershell
# Run all label generation tests
uv run pytest tests/ -v

# Run dataset integrity tests (after prepare_yolo_data.py)
uv run pytest tests/test_dataset.py -v
```

### 7. Inference

```powershell
# Standard YOLO inference
uv run python predict.py --mode standard --source image.jpg

# SAHI tiled inference (recommended for small ships)
uv run python predict.py --mode sahi --source ./test_images/
```

## Pseudo-Label Limitations

The bounding boxes are **estimates**, not ground truth. This approach works
because:

1. Consistent pseudo-labels teach the model **where** to look and **what** to
   classify
2. YOLO's built-in augmentation (mosaic, mixup, copy-paste) regularizes against
   imperfect boxes
3. The model learns to refine box predictions during training

**Expected outcome**: The model will learn reliable classification and reasonable
localization. For production use, fine-tune on a small set of manually annotated
real images.

## Small Object Detection Strategy

Ships in this dataset are very small (5-30 pixels in a 512-pixel-high image,
just 1-6% of image height). The training config includes:

- **YOLO11m** with C2PSA attention for small object features
- **Large input size** (imgsz=1280) to increase the ship footprint
- **SAHI** tiled inference (512×512 tiles, 20% overlap) at test time
- **Aggressive augmentation** (mosaic, mixup, copy-paste) for robustness

## Project Structure

```
yolo-training/
├── .venv/                    # Virtual environment (uv-managed)
├── configs/
│   ├── data.yaml             # Dataset configuration
│   └── train.yaml            # Training hyperparameters
├── data/
│   ├── raw/                  # Extracted archives (gitignored)
│   │   ├── nor_irship/       # ~60K synthetic images
│   │   └── sanya_ir/         # ~3.5K real IR images
│   ├── processed/            # YOLO-format data (gitignored)
│   │   ├── train/images/     # 80% training images
│   │   ├── train/labels/     # Training labels
│   │   ├── val/images/       # 10% validation images
│   │   ├── val/labels/       # Validation labels
│   │   ├── test/images/      # 10% test + Sanya IR
│   │   └── test/labels/      # Test labels (synthetic only)
│   └── splits.json           # Split metadata
├── scripts/
│   ├── extract_datasets.py   # Archive extraction
│   ├── analyze_dataset.py    # Dataset statistics
│   ├── generate_labels.py    # Pseudo-label generation
│   └── prepare_yolo_data.py  # YOLO dataset builder
├── tests/
│   ├── test_labels.py        # Label format verification
│   └── test_dataset.py       # Dataset integrity checks
├── train.py                  # Training entry point
├── predict.py                # Inference (standard + SAHI)
├── pyproject.toml            # Project config + dependencies
└── README.md
```
