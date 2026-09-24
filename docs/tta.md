# Test-Time Augmentation (TTA) for Object Detection

## Literature Overview

### 1. Better Aggregation in Test-Time Augmentation

| Field | Detail |
|-------|--------|
| **Authors** | Divya Shanmugam, Davis Blalock, Guha Balakrishnan, John Guttag (MIT CSAIL, Rice University) |
| **Venue** | IEEE/CVF International Conference on Computer Vision (ICCV) 2021 |
| **DOI / arXiv** | [arXiv:2201.11472](https://arxiv.org/abs/2201.11472) (extended) / ICCV 2021 |
| **Date** | October 2021 |

**Abstract:** Test-time augmentation (TTA) — the aggregation of predictions across transformed versions of a test input — is a common practice in image classification. This paper presents experimental analyses that shed light on cases where simple averaging is suboptimal. A key finding: even when TTA produces a net improvement in accuracy, it can change many correct predictions into incorrect predictions. The authors propose a learning-based method for aggregating TTA predictions.

**Core Methodology:**
- Analyzes per-class and per-sample effects of TTA, revealing that softmax averaging corrupts many originally correct predictions
- Proposes **AugTTA**: a lightweight learned aggregation that assigns per-augmentation weights via a small prediction network
- Shows that standard TTA (simple average of softmax outputs) is suboptimal because different augmentations contribute unequally across classes
- Experiments on ImageNet, Flowers-102, CIFAR-10/100 across ResNet, DenseNet, VGG architectures

---

### 2. Understanding Test-Time Augmentation

| Field | Detail |
|-------|--------|
| **Authors** | Masanari Kimura |
| **Venue** | ICONIP 2021 (28th International Conference on Neural Information Processing) |
| **DOI** | [10.1007/978-3-030-92185-9_46](https://doi.org/10.1007/978-3-030-92185-9_46) |
| **Date** | December 2021 |

**Abstract:** Provides the first theoretical guarantees for TTA. Proves that the expected error of TTA is less than or equal to the average error of the original model, and under certain assumptions, strictly less.

**Core Methodology:**
- Formalizes TTA as an ensemble over augmented views
- Proves error reduction depends on the **ambiguity term** (diversity of predictions across augmentations)
- Derives closed-form optimal weights for generalized TTA

---

### 3. YOLOv5 TTA Implementation (Reference Code)

| Field | Detail |
|-------|--------|
| **Repository** | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| **Source** | [models/yolo.py, lines 125-137](https://github.com/ultralytics/yolov5/blob/8c6f9e15bfc0000d18b976a95b9d7c17d407ec91/models/yolo.py#L125-L137) |
| **Docs** | [Ultralytics TTA Tutorial](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) |
| **Date** | November 2023 (tutorial); code stable since v5.0 |

**Core Methodology (forward_augment):**
Three scales: 1.0x, 0.83x, 0.67x  |  Flips: none, left-right, none

- **Multi-scale inference**: Runs the detector on 3 scaled versions of the input (1.0x, 0.83x, 0.67x)
- **Horizontal flip**: The middle scale (0.83x) additionally applies a left-right flip (flip code 3 = lr)
- **De-scaling**: Predictions are transformed back to original coordinates via _descale_pred() — inversely scaling box coordinates and de-flipping if needed
- **NMS fusion**: All predictions from the 3 passes are concatenated and fed through standard NMS to produce the final output
- **Inference cost**: ~2-3x slower than single-pass inference

---

### 4. Why TTA Can Hurt Small Object Detection

Small objects are disproportionately affected by TTA:

| Factor | Mechanism | Impact on Small Objects |
|--------|-----------|------------------------|
| **Downscaling** | At 0.67x/0.83x scales, tiny objects shrink below the feature grid resolution | The object's signal is lost entirely; the model outputs noise instead of a detection |
| **Confidence averaging** | NMS fuses predictions by concatenating scores from all scales | Low-confidence or false detections from downscaled passes dilute the correct high-confidence prediction from the original scale |
| **Coordinate rounding** | De-scaling maps bbox coordinates back (x /= scale) after integer rounding | Sub-pixel misalignment is proportionally larger for small boxes (e.g., 1 px error on a 6 px box = 17% IoU loss) |
| **Anchor mismatch** | Predefined anchor boxes are designed for the training scale; downscaling shifts feature distributions | Small-object anchors may no longer align with the actual object at reduced scales |

**Empirical evidence** (YOLOv5x on COCO val2017):
- AP@small improved from 0.351 to 0.361 (+1.0%) when increasing img size from 640 to 832 AND enabling TTA
- However, the img size increase (30% upscale) itself disproportionately helps small objects by enlarging them
- Using TTA **without** increasing input resolution likely shows smaller or negative gains for small objects

**Key takeaway**: TTA is a net-positive technique, but its benefits for small objects are fragile and depend on careful scale selection. Overly aggressive downscaling within TTA can degrade small-object performance.

---

## References

1. Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021). *Better Aggregation in Test-Time Augmentation*. ICCV 2021. [arXiv:2201.11472](https://arxiv.org/abs/2201.11472)
2. Kimura, M. (2021). *Understanding Test-Time Augmentation*. ICONIP 2021. [DOI: 10.1007/978-3-030-92185-9_46](https://doi.org/10.1007/978-3-030-92185-9_46)
3. Ultralytics. *Test-Time Augmentation (TTA) — YOLOv5 Documentation*. [docs.ultralytics.com](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation)
4. Jocher, G. et al. *ultralytics/yolov5*. GitHub. [models/yolo.py#L125-L137](https://github.com/ultralytics/yolov5/blob/8c6f9e15bfc0000d18b976a95b9d7c17d407ec91/models/yolo.py#L125-L137)
