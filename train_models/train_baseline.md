# Baseline Training – YOLOv8

This document describes the baseline training strategy used for the YOLOv8 marine debris detection model.

The baseline does **not** correspond to a fully default YOLOv8 configuration.  
Instead, it reflects an informed baseline obtained after an initial round of experimentation and analysis, where specific hyperparameters were adjusted to better suit the dataset characteristics that were first analised.

This baseline serves as the reference model for subsequent optimization and fine-tuning experiments.

---

## 🎯 Objective

- Establish a strong and stable baseline model
- Evaluate model performance on the test split
- Provide a reference for comparison with Optuna-based optimization and fine-tuning

---

## 🧠 Training Script

**File:** `train_baseline.py`

The training pipeline follows three main steps:

1. Initialize a YOLOv8 model from pretrained weights
2. Train using a fixed, manually tuned configuration
3. Evaluate performance on the test dataset
4. Save the best-performing model weights

---

## ⚙️ Training Configuration

### Core Parameters

| Parameter | Value | Description |
|---------|-------|-------------|
| `imgsz` | 640 | Input image resolution |
| `batch` | 32 | Number of images per batch |
| `epochs` | 30 | Maximum number of training epochs |
| `patience` | 5 | Early stopping patience |
| `freeze` | 8 | Number of backbone layers frozen |

Freezing early backbone layers helps preserve pretrained visual features while adapting higher-level representations to marine debris detection.

---

### Optimizer and Learning Rate Strategy

| Parameter | Value | Description |
|--------|-------|-------------|
| `optimizer` | AdamW | Optimizer with decoupled weight decay |
| `lr0` | 0.003 | Initial learning rate |
| `lrf` | 0.01 | Final learning rate multiplier |
| `weight_decay` | 0.0005 | L2 regularization strength |
| `momentum` | 0.937 | Momentum term for optimization (default) |
| `warmup_epochs` | 5 | Learning rate warm-up duration |
| `warmup_bias_lr` | 0.1 | Warm-up learning rate for bias parameters |

---

### Detection-Specific Parameters

| Parameter | Value | Description |
|---------|-------|-------------|
| `iou` | 0.5 | IoU threshold for positive detections |
| `box` | 10.0 | Bounding box regression loss weight |
| `cls` | 0.8 | Classification loss weight |
| `dfl` | 2.0 | Distribution focal loss weight |

A reduced IoU threshold was adopted to mitigate excessive bounding box overlap in cluttered scenes. The bounding box loss weight (`box`) was increased relative to the default configuration to improve localization accuracy, which is particularly important for detecting small objects. The classification loss weight (`cls`) was also increased due to the presence of visually similar classes, aiming to enhance class discrimination. Finally, the distribution focal loss (`dfl`) weight was increased not only to improve fine-grained bounding box localization for small objects, but also to amplify the contribution of harder and less frequent examples during training. This adjustment helps mitigate the effects of class imbalance by ensuring that poorly localized instances, which are more common in minority classes, have a stronger influence on the optimization process.


---

## 📊 Validation Performance

**Validation split (115 images, 1435 instances)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|------|----------|--------|--------|-----------|
| all | 0.746 | 0.685 | 0.740 | 0.447 |
| can | 0.848 | 0.750 | 0.838 | 0.495 |
| foam | 0.897 | 0.768 | 0.858 | 0.597 |
| plastic | 0.618 | 0.684 | 0.652 | 0.389 |
| plastic bottle | 0.761 | 0.731 | 0.793 | 0.465 |
| unknow | 0.607 | 0.491 | 0.557 | 0.292 |

---

## 🧪 Test Performance

**Test split (60 images, 630 instances)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|------|----------|--------|--------|-----------|
| all | 0.745 | 0.668 | 0.744 | 0.445 |
| can | 0.884 | 0.717 | 0.863 | 0.503 |
| foam | 0.867 | 0.652 | 0.792 | 0.542 |
| plastic | 0.591 | 0.641 | 0.658 | 0.380 |
| plastic bottle | 0.756 | 0.778 | 0.817 | 0.497 |
| unknow | 0.629 | 0.554 | 0.590 | 0.305 |

**Overall test metrics:**

| Metric | Value |
|------|-------|
| mAP@50–95 | 0.445 |
| mAP@50 | 0.744 |
| mAP@75 | 0.484 |

---

## ⚡ Computational Performance

- **Model size:** 3.0M parameters
- **GFLOPs:** 8.1
- **Inference time:** ~7.8 ms per image (Tesla T4)
- **Total training wall time:** ~13 minutes

---

## 💾 Output Artifact

- **Saved model:** `yolov8n_marinedebris_baseline.pt`

This model is used as the reference baseline for all subsequent optimization and fine-tuning experiments.

---

## 🧪 Notes

- This baseline was refined beyond YOLOv8 defaults to better match dataset characteristics.
- Detailed comparisons with Optuna-based optimization and fine-tuning are documented in separate Markdown files.
