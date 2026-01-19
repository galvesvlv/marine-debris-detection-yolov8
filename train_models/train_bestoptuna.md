# Hyperparameter Optimization – YOLOv8 (Optuna)

This document describes the hyperparameter optimization strategy based on Optuna used to train YOLOv8 models for marine debris detection.

The objective of this step was to explore the hyperparameter space systematically and evaluate whether automated tuning could outperform the manually adjusted baseline configuration.

---

## 🎯 Objective

- Perform automated hyperparameter search using Optuna
- Identify promising configurations for YOLOv8 training
- Compare Optuna-based results against the baseline model
- Assess the impact of computational constraints on optimization quality

---

## 🧠 Optimization Pipeline

The Optuna optimization pipeline is composed of three main components:

1. Definition of the search space and objective function
2. Execution of multiple short training trials
3. Final training using the best hyperparameters obtained

---

## ⚙️ Optuna Objective Function

**File:** `src/optimizer.py`

The Optuna objective function:

- Samples hyperparameters from predefined ranges
- Trains a YOLOv8 model for a limited number of epochs
- Evaluates performance on the validation split
- Returns validation mAP@50–95 as the optimization objective

### Search Space

| Hyperparameter | Range |
|---------------|-------|
| `lr0` | 1e-4 – 1e-2 (log scale) |
| `lrf` | 0.01 – 0.2 |
| `weight_decay` | 1e-5 – 1e-3 (log scale) |
| `box` | 7.0 – 10.0 |
| `cls` | 0.4 – 0.9 |
| `dfl` | 1.4 – 2.0 |
| `iou` | 0.4 – 0.7 |

---

## ⚠️ Computational Constraints

Due to GPU limitations in the free-tier Google Colab environment, each Optuna trial was trained for:

- **Epochs:** 8  
- **Patience:** 2  

This constraint was necessary to allow multiple trials (20) to be executed within the available runtime.

As a consequence, the optimization process prioritizes **early learning dynamics**, rather than full convergence behavior.

---

## 🔍 Hyperparameter Optimization Execution

**File:** `tuning/train_tuning.py`

- Number of trials: **20**
- Optimization metric: **Validation mAP@50–95**
- Output: CSV file containing all trial results

The optimization results were exported and later used to retrain a final model using the best-performing parameter set.

---

## 🚀 Final Training with Best Optuna Parameters

**File:** `train_bestoptuna.py`

The best hyperparameters identified by Optuna were used to train a final YOLOv8 model with an extended training schedule.
- Best parameters in: https://drive.google.com/file/d/1xQYyfBiTHTl7RjTTMXiWmbYblV4YXXQ1/view?usp=drive_link

### Training Setup

- Input resolution: 640 × 640
- Batch size: 32
- Maximum epochs: 50
- Early stopping patience: 5
- Frozen backbone layers: 8
- Optimizer: AdamW

---

## 📊 Validation Performance

**Validation split (115 images, 1435 instances)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|------|----------|--------|--------|-----------|
| all | 0.661 | 0.681 | 0.696 | 0.407 |
| can | 0.880 | 0.587 | 0.781 | 0.466 |
| foam | 0.747 | 0.788 | 0.820 | 0.538 |
| plastic | 0.566 | 0.707 | 0.640 | 0.357 |
| plastic bottle | 0.678 | 0.738 | 0.758 | 0.428 |
| unknow | 0.433 | 0.582 | 0.483 | 0.245 |

---

## 🧪 Test Performance

**Test split (60 images, 630 instances)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|------|----------|--------|--------|-----------|
| all | 0.750 | 0.556 | 0.696 | 0.405 |
| can | 0.815 | 0.415 | 0.746 | 0.411 |
| foam | 0.888 | 0.652 | 0.771 | 0.528 |
| plastic | 0.716 | 0.587 | 0.666 | 0.370 |
| plastic bottle | 0.769 | 0.692 | 0.780 | 0.458 |
| unknow | 0.561 | 0.432 | 0.518 | 0.259 |

**Overall test metrics:**

| Metric | Value |
|------|-------|
| mAP@50–95 | 0.405 |
| mAP@50 | 0.696 |
| mAP@75 | 0.435 |

---

## ⚠️ Early Stopping Behavior

During the final Optuna-based training, early stopping was triggered at **22 epochs**, unlike the baseline configuration.

This behavior suggests **premature convergence**, potentially caused by:

- Suboptimal hyperparameter combinations
- Limited exploration of the loss landscape during short Optuna trials
- Overemphasis on early validation performance due to reduced training duration per trial

This effect may indicate convergence toward a local or shallow minimum rather than a globally optimal configuration.

---

## 🔍 Comparison with Baseline

- The Optuna-optimized model exhibits **similar qualitative behavior** to the baseline.
- Quantitative metrics remain consistently **below the baseline model**.
- The constrained number of epochs per trial likely limited Optuna’s ability to identify truly superior configurations.

---

## 🧪 Notes

- Hyperparameter optimization was constrained by GPU availability.
- Results should be interpreted as indicative rather than definitive.
- Further optimization with longer trials and more epochs could potentially improve Optuna-based performance.
