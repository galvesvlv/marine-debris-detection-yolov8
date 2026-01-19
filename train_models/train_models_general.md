# Model Training and Optimization

This directory contains all scripts related to training, hyperparameter optimization, and fine-tuning of the YOLOv8 models used for this marine debris detection project.

The files in this directory are organized to clearly separate baseline training, Optuna-based optimization, and fine-tuning experiments. Detailed discussions about parameter choices and evaluation results are provided in dedicated Markdown files.

---

## 📂 Directory Structure

```text
train_models/
 ├── src/
 │   ├── optimizer.py
 │   └── yolov8.py
 │
 ├── tuning/
 │   └── train_tuning.py
 │
 ├── train_baseline.py
 ├── train_bestoptuna.py
 └── train_finetuning_baseline.py
```

---
## 🧩 Source Modules (`src/`)

### `src/yolov8.py`

Provides a lightweight wrapper around the Ultralytics YOLOv8 API.

This module centralizes common model operations such as:

- Model initialization
- Training
- Evaluation
- Saving best-performing weights

It abstracts repetitive YOLOv8 calls and helps keep training scripts consistent and easier to maintain.

---

### `src/optimizer.py`

Defines utilities for hyperparameter optimization using Optuna.

This module includes:

- A reproducibility helper (`set_seed`)
- The Optuna objective function (`objective`) that trains and evaluates YOLOv8 models using sampled hyperparameters
- A helper function (`load_best_params`) to retrieve the best hyperparameter set from the exported Optuna CSV results

This file is used by the tuning pipeline and by scripts that retrain models using the best Optuna parameters.

---

## 🔍 Hyperparameter Optimization (`tuning/`)

### `tuning/train_tuning.py`

Entry-point script for running Optuna-based hyperparameter optimization.

This script:

- Creates an Optuna study configured to maximize validation performance
- Runs multiple trials, where each trial trains a YOLOv8 model using sampled hyperparameters
- Prints the best trial value and parameters to the console
- Exports all trial results to a CSV file (used later for analysis and final training)
    - https://drive.google.com/file/d/1xQYyfBiTHTl7RjTTMXiWmbYblV4YXXQ1/view?usp=sharing

---

## 🚀 Training Scripts

### `train_baseline.py`

Runs baseline training of a YOLOv8 model using a fixed configuration.

This script:

- Trains a YOLOv8 model starting from pretrained weights
- Evaluates performance on the test split
- Saves the best-performing weights as the baseline reference model

---

### `train_bestoptuna.py`

Trains a YOLOv8 model using the best hyperparameters obtained from Optuna optimization.

This script:

- Loads the best hyperparameters from the Optuna results CSV
- Trains a new YOLOv8 model using these parameters
- Evaluates performance on the test split
- Saves the trained model weights and exports the selected hyperparameters used

---

### `train_finetuning_baseline.py`

Fine-tunes the previously trained baseline model to improve performance.

This script:

- Loads the baseline YOLOv8 model weights
- Performs additional training (fine-tuning) with a reduced learning rate
- Evaluates performance on the test split
- Generates a confusion matrix visualization
- Saves the fine-tuned model weights used in the inference pipeline

---

## 🧪 Notes

- All scripts rely on shared configuration defined in the global `config` module (e.g., dataset path, device selection, weight paths).
- Training and evaluation metrics are printed to the console and stored by Ultralytics under the configured `runs/` directories.
- Detailed parameter decisions and performance comparisons are documented in separate Markdown files within this directory.
