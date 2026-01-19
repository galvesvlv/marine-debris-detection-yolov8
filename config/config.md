# Configuration Module

This directory contains global configuration settings used across the project, including path definitions, model references, dataset locations, and device selection logic.

All paths are defined relative to the project root to ensure portability and reproducibility across different environments.

---

## 📁 Path Management

The configuration module centralizes all directory and file paths, such as:

- Project root directory
- Dataset and annotation files (YOLOv8 format)
- Model weight files
- Training, preprocessing, and inference directories

This approach avoids hard-coded paths and simplifies maintenance.

---

## 🤖 Model Configuration

The module defines references to:

- Baseline YOLOv8 model weights
- Optimized and tuned model versions
- Final model selected for inference
- Hyperparameter files used during training and optimization

These references allow consistent model loading across training and inference pipelines.

---

## ⚙️ Device Selection

The configuration automatically selects the best available compute device:

- `cuda` if an NVIDIA GPU is available
- `cpu`
