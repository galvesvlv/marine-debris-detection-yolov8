# Imports
from pathlib import Path
import torch
import os

# Global Paths
ROOT_DIR  = Path(__file__).resolve().parents[1]
# Local Paths
CONFIG = ROOT_DIR / "config"
DATA = ROOT_DIR / "data"
INFERENCE = ROOT_DIR / "inference"
PREPROCESSING = ROOT_DIR / "preprocessing"
TRAIN_MODELS = ROOT_DIR / "train_models"
WEIGHTS_YOLOV8 = ROOT_DIR / "weights_yolov8"

# Yolov8 paths
DATASET_DIR_YOLO = DATA / "dataset_marinedebris_yolov8"
DATASET_YAML = DATASET_DIR_YOLO / "data.yaml"

# Models
MODEL_NAME_YOLO = "yolov8n.pt"
MODEL_NAME_YOLO_FINAL1 = WEIGHTS_YOLOV8 / "yolov8n_marinedebris_best_final_1.pt"
WEIGHTS_YOLOV8_BASELINE = WEIGHTS_YOLOV8 / "yolov8n_marinedebris_baseline.pt"
WEIGHTS_YOLOV8_BEST = WEIGHTS_YOLOV8 / "yolov8n_marinedebris_best_final.pt"

# Final Model Names
MODEL_NAME_YOLO_FINAL_TUNNED = "yolov8n_marinedebris_best_final_2.pt"
MODEL_NAME_YOLO_FINAL_PARAMS = WEIGHTS_YOLOV8 / "best_params_used_final_model_2.csv"

MODEL_NAME_YOLO_FINAL_BASELINE_TUNNED = "yolov8n_marinedebris_best_baseline_tunned.pt"  # BEST MODEL

MODEL_NAME_YOLO_FINAL_BASELINE_PARAMS = WEIGHTS_YOLOV8 / "best_params_used_baseline_tunned.csv"

# Optmizer
OPTIMIZER_RESULTS = WEIGHTS_YOLOV8 / "optuna_results.csv"

# Config Device
DEVICE = (
          "cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu"
          )

print(f"GPU is available? {torch.cuda.is_available()}")

print(f"Using {DEVICE} device")

# InferenceTest Prediction
TEST_IMAGE1 = INFERENCE / "test_image_marinedebris1.png"
TEST_IMAGE2 = INFERENCE / "test_image_marinedebris2.jpg"
TEST_VIDEO = INFERENCE / "marine-debris-polution.mp4"