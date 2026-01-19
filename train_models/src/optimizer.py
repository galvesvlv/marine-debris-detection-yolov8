# Imports
import torch
import random
import pandas as pd
import numpy as np
import gc

from src.yolov8 import ModelYoloV8
from config.config import (
                           MODEL_NAME_YOLO,
                           DATASET_YAML,
                           DEVICE
                           )

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    This function fixes the random state for Python's built-in random module,
    NumPy, and PyTorch to ensure deterministic behavior across runs.

    Parameters
    ----------
    seed : int, optional
        Random seed value used to initialize all random number generators.
        Defaults to 42.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def objective(trial):
    """
    Optuna objective function for YOLOv8 hyperparameter optimization.

    This function defines the search space, trains a YOLOv8 model using
    the sampled hyperparameters, evaluates it on the validation split,
    and returns the optimization metric.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object used to sample hyperparameters.

    Returns
    -------
    float
        Validation mAP (IoU 0.50:0.95) used as the optimization objective.
    """

    set_seed()

    params = {
              "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
              "lrf": trial.suggest_float("lrf", 0.01, 0.2),
              "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
              "box": trial.suggest_float("box", 7., 10.0),
              "cls": trial.suggest_float("cls", 0.4, 0.9),
              "dfl": trial.suggest_float("dfl", 1.4, 2.0),
              "iou": trial.suggest_float("iou", 0.4, 0.7),
              }

    model = ModelYoloV8(MODEL_NAME_YOLO)

    model.fit(
              data=DATASET_YAML,
              device=DEVICE,
              name=f"optuna_trial_{trial.number}",
              project="runs/optuna",
              imgsz=640,

              epochs=8,
              patience=2,
              batch=32,

              freeze=8,

              optimizer="AdamW",
              warmup_epochs=2,
              warmup_bias_lr=0.1,
              momentum = 0.937,

              verbose=False,
              **params
              )

    model.evaluate(
                   data=DATASET_YAML,
                   device=DEVICE,
                   split="val",
                   )

    value = model.metrics.box.map  # type: ignore

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return value


def load_best_params(csv_path):
    """
    Load the best hyperparameters from an Optuna trials CSV file.

    This function filters completed trials, selects the one with the
    highest objective value, and extracts the corresponding parameters.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file exported by Optuna containing trial results.

    Returns
    -------
    dict
        Dictionary containing the best hyperparameters, ready to be passed
        to a YOLOv8 training configuration.
    """
    
    df = pd.read_csv(csv_path)

    # Best trial
    df = df[df["state"] == "COMPLETE"]
    best_row = df.sort_values("value", ascending=False).iloc[0]

    # Best Params
    best_params = {
                   "lr0":          float(best_row["params_lr0"]),
                   "lrf":          float(best_row["params_lrf"]),
                   "weight_decay": float(best_row["params_weight_decay"]),
                   "box":          float(best_row["params_box"]),
                   "cls":          float(best_row["params_cls"]),
                   "dfl":          float(best_row["params_dfl"]),
                   "iou":          float(best_row["params_iou"]),
                   }

    return best_params