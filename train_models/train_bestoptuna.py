import pandas as pd
from train_models.src.yolov8 import ModelYoloV8
from train_models.src.optimizer import load_best_params
from config.config import (
                           ROOT_DIR,
                           OPTIMIZER_RESULTS,
                           MODEL_NAME_YOLO,
                           DATASET_YAML,
                           DEVICE,
                           )

def main():
    """
    Final training and evaluation script for YOLOv8 using Optuna-selected parameters.

    This script loads the best hyperparameters obtained from Optuna optimization,
    trains a final YOLOv8 model on the full training setup, evaluates it on the
    test dataset, and saves both the trained weights and the parameters used.
    """

    best_params = load_best_params(OPTIMIZER_RESULTS)
    print(best_params)

    model = ModelYoloV8(MODEL_NAME_YOLO)

    # Final Train
    model.fit(
              data=DATASET_YAML,
              device=DEVICE,
              imgsz=640,

              batch=32,
              epochs=50,
              patience=5,

              freeze=8,

              optimizer="AdamW",
              warmup_epochs=5,
              warmup_bias_lr=0.1,
              momentum=0.937,

              **best_params
              )

    metrics = model.evaluate(
                             data=DATASET_YAML,
                             device=DEVICE,
                             split="test"
                             )

    print("FINAL TEST METRICS:", metrics)

    model.save_model(weight_name_model="yolov8n_marinedebris_best_final_1.pt")
    pd.Series(best_params).to_csv(ROOT_DIR / "best_params_used_final_model_1.csv")

if __name__ == "__main__":
    main()