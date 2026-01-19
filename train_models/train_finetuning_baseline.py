import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train_models.src.yolov8 import ModelYoloV8
from train_models.src.optimizer import load_best_params
from config.config import (
                           WEIGHTS_YOLOV8_BASELINE,
                           MODEL_NAME_YOLO_FINAL_BASELINE_TUNNED,
                           DATASET_YAML,
                           DEVICE,
                           )

def main():
    """
    Fine-tuning and evaluation script for the previous trained YOLOv8 baseline model.

    This script fine-tunes the previous trained YOLOv8 baseline model version in "train_baseline.py" 
    using a reduced learning rate, evaluates performance on the test dataset, visualizes the
    confusion matrix, and saves the tuned model weights.
    """

    model = ModelYoloV8(str(WEIGHTS_YOLOV8_BASELINE))

    # Fine-tuning
    model.fit(
              data=DATASET_YAML,
              device=DEVICE,
              imgsz=640,
              batch=16,
              epochs=20,
              patience=5,
              freeze=0,
              lr0=0.003 * 0.1
              )

    metrics = model.evaluate(
                             data=DATASET_YAML,
                             device=DEVICE,
                             split="test"
                             )

    # Confusion Matrix
    cm = model.metrics.confusion_matrix.matrix.astype(int)  # type: ignore
    sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                cbar=True,
                linewidths=0.5,
                linecolor="white"
                )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.show()

    print("FINAL TEST METRICS:", metrics)

    model.save_model(weight_name_model=MODEL_NAME_YOLO_FINAL_BASELINE_TUNNED)

if __name__ == "__main__":
    main()