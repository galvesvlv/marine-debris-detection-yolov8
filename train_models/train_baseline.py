from train_models.src.yolov8 import ModelYoloV8
from config.config import (
                           DATASET_YAML,
                           DEVICE,
                           )

def main():
    """
    Baseline training and evaluation script for YOLOv8.

    This script trains a YOLOv8 model for marine debris detection using a
    fixed baseline configuration, evaluates the trained model on the test
    split, and saves the best-performing weights to disk.
    """

    model_yolov8 = ModelYoloV8()  # 640x640 Size

    # Training
    trained_yolov8 = model_yolov8.fit(
                                      data=DATASET_YAML,
                                      device=DEVICE,
                                      name="baseline",
                                      project="runs/baseline",
                                      imgsz=640,

                                      batch=32,
                                      epochs=30,
                                      patience=5,
                                      freeze=8,  # 10 -> last block layers of the backbone

                                      # Optimizer
                                      optimizer="AdamW",
                                      warmup_epochs = 5,
                                      warmup_bias_lr=0.1,
                                      momentum = 0.937,

                                      # IOU: The smaller the number, the lower the chance of overlap.
                                      iou = 0.5,  # default = 0.7,

                                      weight_decay = 0.0005,
                                      lr0 = 0.003,
                                      lrf = 0.01,  # lr_final = lr0 * lrf

                                      # Losses
                                      box = 10.,  # default = 7.5
                                      cls = 0.8,  # default = 0.5
                                      dfl = 2.,   # default = 1.5
                                      )

    # Test Metrics
    print("\n" * 5)
    print(f"TEST METRICS")
    metrics_yolov8 = model_yolov8.evaluate(
                                           data=DATASET_YAML,
                                           device=DEVICE,
                                           split="test"
                                           )
    print(metrics_yolov8)

    # Saving model in memory
    model_yolov8.save_model(weight_name_model="yolov8n_marinedebris_baseline.pt")

if __name__ == "__main__":
    main()