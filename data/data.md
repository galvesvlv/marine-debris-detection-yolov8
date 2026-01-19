# Dataset Description

This directory contains the marine debris dataset used for training, validation, and testing of the YOLOv8 object detection models.

The dataset follows the standard YOLOv8 directory and annotation structure and was obtained from Roboflow.

---
## 📦 Dataset Source

- Platform: Roboflow
- Project: Marine Debris
- Version: 2
- License: CC BY 4.0  
- URL: https://app.roboflow.com/galvesvlv/marine-debris-i2ge3-3hnmu/2

---
## 🏷️ Classes
- `can`
- `foam`
- `plastic`
- `plastic bottle`
- `unknow`

---
## 📊 Dataset Size

| Split | Number of Images |
|------|------------------|
| Train | 1200 |
| Validation | 115 |
| Test | 60 |

### Notes
- The reported number of training images includes data augmentation applied during the dataset preparation process on Roboflow.
- Prior to data augmentation, the dataset was split as follows:
  - 70% training
  - 20% validation
  - 10% testing

---
## 📂 Directory Structure (YOLOv8 Format)

```text
dataset_marinedebris_yolov8/
 ├── train/
 │   ├── images/
 │   └── labels/
 ├── valid/
 │   ├── images/
 │   └── labels/
 ├── test/
 │   ├── images/
 │   └── labels/
 └── data.yaml
