# Fine-Tuning the Baseline Model – YOLOv8

This document describes the final fine-tuning stage applied to the baseline YOLOv8 model for marine debris detection project.

After evaluating all trained configurations, the manually adjusted baseline model outperformed the Optuna-optimized model. Therefore, the baseline weights were used as a strong initialization point for an additional fine-tuning step unfreezing all the backbone.

---

## 🎯 Motivation

Although hyperparameter optimization was explored, the baseline model achieved superior and more stable performance.

Instead of further tuning hyperparameters, the strategy adopted here was to **fine-tune the baseline model itself**, allowing the network to better adapt its internal representations to the marine debris domain.

The objectives of this step were to:

- Improve domain-specific feature learning
- Refine localization and classification performance
- Reduce systematic misclassifications observed in previous stages
- Maximize performance without changing the overall model architecture

---

## 🧠 Fine-Tuning Strategy

**File:** `train_finetuning_baseline.py`

The fine-tuning procedure differs from the baseline training in three key aspects.

---

### 1️⃣ Full Backbone Unfreezing

```text
freeze = 0
```

All backbone layers were unfrozen, allowing gradients to flow through the entire network.

This choice enables:

- Adaptation of low-level features (textures, edges, reflections)
- Better representation learning for marine environments
- Reduction of bias inherited from ImageNet pretraining

This step can be useful because marine debris imagery may differs from typical natural image datasets.

---

### 2️⃣ Reduced Learning Rate

```text
lr0 = 0.003 × 0.1
```
A reduced learning rate was used to:

- Preserve previously learned representations
- Avoid catastrophic forgetting
- Enable controlled, fine-grained weight updates

This setup favors refinement rather than aggressive re-learning.

---

### 3️⃣ Conservative Training Schedule

| Parameter | Value |
|---------|-------|
| Batch size | 16 |
| Epochs | 20 |
| Early stopping patience | 5 |
| Image size | 640 × 640 |

A smaller batch size increases gradient variability, which can help escape shallow minima during fine-tuning.

---

## 📊 Test Performance

**Test split (60 images, 630 instances)**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|------|----------|--------|--------|-----------|
| all | 0.758 | 0.692 | 0.779 | 0.464 |
| can | 0.866 | 0.732 | 0.901 | 0.494 |
| foam | 0.799 | 0.742 | 0.842 | 0.593 |
| plastic | 0.683 | 0.639 | 0.714 | 0.404 |
| plastic bottle | 0.768 | 0.799 | 0.841 | 0.518 |
| unknow | 0.673 | 0.547 | 0.599 | 0.310 |

**Overall metrics:**

| Metric | Value |
|------|-------|
| mAP@50–95 | 0.464 |
| mAP@50 | 0.779 |
| mAP@75 | 0.497 |

This configuration achieved the **best overall performance** among all trained models.

---

## 🔍 Confusion Matrix Analysis

The confusion matrix was computed using the test dataset and represents raw prediction counts.
    - It can be seen in: https://drive.google.com/file/d/1gkqJLfjIw-wY4HK2-5YKO4LEG88ovJ9-/view?usp=sharing

**Key observations:**

- **Can (0)**  
  Strong diagonal dominance, indicating robust detection performance. Limited confusion occurs mainly with the *unknow* class and with the *background*, corresponding to missed detections or low-confidence predictions in visually cluttered regions.

- **Foam (1)**  
  High number of correct predictions with minor confusion toward *plastic*, *plastic bottle*, and *unknow*. A noticeable number of instances are associated with the *background*, reflecting challenges in detecting low-contrast foam regions over the ocean surface.

- **Plastic (2)**  
  Correctly detected in most cases, with minor confusion toward *foam*, *plastic bottle* and *unknow*. A significant portion of plastic instances are associated with the *background*, indicating missed detections in scenes with complex textures, lighting conditions, waves, foam from the waves and other types of pollution.

- **Plastic bottle (3)**  
  Good overall class separation, with moderate confusion with *plastic*. A considerable number of false negatives appear as *background*, likely due to complex textures, lighting conditions, waves, foam from the waves and other types of pollution.

- **Unknow (4)**  
  Displays moderate confusion across multiple classes, which is expected given its heterogeneous and loosely defined nature. Background-related errors are also present, reflecting intrinsic ambiguity in this category and the difficulty of the model has in dealing with the complexity of this type of background.

- **Background (5)**  
  Represents false positives and false negatives rather than a semantic class. Significant interaction with all classes is observed, likely caused by the non-uniform ocean background, including waves, foam patterns, reflections, and other types of marine pollution that visually resemble debris. This appears to be the model's greatest difficulty in predicting these types of pollution observed in this dataset.


---

## ⚡ Computational Performance

- **Model size:** 3.0M parameters
- **GFLOPs:** 8.1
- **Inference time:** ~9 ms per image (Tesla T4)
- **Total training wall time:** ~10 minutes

---

## 💾 Output Artifact

- **Final model:** `yolov8n_marinedebris_best.pt
    - https://drive.google.com/file/d/1RoUJbsBiNqSECL8iWENET07Wq1yvXVCO/view?usp=drive_link
