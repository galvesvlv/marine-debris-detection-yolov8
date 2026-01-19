# YOLOv8 Trained Weights and Optimization Results

This directory provides references to all trained YOLOv8 model versions and the results of the hyperparameter optimization process performed using Optuna.  
Due to file size constraints, model weights and result files are hosted externally and are not versioned directly in this repository.

---

## 📦 Available Model Weights

1. **yolov8n_marinedebris_baseline.pt**  
   Baseline YOLOv8n model trained with small changes in default hyperparameters.  
   🔗 Download: https://drive.google.com/file/d/1nDGSIFQxNPSGYsMdfdB2_4JCkqHwY7Gv/view?usp=sharing

2. **yolov8n_marinedebris_best_final_1.pt**  
   Optimized YOLOv8n model trained using the best hyperparameters obtained from Optuna.  
   🔗 Download: https://drive.google.com/file/d/1NInQ3uZvY5madF00TMSz0VCx5aC16wNF/view?usp=drive_link

3. **yolov8n_marinedebris_best_baseline_tunned.pt**  
   Final tuned YOLOv8n model used for inference in this project.  
   🔗 Download: https://drive.google.com/file/d/1RoUJbsBiNqSECL8iWENET07Wq1yvXVCO/view?usp=sharing

---

## 📊 Hyperparameter Optimization Results

- **optuna_results.csv**  
  CSV file containing the evaluated hyperparameter combinations and their corresponding performance metrics during the Optuna optimization process.  
  🔗 Download: https://drive.google.com/file/d/1xQYyfBiTHTl7RjTTMXiWmbYblV4YXXQ1/view?usp=sharing

---

## 🧪 Notes

- All models were trained using GPU acceleration in a Google Colab environment.
- The tuned model (`yolov8n_marinedebris_best_baseline_tunned.pt`) is the one used in the inference pipeline.
- The baseline model was chosen for further refinement because it achieved better performance than the model obtained through Optuna-based hyperparameter tuning. Detailed experiments and analyses supporting this decision are available in the `train_models` directory.
- External hosting is used to keep the repository lightweight and reproducible.
