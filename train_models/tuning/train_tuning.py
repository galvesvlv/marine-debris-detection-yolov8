# Imports
import optuna
from train_models.src.optimizer import objective
from config.config import OPTIMIZER_RESULTS

# Main
def main():
    """
    Run the Optuna optimization pipeline for YOLOv8.

    This function creates an Optuna study, executes hyperparameter
    optimization, reports the best trial, and saves all trial results
    to disk.
    """
    
    study = optuna.create_study(
                                direction="maximize",
                                study_name="yolov8_marine_debris"
                                )

    study.optimize(
                   objective,
                   n_trials=20,
                   timeout=None
                   )

    # Metrics
    print("Best trial:")
    print("  Value:", study.best_value)
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    study.trials_dataframe().to_csv(OPTIMIZER_RESULTS, index=False)

if __name__ == "__main__":
    main()