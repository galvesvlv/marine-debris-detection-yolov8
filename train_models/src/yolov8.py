# Imports
from ultralytics import YOLO  # type: ignore
import shutil
from pathlib import Path
from config.config import (
                           ROOT_DIR, 
                           MODEL_NAME_YOLO
                           )


# Classes
class ModelYoloV8():
    """
    Wrapper class for training, evaluating, and saving YOLOv8 models.

    This class provides a thin abstraction over the Ultralytics YOLO API,
    exposing common workflows such as training, validation, metric extraction,
    and saving the best model weights.
    """

    # Atributes
    def __init__(self, model_name: str = MODEL_NAME_YOLO):
        """
        Initialize a YOLOv8 model.

        Parameters
        ----------
        model_name : str, optional
            Name or path of the YOLOv8 model to load (e.g., 'yolov8n.pt',
            'yolov8m.pt'). Defaults to MODEL_NAME_YOLO.
        """

        self.model = YOLO(model_name)
        self.model_name = model_name
        self.metrics = None


    # Methods
    def fit(self, **kwargs):
        """
        Train the YOLOv8 model.

        This method is a direct wrapper around ``YOLO.train`` and forwards
        all keyword arguments to the underlying Ultralytics API.

        Parameters
        ----------
        **kwargs
            Keyword arguments supported by ``YOLO.train`` (e.g., data, epochs,
            imgsz, batch, device, optimizer).

        Returns
        -------
        object
            Training results object returned by ``YOLO.train``.
        """

        return self.model.train(**kwargs)

    def evaluate(self, **kwargs):
        """
        Evaluate the YOLOv8 model on a validation or test dataset.

        This method runs model validation and extracts the most common
        detection metrics related to bounding boxes.

        Parameters
        ----------
        **kwargs
            Keyword arguments supported by ``YOLO.val`` (e.g., data, split,
            imgsz, device).

        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - 'map50_95': mean Average Precision at IoU 0.50:0.95
            - 'map50'   : mean Average Precision at IoU 0.50
            - 'map75'   : mean Average Precision at IoU 0.75
            - 'per_class_map': per-class mAP values

        Raises
        ------
        ValueError
            If evaluation fails or expected metrics are unavailable.
        """

        self.metrics = self.model.val(**kwargs)

        if self.metrics is None or not hasattr(self.metrics, "box"):
            raise ValueError("Evaluation failed or metrics are unavailable.")

        return {
                "map50_95": self.metrics.box.map,
                "map50": self.metrics.box.map50,
                "map75": self.metrics.box.map75,
                "per_class_map": self.metrics.box.maps
                }

    def save_model(self, weight_name_model):
        """
        Save the best model weights after training.

        This method copies the best-performing weights (as determined during
        training) to a user-defined location.

        Parameters
        ----------
        weight_name_model : str
            Filename (or relative path) to store the best model weights.

        Raises
        ------
        ValueError
            If training has not been completed or best weights are unavailable.
        """
        
        if self.model.trainer is None or self.model.trainer.best is None:
            raise ValueError("Model training has not been completed or 'best' weights are unavailable.")
        
        best = self.model.trainer.best
        shutil.copy(best, ROOT_DIR / weight_name_model)