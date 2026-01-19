# imports
from pathlib import Path
from collections import Counter, defaultdict
import yaml
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Classes
class PreProcessorYoloV8:
    """
    Utility class for inspecting and visualizing YOLOv8 datasets.

    This class provides tools to:
    - Load class definitions from a YOLO data.yaml file
    - Count object instances per class and dataset split
    - Visualize class distribution across splits
    - Plot example images with YOLO bounding box annotations

    It assumes a standard YOLO directory structure:
    data_path/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
    """
    def __init__(self, model_name, data_path, yaml_path):
        """
        Initialize the YOLOv8 dataset preprocessor.

        Parameters
        ----------
        model_name : str
            Name or identifier of the YOLO model (e.g., 'yolov8n', 'yolov8m').
            Stored for reference and logging purposes.
        data_path : pathlib.Path
            Root directory of the YOLO dataset.
        yaml_path : pathlib.Path
            Path to the YOLO data.yaml file containing class definitions.
        """

        self.model_name = model_name
        self.data_path = data_path
        self.yaml_path = yaml_path
        self.class_map = self._load_classes()  # for yolov8
        self.results = None


    def _load_classes(self) -> dict:
        """
        Load class names from the YOLO data.yaml file.

        Returns
        -------
        dict
            Mapping from class index (int) to class name (str).

        Raises
        ------
        TypeError
            If the 'names' field in the YAML file is not a list.
        """

        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        names = data["names"]

        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}

        raise TypeError("Formato inválido para 'names' no data.yaml")

    def _count_split(self, split: str) -> Counter:
        """
        Count object instances per class for a given dataset split.

        Parameters
        ----------
        split : str
            Dataset split name ('train', 'valid', or 'test').

        Returns
        -------
        collections.Counter
            Counter mapping class_id (int) to number of objects.
        """

        labels_path = self.data_path / split / "labels"
        counter = Counter()

        for label_file in labels_path.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

        return counter


    def count_all(self) -> dict:
        """
        Count object instances per class for all dataset splits.

        The results are stored internally and returned as a dictionary
        indexed by split name and class name.

        Returns
        -------
        dict
            Nested dictionary of the form:
            {
                'train': {'class_name': count, ...},
                'valid': {'class_name': count, ...},
                'test':  {'class_name': count, ...}
            }
        """

        self.results = {}

        for split in ["train", "valid", "test"]:
            split_counter = self._count_split(split)

            self.results[split] = {
                                    self.class_map[class_id]: count
                                    for class_id, count in split_counter.items()
                                    }

        return self.results


    def _autolabel(self, bars, values, total):
        """
        Attach percentage labels above bar plots.

        Parameters
        ----------
        bars : matplotlib.container.BarContainer
            Bars returned by matplotlib's bar() function.
        values : iterable
            Numerical values corresponding to each bar.
        total : float
            Total value used to compute percentages.
        """

        for bar, v in zip(bars, values):
            if v == 0:
                continue

            pct = 100 * v / total

            plt.text(
                     bar.get_x() + bar.get_width() / 2,
                     bar.get_height(),
                     f"{pct:.1f}%",
                     ha="center",
                     va="bottom",
                     fontsize=8
                     )


    def classes_show(self):
        """
        Plot class distribution per dataset split.

        Displays a grouped bar chart showing the number and percentage
        of objects per class for train, validation, and test splits.

        Raises
        ------
        RuntimeError
            If count_all() has not been executed beforehand.
        """

        if self.results is None:
            raise RuntimeError("Execute count_all() antes de chamar classes_show().")

        df = pd.DataFrame(self.results).fillna(0)

        classes = df.index
        x = np.arange(len(classes))
        width = 0.25
        totals = df.sum(axis=0)

        # Plot
        plt.figure(figsize=(10, 5))

        bars_train = plt.bar(x - width, df["train"], width, label="Train")
        bars_val   = plt.bar(x,         df["valid"], width, label="Val")
        bars_test  = plt.bar(x + width, df["test"],  width, label="Test")

        self._autolabel(bars_train, df["train"], totals["train"])
        self._autolabel(bars_val,   df["valid"], totals["valid"])
        self._autolabel(bars_test,  df["test"],  totals["test"])

        plt.xticks(x, classes, rotation=45)  # type: ignore
        plt.ylabel("Number of objects")
        plt.title("YOLO class distribution (%) per split")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_class_examples(self, split: str = "train"):
        """
        Plot one example image per class with YOLO bounding boxes.

        For each class, a random labeled image is selected and all
        bounding boxes are drawn. The target class is highlighted.

        Parameters
        ----------
        split : str, optional
            Dataset split to visualize ('train', 'valid', or 'test'),
            by default 'train'.
        """
    
        images_path = self.data_path / split / "images"
        labels_path = self.data_path / split / "labels"
    
        # Map: class_id -> label file
        class_examples = {}
    
        # Find one image per class
        for label_file in labels_path.glob("*.txt"):
            with open(label_file) as f:
                lines = f.readlines()
    
            for line in lines:
                class_id = int(line.split()[0])
    
                class_examples.setdefault(class_id, []).append(label_file)
        
        class_examples = {
                          cid: random.choice(files)
                          for cid, files in class_examples.items()
                         }

        n_classes = len(class_examples)
        ncols = min(4, n_classes)
        nrows = int(np.ceil(n_classes / ncols))
    
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
    
        for ax, (class_id, label_file) in zip(axes, class_examples.items()):
            image_file = images_path / (label_file.stem + ".jpg")
    
            if not image_file.exists():
                image_file = images_path / (label_file.stem + ".png")
    
            img = plt.imread(image_file)
            h, w = img.shape[:2]
    
            ax.imshow(img)
            ax.axis("off")
    
            with open(label_file) as f:
                for line in f:
                    cid, xc, yc, bw, bh = map(float, line.split())
    
                    # YOLO → pixel coords
                    xmin = (xc - bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    xmax = (xc + bw / 2) * w
                    ymax = (yc + bh / 2) * h
    
                    color = "blue" if int(cid) == class_id else "lime"
    
                    rect = plt.Rectangle(              # type: ignore
                                         (xmin, ymin),
                                         xmax - xmin,
                                         ymax - ymin,
                                         fill=False,
                                         color=color,
                                         linewidth=2
                                         )
                    ax.add_patch(rect)
    
            class_name = self.class_map[class_id]
            ax.set_title(class_name, fontsize=12, color="blue")
    
        # Remove empty axes
        for ax in axes[len(class_examples):]:
            ax.axis("off")
    
        plt.tight_layout()
        plt.show()
        