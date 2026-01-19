# Imports
from config.config import (
                           DATASET_DIR_YOLO, 
                           DATASET_YAML
                           )

from preprocessing.preprocessing import PreProcessorYoloV8

# Main
def main():
    prep = PreProcessorYoloV8(
                              model_name="yolov8",
                              data_path=DATASET_DIR_YOLO,
                              yaml_path=DATASET_YAML
                              )

    counts = prep.count_all()

    for split, classes in counts.items():
        print(f"\n{split.upper()}")

        for cls, n in classes.items():
            print(f"{cls}: {n}")

    prep.classes_show()

    prep.plot_class_examples()


if __name__ == "__main__":
    main()