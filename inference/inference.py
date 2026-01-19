from pathlib import Path
import cv2
from ultralytics import YOLO  # type: ignore
import numpy as np
from norfair import Detection, Tracker  # type: ignore

CLASS_COLORS = {
                "can": (255, 0, 0),               # blue
                "foam": (0, 255, 255),            # yellow
                "plastic": (0, 255, 0),           # green
                "plastic bottle": (0, 165, 255),  # orange
                "unknow": (128, 128, 128),        # gray
                }

class InferencePicture():
    """
    Perform YOLOv8 inference on a single image.

    This class runs object detection on an input image and returns
    the annotated image as a NumPy array, suitable for API responses
    or further processing.
    """

    def __init__(self, weights_yolo, image_path):
        """
        Initialize the image inference pipeline.

        Parameters
        ----------
        weights_yolo : str or pathlib.Path
            Path to the YOLOv8 model weights file.
        image_path : str or pathlib.Path
            Path to the input image used for inference.
        """

        self.model = YOLO(weights_yolo)
        self.image_path = image_path

    def run(self):
        """
        Run YOLOv8 inference and return the annotated image.

        Returns
        -------
        np.ndarray
            Annotated image in RGB format (H, W, 3), dtype uint8.
        """

        results = self.model.predict(
                                     source=self.image_path,
                                     imgsz=640,
                                     agnostic_nms=True
                                     )
        
        img_bgr = results[0].plot()

        return img_bgr

class InferenceVideo():
    """
    Perform object detection and tracking on a video using a YOLO model.

    This class loads a trained YOLO model, iterates over video frames,
    performs inference, tracks detected objects across frames, and
    renders bounding boxes with object IDs, class labels, and confidence
    scores on the output video.
    """

    def __init__(self, input_path: str, model_path):
        """
        Initialize the video inference pipeline.

        Parameters
        ----------
        input_path : str
            Path to the input video file.
        model_path : str
            Path to the trained YOLO model weights.
        """

        self.input_path = input_path
        self.model = YOLO(model_path)
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=100)


    def run(self):
        """
        Run inference and tracking over the entire video.

        This version explicitly controls video reading and writing
        using OpenCV to ensure compatibility in Docker environments.
        """

        # OpenCV reader
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not fps or fps <= 0:
            fps = 30.0

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # OpenCV writer
        in_path = Path(self.input_path)
        output_path = str(in_path.with_name(in_path.stem + "_annotated.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Frame loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference
            results = self.model(frame, agnostic_nms=True, conf=0.4)
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = xyxy.tolist()

                    center = np.array(
                                      [(x1 + x2) / 2, (y1 + y2) / 2],
                                      dtype=np.float32
                                      )

                    detections.append(
                        Detection(
                            points=center,
                            scores=np.array([float(conf)]),
                            data={
                                  "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                  "class_name": self.model.names[int(cls_id)],
                                  "conf": float(conf),
                                  },
                                  )
                                      )

            # Norfair tracking
            tracked_objects = self.tracker.update(detections=detections)

            # Drawing bounding boxes
            for obj in tracked_objects:
                det = obj.last_detection
                if det is None or det.data is None:
                    continue

                x1, y1, x2, y2 = det.data["bbox"]
                label = det.data["class_name"]
                conf = det.data["conf"]

                color = CLASS_COLORS.get(label, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                            frame,
                            f"ID {obj.id} | {label} {conf:.2f}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                            )

            # writer
            writer.write(frame)

        # Cleanup
        cap.release()
        writer.release()

        return output_path