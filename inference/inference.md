# Inference API

This directory contains the FastAPI-based inference service responsible for running object detection and tracking using a trained YOLOv8 model.

The API receives images or videos, performs inference using the trained model, and returns annotated outputs to the frontend application.

---

## 🎯 Purpose

The inference module is designed to:

- Serve YOLOv8 object detection models through a REST API
- Perform image inference and return annotated images
- Perform video inference with object tracking and return annotated videos
- Provide a scalable backend for the Streamlit frontend

All inference is executed server-side; no model logic is implemented in the frontend.

---

## 🧠 Model Used

The inference service uses the final tuned baseline YOLOv8 model:

- **Model**: `yolov8n_marinedebris_best_baseline_tunned.pt`  
- **Description**: Best-performing model selected after training, evaluation and tuning
- **Download link**:  
  https://drive.google.com/file/d/1RoUJbsBiNqSECL8iWENET07Wq1yvXVCO/view?usp=drive_link

The model file is not versioned directly in the repository due to size constraints.

---

## 📡 API Endpoints

### `POST /predict/image`

Runs object detection on a single image.

- **Input**: Image file (`.jpg`, `.jpeg`, `.png`)
- **Output**: Annotated image (`image/jpeg`)
- **Processing**:
  - YOLOv8 object detection
  - Bounding boxes and class labels rendered on the image

---

### `POST /predict/video`

Runs object detection and tracking on a video.

- **Input**: Video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
- **Output**: Annotated video (`video/mp4`)
- **Processing**:
  - Frame-by-frame YOLOv8 inference
  - Object tracking using Norfair (ID persistence across frames)
  - Bounding boxes, object IDs, class labels, and confidence scores rendered per frame

---

## 🧩 Core Components

### `inference.py`

Implements the inference logic:

- **`InferencePicture`**
  - Runs YOLOv8 inference on a single image
  - Returns an annotated image as a NumPy array

- **`InferenceVideo`**
  - Runs YOLOv8 inference on video frames
  - Applies object tracking using Norfair
  - Writes and returns an annotated video file

---

### `api_config.py`

Centralizes inference configuration:

- Model path definition
- Supported image and video extensions
- Temporary file handling for uploaded inputs

---

### `app.py`

Defines the FastAPI application:

- Initializes the API service
- Exposes inference endpoints
- Handles file validation, error handling, and cleanup
- Encodes and returns inference results

---

## 📦 Dependencies

All required dependencies for the inference service are listed in:

```text
requirements.txt
```

### Key libraries include:
- FastAPI / Uvicorn — REST API framework and ASGI server
- Ultralytics (YOLOv8) — Object detection model
- OpenCV — Image and video processing
- Norfair — Multi-object tracking
- NumPy and Pillow — Numerical operations and image handling

---
## 🐳 Docker Support

The inference service is fully containerized using Docker.

### `Dockerfile`

The Dockerfile:

- Uses Python 3.12 as the base image
- Installs system dependencies required for video processing (e.g., FFmpeg)
- Installs Python dependencies from requirements.txt
- Exposes port 8000
- Launches the FastAPI application using Uvicorn

---
## ▶️ Running the Inference API

From the project root directory, start the inference service using Docker Compose:

```bash
docker compose up --build -d
```

The API will be available in: http://localhost:8000
