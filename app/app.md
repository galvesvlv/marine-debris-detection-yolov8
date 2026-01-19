# Frontend Application

This directory contains the Streamlit-based frontend used to interact with the marine debris detection system powered by YOLOv8.

The frontend provides a simple web interface for uploading images or videos and visualizing the object detection results returned by the inference API.

---

## 🎯 Purpose

The frontend application allows users to:

- Upload images or videos containing marine debris
- Send inputs to the inference API
- Visualize annotated images directly in the browser
- Download processed images and videos

The frontend does not perform inference locally; all predictions are handled by the backend API.

---

## 🧠 Application Overview

### `app.py`

The main Streamlit application responsible for:

- Rendering the user interface
- Handling file uploads (images and videos)
- Communicating with the inference API via HTTP requests
- Displaying and exporting annotated results

Key features include:

- Support for both **image** and **video** inference
- Clear user instructions via sidebar
- Progress indicators during inference
- Download buttons for annotated outputs

---

## 🔌 API Integration

The frontend communicates with the inference API through the following endpoints:

- `POST /predict/image` — image inference
- `POST /predict/video` — video inference

The API base URL is configured via the environment variable:

```text
API_URL
