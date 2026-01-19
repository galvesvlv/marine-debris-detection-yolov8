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

---

## ⚙️ Configuration

The frontend connects to the inference API using the API_URL environment variable.

Default value:
- http://localhost:8000

To override the API address, define the API_URL environment variable before starting the frontend.

When using Docker Compose, this variable is typically defined automatically.

---

## ▶️ Running the Frontend

### Using Docker (recommended)

From the project root directory, run:

docker compose up --build -d

The Streamlit interface will be available at:

http://localhost:8501

---

### Running locally (optional)

From the frontend directory:

pip install -r requirements.txt  
streamlit run app.py

---

## 🐳 Docker Support

The frontend is fully containerized using Docker.

### Dockerfile

The Dockerfile:

- Uses python:3.12-slim as the base image
- Installs Python dependencies from requirements.txt
- Copies the Streamlit application files
- Exposes port 8501
- Launches the application using Streamlit

---

## ⚠️ Requirements

- The inference API must be running and accessible
- Network connectivity between frontend and API must be available

---

## 🧪 Typical Usage Flow

1. Start the inference API service
2. Start the frontend application
3. Open the Streamlit interface in a browser
4. Upload an image or video
5. View and download the annotated output