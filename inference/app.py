# Imports
import os
import shutil
from pathlib import Path
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from inference import InferencePicture, InferenceVideo  # type: ignore
from api_config import (
                        _save_upload_to_tmp,
                        MODEL_PATH,
                        IMAGE_EXTENSIONS,
                        VIDEO_EXTENSIONS
                        )

# Out of docker in ROOT
"""
from inference.inference import InferencePicture, InferenceVideo

from inference.api_config import (
                                  _save_upload_to_tmp,
                                  MODEL_PATH,
                                  IMAGE_EXTENSIONS,
                                  VIDEO_EXTENSIONS
                                  )
"""

# App
app = FastAPI(
              title="Marine Debris YOLOv8 Inference API",
              version="0.1.0",
              )

# Routes
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Run YOLOv8 inference on an uploaded image and return the annotated image.
    """

    suffix = Path(file.filename).suffix.lower()  # type: ignore

    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid image format: {suffix}",
                            )

    try:
        # Save uploaded image
        image_path = _save_upload_to_tmp(file)

        # Run inference
        infer = InferencePicture(
                                 weights_yolo=str(MODEL_PATH),
                                 image_path=str(image_path),
                                 )

        img_det = infer.run()

        # Encode as JPG
        success, encoded = cv2.imencode(".jpg", img_det)
        if not success:
            raise RuntimeError("Failed to encode image.")

        return Response(
                        content=encoded.tobytes(),
                        media_type="image/jpeg",
                        )

    except Exception as e:
        raise HTTPException(
                            status_code=500,
                            detail=f"Inference error: {str(e)}",
                            )

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(image_path.parent)
        except Exception:
            pass


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Run YOLOv8 inference + tracking on an uploaded video and return the
    annotated video.
    """

    suffix = Path(file.filename).suffix.lower()  # type: ignore

    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid video format: {suffix}",
                            )

    try:
        # Save uploaded video
        video_path = _save_upload_to_tmp(file)

        # Run inference + tracking
        infer = InferenceVideo(
                               input_path=str(video_path),
                               model_path=str(MODEL_PATH),
                               )

        output_path = infer.run()


        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return Response(
                        content=video_bytes,
                        media_type="video/mp4",
                        )

    except Exception as e:
        raise HTTPException(
                            status_code=500,
                            detail=f"Video inference error: {str(e)}",
                            )

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(video_path.parent)
        except Exception:
            pass
