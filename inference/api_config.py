# Imports
from pathlib import Path
import shutil
import tempfile
from fastapi import UploadFile

# Configuration
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "yolov8n_marinedebris_best_baseline_tunned.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Helper functions
def _save_upload_to_tmp(file: UploadFile) -> Path:
    """
    Save an uploaded file to a temporary directory and return its path.
    """
    suffix = Path(file.filename).suffix.lower()  # type: ignore 

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / f"input{suffix}"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return tmp_path