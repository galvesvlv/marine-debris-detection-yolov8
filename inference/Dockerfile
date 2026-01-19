# API Docker

# Image
FROM python:3.12

# Metadata
LABEL maintainer="galvesvlv@gmail.com"

# System dependecies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY application
COPY . .

# Expose API port
EXPOSE 8000

# Running FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]