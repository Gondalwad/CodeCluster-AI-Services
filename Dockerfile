# Unified Dockerfile for CodeCluster AI Proctoring System (Backend + ML gRPC Service)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/CodeClusterML:/app/backend"

# Install system dependencies (OpenCV, MediaPipe & C++ libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python dependencies
COPY CodeClusterML/requirements.txt /app/ml_requirements.txt
COPY backend/requirements.txt /app/backend_requirements.txt
RUN pip install --no-cache-dir -r /app/ml_requirements.txt -r /app/backend_requirements.txt

# Copy source directories and weights
COPY CodeClusterML /app/CodeClusterML
COPY backend /app/backend
COPY start_services.sh /app/start_services.sh

# Make start script executable
RUN chmod +x /app/start_services.sh

# Expose FastAPI HTTP/WebSocket port (8000) and gRPC port (50051)
EXPOSE 8000 50051

ENTRYPOINT ["/app/start_services.sh"]
