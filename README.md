# 🛡️ CodeCluster AI Services — Real-Time Multimodal AI Proctoring Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![gRPC](https://img.shields.io/badge/gRPC-1.68-244c5a.svg?style=flat&logo=grpc&logoColor=white)](https://grpc.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-007ACC.svg?style=flat&logo=google&logoColor=white)](https://mediapipe.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C.svg?style=flat&logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**CodeCluster AI Services** is an enterprise-grade, real-time multimodal automated proctoring platform designed for secure online examinations. It features a dual-service architecture: a high-performance **FastAPI Gateway & WebSocket Orchestrator** paired with an **8-Model Python ML Worker (`CodeClusterML`)** connected over ultra-low-latency **gRPC**.

---

## 🌟 Key Features

- 👁️ **Eye Gaze Tracking**: Real-time detection of candidates looking away (LEFT, RIGHT, UP, DOWN, OFF_SCREEN).
- 🗣️ **Continuous Speech & Audio Monitoring**: Silero VAD (Voice Activity Detection) running on microphone streams to detect unauthorized speech and whisper background noise.
- 📐 **3D Head Pose Estimation**: MediaPipe FaceLandmarker + OpenCV `solvePnP` tracking pitch, roll, and yaw angles (detecting head turns > 30°).
- 👤 **Face Counter & Multi-Person Alert**: Real-time detection of zero faces (missing candidate) or 2+ faces (unauthorized presence).
- 📸 **Snapshot Face Authentication**: Cosine-similarity verification with InsightFace (`buffalo_l`) matching candidates against reference photos every 5 seconds.
- 📱 **Object & Banned Device Detection**: YOLOv8 real-time detection of banned examination objects (cell phones, laptops, books).
- 👓 **Spectacles & Eyewear Detection**: ONNX neural model identifying glasses with 10-frame majority voting.
- 👁️‍🗨️ **Blink Rate & EAR Analysis**: Eye Aspect Ratio tracking for drowsiness and suspicious eye movement patterns.
- ⚡ **Real-Time WebSocket Gateway**: High-throughput FastAPI backend managing live candidate streams, warning deduplication, grace periods, and candidate trust score decay.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │             Candidate Web Browser            │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
        WebSocket @ 30 FPS                                 JPEG Snapshot @ 5s
  (MediaPipe Client Landmarks / Telemetry)           (High-Res Image Stream)
                    │                                                   │
                    ▼                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend Gateway                           │
│                            (Port 8000 / WebSockets)                          │
│                                                                              │
│   • Connection Lifecycle Manager & Session Store                             │
│   • Candidate Trust Score Engine (Starts @ 100, decays per violation)         │
│   • Warning Policy Manager (Grace period, deduplication & escalation)         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                               gRPC IPC (Port 50051)
                            `proctor.proto` Stubs
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CodeClusterML Engine (Port 50051)                     │
│                                                                              │
│   ┌──────────────────────────────────┐  ┌────────────────────────────────┐   │
│   │    VideoMonitor (Thread @ 10FPS) │  │  AudioMonitor (Always On)     │   │
│   │    • Gaze Tracker (MediaPipe)    │  │  • Silero VAD (PyTorch)        │   │
│   │    • Head Pose (SolvePnP)        │  │                                │   │
│   │    • Blink Detector (EAR)        │  └────────────────────────────────┘   │
│   │    • Face Counter                │                                       │
│   └──────────────────────────────────┘                                       │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │    5-Second Snapshot Pipeline (Multi-Model Heavy Inference)          │   │
│   │    • YOLOv8 Object Detection (Banned Devices)                        │   │
│   │    • InsightFace Auth (Buffalo_L Cosine Similarity)                   │   │
│   │    • Spectacles Classifier (ONNX Neural Net)                         │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Models Specification

| # | Model | Implementation | Processing Cadence | Output Metrics |
|---|---|---|---|---|
| 1 | **Eye Gaze Tracker** | MediaPipe FaceMesh (Irises) | Continuous (10 FPS) | `gazeDirection` (CENTER, LEFT, RIGHT, UP, DOWN, OFF_SCREEN) |
| 2 | **3D Head Pose** | OpenCV `solvePnP` | Continuous (10 FPS) | Yaw, Pitch, Roll angles & `isHeadTurned` flag |
| 3 | **Blink & EAR** | MediaPipe Eye Aspect Ratio | Continuous (10 FPS) | `eyesClosed` boolean & EAR score |
| 4 | **Face Counter** | MediaPipe FaceDetection | Continuous (10 FPS) | `faceCount` (0, 1, or 2+) & `faceStatus` |
| 5 | **Speech Detector** | Silero VAD (PyTorch) | Continuous (Audio Stream) | `isHumanSpeech` & speech probability score |
| 6 | **Object Detector** | YOLOv8n (Ultralytics COCO) | Snapshot (Every 5s) | Banned items list (`cell phone`, `book`, `laptop`) |
| 7 | **Face Authentication**| InsightFace (`buffalo_l`) | Snapshot (Every 5s) | `similarityScore` & `faceMatched` threshold check |
| 8 | **Spectacles Detector**| Pre-trained ONNX Model | Snapshot (Every 5s) | `specsDetected` boolean (with 10-frame windowing) |

---

## 📁 Repository Structure

```
CodeCluster-AI-Services/
├── CodeClusterML/                 # Python ML Engine & gRPC Worker
│   ├── config.py                  # Single source of truth for ML thresholds & config
│   ├── grpc_service/              # gRPC Server & Protobuf Definitions
│   │   ├── ml_server.py           # Main gRPC server runner (Port 50051)
│   │   ├── servicer.py            # ProctoringServicer implementation
│   │   └── proctor.proto          # Protocol Buffers contract
│   ├── models/                    # Individual ML Model Implementations
│   │   ├── gaze_tracker.py        # MediaPipe Gaze Estimation
│   │   ├── head_pose.py           # 3D Head Orientation via SolvePnP
│   │   ├── blink_detector.py      # Eye Aspect Ratio (EAR) calculation
│   │   ├── face_counter.py        # Real-time multi-face detector
│   │   ├── speech_detector.py     # Silero VAD PyTorch wrapper
│   │   ├── object_detector.py     # YOLOv8 banned object classifier
│   │   ├── face_auth.py           # InsightFace facial embedding matcher
│   │   └── spectacles_detector.py # ONNX eyeglass detection
│   ├── pipeline/                  # Asynchronous Monitoring Pipelines
│   │   ├── video_monitor.py       # Continuous Video Thread (10 FPS)
│   │   ├── audio_monitor.py       # Continuous Audio Thread (VAD)
│   │   └── proctoring_pipeline.py # 5-second snapshot aggregator
│   ├── weights/                   # Neural network weights (YOLO, ONNX, InsightFace)
│   ├── requirements.txt           # Python ML dependencies
│   └── Dockerfile                 # Container setup for ML worker
├── backend/                       # FastAPI Orchestrator & Gateway
│   ├── app.py                     # Main FastAPI server entry point (Port 8000)
│   ├── api/                       # API Endpoints & WebSockets
│   │   └── websocket.py           # Real-time WebSocket connection handler
│   ├── clients/                   # gRPC Client to CodeClusterML
│   │   └── ai_client.py           # Asynchronous gRPC connection wrapper
│   ├── config/                    # Backend policies & warning configurations
│   │   └── warning_policy.py      # Trust score decay & warning rules
│   ├── services/                  # Business Logic & Orchestration
│   │   ├── proctor_service.py     # Session state & snapshot orchestration
│   │   ├── rule_engine.py         # Violation trigger validator
│   │   └── warning_manager.py     # Warning count & penalty administrator
│   └── requirements.txt           # Backend dependencies
├── docker-compose.yml             # Full-stack Docker orchestration
├── run_project.bat                # Windows one-click launcher script
└── README.md                      # Project documentation
```

---

## ⚙️ Configuration & Thresholds

All ML model parameters and detection sensitivity levels are central in `CodeClusterML/config.py`:

```python
# Head Pose & Gaze Thresholds
HEAD_YAW_THRESHOLD_DEG      = 30.0   # Yaw angle limit before triggering HEAD_TURNED
GAZE_CONSECUTIVE_FRAMES     = 8      # Frame persistence for LOOKING_AWAY trigger

# Face Auth & Recognition
FACE_MATCH_THRESHOLD        = 0.72   # InsightFace cosine similarity threshold

# Object & Spectacles Detection
YOLO_CONFIDENCE_THRESHOLD   = 0.65   # Object detection confidence filter
BANNED_CLASSES              = ["cell phone", "book", "laptop"]
SPECS_CONFIDENCE_THRESHOLD  = 0.50   # ONNX Spectacles detection threshold

# Audio & Speech Detection
SPEECH_CONFIDENCE_THRESHOLD = 0.50   # Silero VAD speech probability threshold
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.12+**
- **Git**
- **Webcam & Microphone** (for local hardware testing)
- **C++/Build Tools** (for InsightFace compilation if installing from source)

### Local Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Gondalwad/CodeCluster-AI-Services.git
   cd CodeCluster-AI-Services
   ```

2. **Setup Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies & PyTorch:**
   ```bash
   pip install torch==2.5.1 torchaudio==2.5.1
   pip install -r CodeClusterML/requirements.txt
   pip install -r backend/requirements.txt
   ```

4. **Download Model Weights:**
   ```bash
   # Download YOLOv8n weights into CodeClusterML/weights/
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   mkdir -p CodeClusterML/weights
   mv yolov8n.pt CodeClusterML/weights/yolov8n.pt
   ```
   > *Note: InsightFace (`buffalo_l`) and Silero VAD weights download automatically on initial startup.*

5. **Run Services:**
   - **Start ML gRPC Server:**
     ```bash
     python CodeClusterML/grpc_service/ml_server.py
     ```
   - **Start FastAPI Backend:**
     ```bash
     uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
     ```
   - *Alternatively on Windows, run `run_project.bat` to launch all components simultaneously.*

---

### Docker Containerized Deployment

To spin up the entire AI Proctoring suite in isolated Docker containers:

```bash
docker-compose up --build
```

- **FastAPI Backend Gateway**: `http://localhost:8000`
- **gRPC ML Service**: `localhost:50051`

---

## 🧪 Testing & Verification

### Protocol Buffers Stub Generation
If modifying `CodeClusterML/grpc_service/proctor.proto`, recompile the stubs via:

```bash
python -m grpc_tools.protoc -ICodeClusterML/grpc_service --python_out=CodeClusterML/grpc_service --grpc_python_out=CodeClusterML/grpc_service CodeClusterML/grpc_service/proctor.proto
```

---

## 🚨 Violation Categories & Trust Decay

| Event Trigger | Detection Origin | Penalty Threshold / Action |
|---|---|---|
| `LOOKING_AWAY` | VideoMonitor Gaze | 8+ consecutive frames looking off-screen |
| `HEAD_TURNED` | VideoMonitor Head Pose | Yaw magnitude > 30° for 6+ frames |
| `FACE_MISSING` | VideoMonitor FaceCounter | Candidate face absent for 10+ frames |
| `MULTIPLE_FACES` | VideoMonitor FaceCounter | 2 or more faces detected in frame |
| `SPEECH_DETECTED` | AudioMonitor Silero VAD | Active human speech detected on microphone |
| `BANNED_OBJECT` | Snapshot YOLOv8 | Detection of phone, book, or external computer |
| `IDENTITY_MISMATCH` | Snapshot InsightFace | Cosine match score < 0.72 against baseline |
| `SPECTACLES_DETECTED` | Snapshot ONNX | Eyeglass presence confirmed |

---

## 📜 License

This project is proprietary and confidential. Developed for the **CodeCluster Online Examination Platform**. All rights reserved.
