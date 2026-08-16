# 🤖 CodeClusterML — AI Proctoring Worker Service

`CodeClusterML` is the Python-based machine learning inference worker for the CodeCluster Online Examination Proctoring Platform. It runs 8 deep learning and computer vision models across real-time video/audio monitoring threads and a high-performance gRPC server interfacing with the backend orchestrator.

---

## 🏗️ Worker Architecture

```
                                  gRPC Client (Port 50051)
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  ProctoringServicer       │
                               │  (gRPC Service Interface) │
                               └─────────────┬─────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               │                             │                             │
               ▼                             ▼                             ▼
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │ VideoMonitor Thread   │     │ AudioMonitor Thread   │     │ 5-Second Snapshot     │
   │ (~10 FPS Loop)        │     │ (Always-On Listener)  │     │ Pipeline              │
   │                       │     │                       │     │                       │
   │ • GazeTracker         │     │ • Silero VAD (Speech) │     │ • YOLOv8 Object Det   │
   │ • HeadPose (solvePnP) │     └───────────────────────┘     │ • InsightFace Auth    │
   │ • BlinkDetector (EAR) │                                   │ • Spectacles (ONNX)   │
   │ • FaceCounter         │                                   └───────────────────────┘
   └───────────────────────┘
```

---

## 🧠 ML Models Specification

| Model | Submodule | Framework / Library | Execution Cadence | Purpose & Metrics |
|---|---|---|---|---|
| **Eye Gaze Tracker** | `models/gaze_tracker.py` | MediaPipe FaceMesh (Iris Landmarks) | Continuous (~10 FPS) | Identifies gaze direction (CENTER, LEFT, RIGHT, UP, DOWN, OFF_SCREEN) |
| **3D Head Pose** | `models/head_pose.py` | OpenCV `solvePnP` + MediaPipe | Continuous (~10 FPS) | Computes pitch, roll, yaw angles and flags `isHeadTurned` (>30°) |
| **Blink Detector** | `models/blink_detector.py` | MediaPipe EAR calculation | Continuous (~10 FPS) | Computes Eye Aspect Ratio and flags `eyesClosed` |
| **Face Counter** | `models/face_counter.py` | MediaPipe FaceDetection | Continuous (~10 FPS) | Detects number of human faces in frame (`0`, `1`, `2+`) |
| **Speech Detector** | `models/speech_detector.py` | Silero VAD (PyTorch) | Continuous (Mic Input) | Evaluates human voice probability (`isHumanSpeech`) |
| **Object Detector** | `models/object_detector.py` | YOLOv8n (Ultralytics) | Snapshot (Every 5s) | Detects prohibited exam items (`cell phone`, `book`, `laptop`) |
| **Face Auth** | `models/face_auth.py` | InsightFace (`buffalo_l`) | Snapshot (Every 5s) | Calculates facial embedding cosine similarity against baseline photo |
| **Spectacles Detector**| `models/spectacles_detector.py`| ONNX Runtime Model | Snapshot (Every 5s) | Detects eyeglasses with 10-frame majority voting smooth window |

---

## 📁 Directory Structure

```
CodeClusterML/
├── config.py                      # Global thresholds & configuration parameters
├── requirements.txt               # Python package dependencies
├── Dockerfile                     # Container definition for ML worker
├── docker-compose.yml             # Local docker compose setup
├── grpc_service/                  # gRPC Server & Protocol Buffers
│   ├── ml_server.py               # gRPC server entry point (Port 50051)
│   ├── servicer.py                # gRPC service implementation
│   ├── proctor.proto              # Protobuf schema definition
│   ├── proctor_pb2.py             # Compiled Protobuf messages
│   └── proctor_pb2_grpc.py        # Compiled gRPC stubs
├── models/                        # ML Model Logic
│   ├── gaze_tracker.py
│   ├── head_pose.py
│   ├── blink_detector.py
│   ├── face_counter.py
│   ├── speech_detector.py
│   ├── object_detector.py
│   ├── face_auth.py
│   └── spectacles_detector.py
├── pipeline/                      # Threading & Aggregation Pipelines
│   ├── video_monitor.py           # Background video processing thread
│   ├── audio_monitor.py           # Background audio VAD listening thread
│   └── proctoring_pipeline.py     # Multimodal snapshot aggregator
├── utils/                         # Shared Utilities
│   └── image_utils.py             # BGR/RGB conversion & image preprocessing
└── weights/                       # Pre-trained Model Weights
    ├── yolov8n.pt                 # YOLO object detector weights
    └── glasses_hf/                # ONNX spectacles classifier weights
```

---

## ⚙️ Configuration & Thresholds

Thresholds are centralized in `config.py`:

```python
HEAD_YAW_THRESHOLD_DEG      = 30.0   # Degrees yaw shift before triggering head turn alert
FACE_MATCH_THRESHOLD        = 0.72   # Cosine similarity cutoff for InsightFace auth
YOLO_CONFIDENCE_THRESHOLD   = 0.65   # Minimum confidence score for YOLO object detection
BANNED_CLASSES              = ["cell phone", "book", "laptop"]
SPEECH_CONFIDENCE_THRESHOLD = 0.50   # Silero VAD speech detection confidence threshold
SPECS_CONFIDENCE_THRESHOLD  = 0.50   # ONNX spectacles model threshold
SPECS_SMOOTHING_WINDOW      = 10     # Majority voting frame buffer size
GRPC_PORT                   = 50051  # Port for gRPC service
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate
```

### 2. Dependency Installation
```bash
pip install torch==2.5.1 torchaudio==2.5.1
pip install -r requirements.txt
```

### 3. Model Weights Initialization
```bash
# Download YOLOv8n weights into weights/ directory
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt weights/yolov8n.pt
```

### 4. Running the ML Server
Start the gRPC server:
```bash
python grpc_service/ml_server.py
```
The server will log startup confirmation and begin listening on `127.0.0.1:50051`.

---

## 📡 gRPC Interface API

`CodeClusterML` exposes the following RPC endpoints defined in `grpc_service/proctor.proto`:

- `rpc AnalyzeSnapshot (SnapshotRequest) returns (ProctoringResponse)`: Evaluates candidate image frames against YOLO object detector, InsightFace facial authentication, and Spectacles classifier.
- `rpc StreamProctoring (stream FrameChunk) returns (stream ProctoringResponse)`: Real-time bi-directional streaming for telemetry & telemetry processing.

---

## 📜 License

Proprietary and Confidential — **CodeCluster AI Team**.
