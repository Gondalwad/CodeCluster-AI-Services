# CodeCluster AI Proctoring System — Complete Model & Architecture Backup Reference

> **Important**: This document serves as the authoritative technical reference and backup documentation for the CodeCluster ML Proctoring System. Do not alter model code without referencing these exact specifications.

---

## 1. System Architecture Overview

```
 ┌─────────────────┐       WebSocket        ┌─────────────────┐        gRPC          ┌─────────────────────┐
 │ Frontend Client │ ─────────────────────> │ Backend Server  │ ───────────────────> │   ML Server (gRPC)  │
 │ (React / Web)   │ <───────────────────── │ (FastAPI :8000) │ <─────────────────── │  (Python :50051)    │
 └─────────────────┘      Rule Violations   └─────────────────┘    Frame Response    └──────────┬──────────┘
                                                     │                                          │
                                           ┌─────────┴─────────┐                      ┌─────────┴─────────┐
                                           │   Warning Policy  │                      │ 5 Vision Models + │
                                           │  (Consecutive F)  │                      │   Audio Monitor   │
                                           └───────────────────┘                      └───────────────────┘
```

---

## 2. ML Models Inventory & Specifications

### 2.1 Spectacles Detector (`SpectaclesDetector`)
* **File Location**: [`CodeClusterML/models/spectacles_detector.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/spectacles_detector.py)
* **Model Architecture**: ONNX Classifier (`joze2344/glasses-detector`)
* **Weight Path**: `weights/glasses_hf/glasses_model.onnx`
* **Input Resolution**: `224 x 224` RGB image
* **Class Probabilities Index Mapping** (Empirically Verified):
  * **Index 0 (`probs[0]`)** = **WEARING SPECTACLES** (`p_specs`, range `0.85 - 0.98` when wearing glasses)
  * **Index 1 (`probs[1]`)** = **NO SPECTACLES** (`p_no_specs`, range `0.01 - 0.11` when wearing glasses)
* **Pre-processing Requirements**:
  * **1:1 Square Letterboxing**: Face crop must be padded to a 1:1 square aspect ratio before resizing to `224x224` to prevent facial feature compression that creates false glasses frame lines.
  * **Scaling**: `(rgb - 0.0) / 255.0` float conversion.
* **Decision Logic**:
  * `raw_detected = (p_specs > p_no_specs) and (p_specs >= 0.70)`
* **Temporal Smoothing**:
  * `_SMOOTHING_WINDOW = 4`
  * `_VOTE_RATIO = 0.50` (requires $\ge$ 2 out of 4 positive frames)

---

### 2.2 Gaze Tracker (`GazeTracker`)
* **File Location**: [`CodeClusterML/models/gaze_tracker.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/gaze_tracker.py)
* **Technology**: MediaPipe FaceMesh (Iris Refinement Mode, 478 landmarks)
* **Landmarks Used**:
  * Left Iris: `[474, 475, 476, 477]`
  * Right Iris: `[469, 470, 471, 472]`
  * Left Eye Corners: `[33, 133]`
  * Right Eye Corners: `[362, 263]`
* **Threshold Criteria**:
  * `avg_ratio < 0.34` $\rightarrow$ **`LEFT`**
  * `avg_ratio > 0.66` $\rightarrow$ **`RIGHT`**
  * Otherwise $\rightarrow$ **`CENTER`**
* **Trigger Condition**: Returns `"LOOKING_AWAY"` when `gazeDirection` is `LEFT` or `RIGHT` and head is not turned.

---

### 2.3 Head Pose Estimator (`HeadPoseEstimator`)
* **File Location**: [`CodeClusterML/models/head_pose.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/head_pose.py)
* **Technology**: OpenCV `solvePnP` with 6-point 3D Face Model + MediaPipe FaceMesh
* **3D Landmarks Used**: Nose tip (1), Chin (152), Left eye corner (33), Right eye corner (263), Left mouth corner (61), Right mouth corner (291)
* **Thresholds**:
  * `HEAD_YAW_THRESHOLD_DEG = 28.0°`
  * `HEAD_PITCH_THRESHOLD_DEG = 25.0°`
* **Trigger Condition**: Returns `"HEAD_TURNED"` when `abs(yawAngle) > 28.0°`.

---

### 2.4 Object Detector (`ObjectDetector`)
* **File Location**: [`CodeClusterML/models/object_detector.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/object_detector.py)
* **Technology**: YOLOv8n (ONNX / PyTorch model)
* **Image Size**: `320 x 320`
* **Confidence Threshold**: `YOLO_CONFIDENCE_THRESHOLD = 0.40`
* **Banned Classes Monitored**: `cell phone`, `book`, `laptop`, `tablet`, `earphone`, `headphone`
* **Violation Code Mapping**: `BANNED_OBJECT:CELL_PHONE`, `BANNED_OBJECT:BOOK`, etc.

---

### 2.5 Face Counter (`FaceCounter`)
* **File Location**: [`CodeClusterML/models/face_counter.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/face_counter.py)
* **Technology**: MediaPipe Face Detection
* **Rules**:
  * 0 faces detected $\rightarrow$ `"FACE_MISSING"`
  * 1 face detected $\rightarrow$ `"OK"`
  * $\ge$ 2 faces detected $\rightarrow$ `"MULTIPLE_FACES"`

---

### 2.6 Audio Monitor (`AudioMonitor`)
* **File Location**: [`CodeClusterML/pipeline/audio_monitor.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/pipeline/audio_monitor.py)
* **Technology**: Silero VAD (Voice Activity Detection)
* **Sampling Rate**: 16kHz PCM mono audio
* **Policy Note**: Audio speech warnings are currently snoozed/logging-only in the rule engine.

---

## 3. Warning Engine Policy Matrix (`backend/config/warning_policy.py`)

The warning manager tracks consecutive violation frames. A formal warning popup is triggered **only** when a violation persists across the configured consecutive frame threshold:

| Violation Rule | Consecutive Frame Threshold | Approx. Time | Severity |
| :--- | :---: | :---: | :--- |
| **Mobile Phone Detected** | **1 frame** | Instant | High |
| **Looking Away** | **3 frames** | ~1.0s | Medium |
| **Spectacles Detected** | **5 frames** | ~1.5s | Medium |
| **Head Turned** | **8 frames** | ~2.5s | Medium |
| **Face Not Visible** | **2 frames** | ~0.6s | High |
| **Multiple Persons** | **2 frames** | ~0.6s | High |
| **Face Auth Failed** | **10 frames** | ~3.0s | Critical |

### Global Cooldown & Termination Policy:
* **Cooldown Duration**: `20 seconds` global cooldown after any warning fires.
* **Max Warnings**: `3 warnings` maximum before session auto-termination.

---

## 4. Key Gotchas & Backup Troubleshooting

1. **Spectacles Index Inversion**:
   * Always remember: `probs[0]` = WEARING GLASSES, `probs[1]` = NO GLASSES in `joze2344/glasses-detector`. Inverting them causes continuous false positive warnings when not wearing glasses.

2. **Letterbox Padding Importance**:
   * Never resize non-square face crops directly to `224x224` without square padding. Direct resizing squishes eyes into dark horizontal bars, mimicking glasses frames.

3. **RAM Process Cache**:
   * Python holds loaded ONNX models in memory. Whenever modifying model threshold constants, **always restart** `ml_server.py` and `app.py`.

---

## 5. Verification Command Checklist

To verify system integrity after any environment restore or update:

```bash
# 1. Launch ML gRPC Server
CodeClusterML\venv\Scripts\python.exe CodeClusterML\grpc_service\ml_server.py

# 2. Launch Backend Server
backend\venv\Scripts\python.exe backend\app.py
```

---

## 6. Single Container & Kubernetes (K8s) Deployment Roadmap

### 6.1 Current Single-Container Setup (Docker)
* **Root Dockerfile**: [`Dockerfile`](file:///c:/PIYUSH/MyProjects/CodeCluster/Dockerfile)
* **Compose File**: [`docker-compose.yml`](file:///c:/PIYUSH/MyProjects/CodeCluster/docker-compose.yml)
* **Image Name**: `Ai_Worker`
* **Container Name**: `Ai_Worker`
* **Entrypoint**: [`start_services.sh`](file:///c:/PIYUSH/MyProjects/CodeCluster/start_services.sh)
* **Command to Run**: `docker compose up --build`
* **Port Mappings**:
  * `8000:8000` (FastAPI / WebSockets)
  * `50051:50051` (gRPC ML Service)

### 6.2 Kubernetes Scaling Architectural Guidance

#### Question: Is 1 Combined Container (ML + Backend) ideal for scaling in Kubernetes?
* **For Current Phase (Development / MVP)**: **YES, EXCELLENT**.
  * Gives zero-latency IPC over `127.0.0.1:50051` loopback.
  * Extremely easy 1-command deployment for testing.

* **For Future High-Scale Production (Kubernetes Phase)**: **DECOUPLE INTO 2 PODS**.
  1. **Backend Pod (FastAPI / WebSockets)**: I/O-bound. Requires minimal CPU/RAM. Scales horizontally (e.g. 10 replicas) to handle thousands of concurrent WebSocket candidate connections.
  2. **ML Service Pod (gRPC Server)**: Compute/Memory-bound. Requires 2GB+ RAM per replica and optional GPU acceleration. Scales based on active video processing load.
  * **Why Decouple on K8s?**: If combined in 1 Pod, scaling up to handle WebSocket traffic forces K8s to spin up heavy ML model weights on every new Pod, wasting compute and RAM. Decoupling allows K8s to scale Backend Pods independently of ML Pods.

