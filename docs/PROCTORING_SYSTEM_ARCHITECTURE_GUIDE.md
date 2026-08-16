# CodeCluster AI Proctoring System — Master Architecture & Data Flow Guide

This document contains the complete technical specification, folder structure, end-to-end data flow, model logic, thresholds, and warning engine policies for the CodeCluster AI Proctoring System.

---

## 1. Focused Folder Structure (ML, Backend, Proctoring Frontend)

```text
CodeCluster/
├── CodeClusterML/                       # Machine Learning gRPC Microservice (Port 50051)
│   ├── config.py                        # Centralized ML model thresholds and env defaults
│   ├── grpc_service/
│   │   ├── ml_server.py                 # gRPC server entry point (Port 50051)
│   │   ├── servicer.py                  # gRPC servicer implementation (RegisterCandidate, AnalyzeFrame, EndSession)
│   │   ├── proctor.proto                # Protobuf protocol schema definition
│   │   ├── proctor_pb2.py               # Generated Protobuf classes
│   │   └── proctor_pb2_grpc.py          # Generated gRPC stubs
│   ├── models/
│   │   ├── spectacles_detector.py       # ONNX Spectacles Detector (Index 0 = WEARING GLASSES, 1:1 Letterboxing)
│   │   ├── gaze_tracker.py              # MediaPipe Iris Gaze Tracker (Left: <0.28, Right: >0.72)
│   │   ├── head_pose.py                 # OpenCV solvePnP Head Pose Estimator (Yaw threshold: 28.0°)
│   │   ├── object_detector.py           # YOLOv8n Banned Object Detector (Mobile phone, book, laptop, etc.)
│   │   ├── face_counter.py              # MediaPipe Face Counter (MISSING / OK / MULTIPLE_FACES)
│   │   ├── face_auth.py                 # InsightFace Face Authenticator (buffalo_l embeddings)
│   │   ├── blink_detector.py            # MediaPipe EAR Blink & Eye Closure Tracker
│   │   └── speech_detector.py           # Silero VAD Human Speech Detector
│   ├── pipeline/
│   │   ├── proctoring_pipeline.py       # Master synchronous pipeline running all vision models inline per frame
│   │   └── audio_monitor.py             # Asynchronous thread processing incoming PCM audio chunks
│   ├── utils/
│   │   └── image_utils.py               # RGB conversion and frame pre-processing utilities
│   └── weights/                         # Model weight files (YOLO pt, ONNX models, InsightFace cache)
│
├── backend/                             # FastAPI Web & WebSocket Backend Service (Port 8000)
│   ├── app.py                           # FastAPI application entry point
│   ├── api/
│   │   ├── websocket.py                 # Live WebSocket endpoint (/ws/proctor/{candidate_id})
│   │   ├── proctor.py                   # HTTP API routes for session lifecycle
│   │   └── routes.py                    # Master API router configuration
│   ├── clients/
│   │   └── ai_client.py                 # Asynchronous gRPC client connecting to ML Server on 127.0.0.1:50051
│   ├── config/
│   │   ├── warning_policy.py            # Consecutive frame thresholds, max warnings (3), cooldown (20s)
│   │   └── settings.py                  # Environment variable configuration
│   ├── services/
│   │   ├── proctor_service.py           # Manages candidate sessions and routes frames to gRPC client
│   │   ├── rule_engine.py               # Maps raw ML violation codes to user-friendly violation labels
│   │   └── warning_manager.py           # Tracks consecutive frame counters, misses, cooldowns, and warning triggers
│   └── protos/                          # Protobuf Python stubs used by gRPC client
│
└── CodeCluster/client/src/             # Proctoring-Related Frontend Components
    ├── pages/
    │   └── Exam/                        # Exam Pages
    │       ├── ExamInstructions.jsx     # Pre-exam guidelines and consent UI
    │       ├── ExamSession.jsx          # Container wrapping ProctoringSession and exam content
    │       └── index.js                 # Export index
    ├── proctoring/                      # Core Proctoring Library
    │   ├── ProctoringSession.jsx        # Root session wrapper (Renders camera preview, warning popup, termination overlay)
    │   ├── config.js                    # Frontend WebSocket URL and webcam resolution config
    │   ├── proctorEvents.js             # Global EventBus string constants
    │   ├── audio/
    │   │   ├── audioCapture.js          # Captures 16kHz PCM mono audio chunks
    │   │   ├── audioStreamer.js         # Streams audio chunks to WebSocket
    │   │   └── microphoneManager.js     # Manages navigator.mediaDevices.getUserMedia audio stream
    │   ├── camera/
    │   │   ├── cameraManager.js         # Manages webcam video stream attached to HTML5 video element
    │   │   ├── frameCapturer.js         # Draws video frames to offscreen canvas & encodes JPEG blobs
    │   │   └── frameScheduler.js        # Schedules frame capture at 3 FPS
    │   ├── components/
    │   │   ├── CameraPreview.jsx        # Draggable picture-in-picture webcam overlay with live FPS & warning counters
    │   │   └── WarningPopup.jsx         # Non-intrusive top-center warning banner with progress bar
    │   ├── context/
    │   │   └── ProctoringContext.jsx    # React Context exposing proctoring state to children
    │   ├── hooks/
    │   │   ├── useProctoring.js         # Master hook orchestrating camera, audio, scheduler, and WebSocket
    │   │   ├── useProctoringLogger.js   # EventBus logger for real-time state debugging
    │   │   └── useWebSocket.js          # Reconnecting WebSocket client sending JPEG frames and receiving violations
    │   └── tests/
    │       └── test_websocket.js        # Standalone WebSocket connection test helper
    └── proctoring-test/
        └── main.jsx                     # Standalone proctoring test harness page
```

---

## 2. End-to-End Data Flow (Frontend ↔ Backend ↔ ML Server)

```
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   FRONTEND (React Client)                                       │
  │  1. Candidate opens ExamSession.jsx -> Click "Start Exam"                                       │
  │  2. cameraManager captures 640x480 webcam stream                                                │
  │  3. frameScheduler grabs canvas frame -> JPEG blob at 3 FPS                                    │
  │  4. audioStreamer grabs 16kHz PCM audio chunk every 250ms                                      │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │
                                                 │ WebSocket Binary Payload (JPEG Frame / Audio Chunk)
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   BACKEND (FastAPI :8000)                                       │
  │  5. api/websocket.py receives frame at /ws/proctor/{candidate_id}                              │
  │  6. Passes binary frame to proctor_service.py -> calls ai_client.py                             │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │
                                                 │ gRPC Protobuf Request (AnalyzeFrame) over 127.0.0.1:50051
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                  ML SERVER (Python gRPC :50051)                                 │
  │  7. servicer.py receives gRPC request -> passes frame to proctoring_pipeline.py               │
  │  8. ALL 5 Vision Models run INLINE synchronously on exact frame:                               │
  │     • SpectaclesDetector -> ONNX (1:1 Letterboxed crop)                                        │
  │     • GazeTracker       -> MediaPipe Iris Landmarks                                            │
  │     • HeadPoseEstimator -> OpenCV solvePnP                                                       │
  │     • ObjectDetector    -> YOLOv8n Banned Classes                                              │
  │     • FaceCounter       -> MediaPipe Face Detection                                            │
  │     • FaceAuthenticator -> InsightFace (every 5 frames)                                        │
  │     • AudioMonitor      -> Silero VAD                                                          │
  │  9. Pipeline compiles violation list e.g. ["SPECTACLES_DETECTED", "LOOKING_AWAY"]                │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │
                                                 │ gRPC Protobuf Response (FrameResponse)
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   BACKEND (FastAPI :8000)                                       │
  │ 10. ai_client.py receives gRPC response -> passes to rule_engine.py                             │
  │ 11. rule_engine.py maps raw codes to labels e.g. ["Spectacles detected", "Looking away"]       │
  │ 12. warning_manager.py updates consecutive frame counters:                                     │
  │     • Increments consecutive frames for active violations                                       │
  │     • Checks threshold (e.g., Spectacles: 5 frames)                                             │
  │     • If threshold met -> Increments warning_count (1/3 -> 2/3) & triggers 20s global cooldown │
  │ 13. websocket.py sends JSON payload back to Frontend                                           │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │
                                                 │ WebSocket JSON Message
                                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   FRONTEND (React Client)                                       │
  │ 14. useWebSocket.js receives JSON message -> dispatches PROCTOR_EVENTS on window               │
  │ 15. WarningPopup.jsx captures event -> displays top-center warning banner with progress bar     │
  │ 16. CameraPreview.jsx updates live debug stats (Warnings count, cooldown remaining)            │
  │ 17. If warning_count >= 3 -> ProctoringSession.jsx renders "Exam Terminated" modal              │
  └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ML Model Catalog, Detection Logic, and Thresholds

### 3.1 Spectacles Detector (`SpectaclesDetector`)
* **File Location**: [`CodeClusterML/models/spectacles_detector.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/spectacles_detector.py)
* **Model Architecture**: ONNX Classifier (`joze2344/glasses-detector`)
* **Weight File**: `weights/glasses_hf/glasses_model.onnx`
* **Input Resolution**: `224 x 224` RGB image
* **Class Index Mapping**:
  * **Index 0 (`probs[0]`)** = **WEARING GLASSES** (`p_specs`)
  * **Index 1 (`probs[1]`)** = **NO GLASSES** (`p_no_specs`)
* **Pre-processing Requirement (1:1 Square Letterboxing)**:
  * Non-square face bounding boxes are padded with zero-valued black pixels to a 1:1 square ratio before resizing to `224x224`. This prevents horizontal feature compression that creates dark eye lines mimicking glasses frames.
* **Decision Logic**:
  * `raw_detected = (p_specs > p_no_specs) and (p_specs >= 0.70)`
* **Temporal Smoothing**:
  * Rolling window `_SMOOTHING_WINDOW = 4`
  * Vote ratio `_VOTE_RATIO = 0.50` (requires $\ge$ 2 out of 4 positive frames)

---

### 3.2 Gaze Tracker (`GazeTracker`)
* **File Location**: [`CodeClusterML/models/gaze_tracker.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/gaze_tracker.py)
* **Technology**: MediaPipe FaceMesh (478 3D landmarks with iris refinement)
* **Iris Landmarks Used**:
  * Left Iris: `[474, 475, 476, 477]`, Left Eye Corners: `[33, 133]`
  * Right Iris: `[469, 470, 471, 472]`, Right Eye Corners: `[362, 263]`
* **Iris Ratio Calculation**:
  $$\text{Ratio} = \frac{\text{Iris\_X} - \text{Corner\_Left\_X}}{\text{Corner\_Right\_X} - \text{Corner\_Left\_X}}$$
* **Threshold Criteria**:
  * `avg_ratio < 0.28` $\rightarrow$ **`LEFT`**
  * `avg_ratio > 0.72` $\rightarrow$ **`RIGHT`**
  * Otherwise $\rightarrow$ **`CENTER`**
* **Trigger Condition**: Returns `"LOOKING_AWAY"` when `gazeDirection` is `LEFT` or `RIGHT` and head is not turned.

---

### 3.3 Head Pose Estimator (`HeadPoseEstimator`)
* **File Location**: [`CodeClusterML/models/head_pose.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/head_pose.py)
* **Technology**: OpenCV `solvePnP` with 6-point 3D Face Model + MediaPipe FaceMesh
* **3D Reference Landmarks**: Nose tip (1), Chin (152), Left eye corner (33), Right eye corner (263), Left mouth corner (61), Right mouth corner (291)
* **Threshold Constants**:
  * `HEAD_YAW_THRESHOLD_DEG = 28.0°`
  * `HEAD_PITCH_THRESHOLD_DEG = 15.0°`
* **Trigger Condition**: Returns `"HEAD_TURNED"` when `abs(yawAngle) > 28.0°`.

---

### 3.4 Object Detector (`ObjectDetector`)
* **File Location**: [`CodeClusterML/models/object_detector.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/object_detector.py)
* **Technology**: YOLOv8n (ONNX / PyTorch model)
* **Input Resolution**: `320 x 320`
* **Confidence Threshold**: `YOLO_CONFIDENCE_THRESHOLD = 0.10` (Rule engine applies instant warning for phones)
* **Banned Classes Monitored**: `cell phone`, `mobile phone`, `book`, `laptop`, `tablet`, `remote`
* **Violation Code Mapping**: `BANNED_OBJECT:CELL_PHONE`, `BANNED_OBJECT:BOOK`, etc.

---

### 3.5 Face Counter (`FaceCounter`)
* **File Location**: [`CodeClusterML/models/face_counter.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/face_counter.py)
* **Technology**: MediaPipe Face Detection (`model_selection=0`, `min_detection_confidence=0.5`)
* **Rules**:
  * `0` faces detected $\rightarrow$ `"FACE_MISSING"`
  * `1` face detected $\rightarrow$ `"OK"`
  * $\ge 2$ faces detected $\rightarrow$ `"MULTIPLE_FACES"`

---

### 3.6 Face Authenticator (`FaceAuthenticator`)
* **File Location**: [`CodeClusterML/models/face_auth.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/models/face_auth.py)
* **Technology**: InsightFace (`buffalo_l` ResNet-50 embedding model)
* **Execution Interval**: Runs every 5 frames when face is centered and head is not turned (`abs(yaw) < 20°`)
* **Similarity Metric**: Cosine similarity between candidate registration embedding and live frame embedding.
* **Match Threshold**: `FACE_MATCH_THRESHOLD = 0.55` (returns `"IDENTITY_MISMATCH"` if `< 0.55`).

---

### 3.7 Audio Monitor (`AudioMonitor`)
* **File Location**: [`CodeClusterML/pipeline/audio_monitor.py`](file:///c:/PIYUSH/MyProjects/CodeCluster/CodeClusterML/pipeline/audio_monitor.py)
* **Technology**: Silero VAD (Voice Activity Detection)
* **Input**: 16kHz PCM mono audio chunks (250ms duration)
* **Confidence Threshold**: `SPEECH_CONFIDENCE_THRESHOLD = 0.50`

---

## 4. Backend Warning Engine & Policy Matrix (`backend/config/warning_policy.py`)

The Backend `WarningManager` tracks consecutive violation frames for each rule. A formal warning is triggered **only** when a violation persists for the required number of consecutive frames:

| Violation Rule | Consecutive Frame Threshold | Approx. Time (at ~3 FPS) | Severity |
| :--- | :---: | :---: | :--- |
| **Mobile Phone Detected** | **1 frame** | Instant | High |
| **Face Not Visible** | **2 frames** | ~0.6s | High |
| **Multiple Persons Detected** | **2 frames** | ~0.6s | High |
| **Book / Laptop / Tablet / Headphones** | **2 frames** | ~0.6s | Medium |
| **Looking Away** | **5 frames** | ~1.5s - 2.0s | Medium |
| **Spectacles Detected** | **5 frames** | ~1.5s - 2.0s | Medium |
| **Head Turned** | **8 frames** | ~2.5s - 3.0s | Medium |
| **Face Authentication Failed** | **10 frames** | ~3.0s - 3.5s | Critical |

### Global Cooldown & Warning Mechanics:
* **Max Warnings Allowed**: `MAX_WARNINGS = 3` (3rd warning triggers exam termination overlay).
* **Global Cooldown (`WARNING_COOLDOWN = 20s`)**: After ANY warning fires, all detection counters freeze for 20 seconds. No new warnings can fire during cooldown.
* **Clean Frame Reset (`MAX_MISSES = 5`)**: If a candidate fixes a violation for 5 consecutive clean frames, the frame counter for that violation resets to 0.

---

## 5. Frontend Component Architecture & Event System

### 5.1 Key Frontend Modules
* **`useProctoring.js`**: Master React hook initializing camera, microphone, frame scheduler, audio streamer, and WebSocket connection.
* **`ProctoringSession.jsx`**: Root session container component. Manages `examTerminated` modal state and wraps exam content.
* **`CameraPreview.jsx`**: Draggable, picture-in-picture webcam view displaying live video feed, active violations, warning count (`X / 3`), and cooldown countdown.
* **`WarningPopup.jsx`**: Top-center warning banner displaying human-readable violation instructions and a 3-second animated progress bar.

### 5.2 Global EventBus Architecture (`proctorEvents.js`)
All proctoring modules communicate asynchronously using custom window events:

```javascript
window.dispatchEvent(new CustomEvent(PROCTOR_EVENTS.WARNING_RECEIVED, { detail: data }));
```

* **`WEBSOCKET_CONNECTED`**: Fired when WebSocket successfully handshakes with Backend.
* **`SNAPSHOT_SENT`**: Fired when JPEG frame is sent to Backend.
* **`WARNING_UPDATED`**: Fired on every incoming WebSocket response to update debug UI counters.
* **`WARNING_RECEIVED`**: Fired when a formal warning is triggered (displays `WarningPopup`).
* **`EXAM_TERMINATED`**: Fired on 3rd warning (stops camera/audio streams and displays `ExamTerminated` overlay).
