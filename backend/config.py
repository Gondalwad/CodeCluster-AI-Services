import os
from dotenv import load_dotenv

load_dotenv()

def _float(key, default):
    return float(os.getenv(key, default))

def _str(key, default):
    return os.getenv(key, default)

# Gaze & Head Pose
GAZE_OFF_SCREEN_THRESHOLD_SEC = _float("GAZE_OFF_SCREEN_THRESHOLD_SEC", 3.0)
HEAD_YAW_THRESHOLD_DEG        = _float("HEAD_YAW_THRESHOLD_DEG", 65.0)
HEAD_PITCH_THRESHOLD_DEG      = _float("HEAD_PITCH_THRESHOLD_DEG", 30.0)

# Object Detection
YOLO_CONFIDENCE_THRESHOLD = _float("YOLO_CONFIDENCE_THRESHOLD", 0.65)
YOLO_MODEL_PATH           = _str("YOLO_MODEL_PATH", "weights/yolov8n.pt")
BANNED_CLASSES            = ["cell phone", "book", "laptop"]

# Spectacles Detection
SPECS_MODEL_PATH          = _str("SPECS_MODEL_PATH", "weights/glasses_hf/glasses_model.onnx")
SPECS_CONFIDENCE_THRESHOLD = _float("SPECS_CONFIDENCE_THRESHOLD", 0.50)
SPECS_SMOOTHING_WINDOW    = int(os.getenv("SPECS_SMOOTHING_WINDOW", 10))

# Face Auth
FACE_MATCH_THRESHOLD = _float("FACE_MATCH_THRESHOLD", 0.55)

# Face Counter
FACE_MISSING_THRESHOLD_SEC = _float("FACE_MISSING_THRESHOLD_SEC", 5.0)

# Speech Detection
SPEECH_CONFIDENCE_THRESHOLD = _float("SPEECH_CONFIDENCE_THRESHOLD", 0.50)

# gRPC
GRPC_PORT = int(os.getenv("GRPC_PORT", 50051))
