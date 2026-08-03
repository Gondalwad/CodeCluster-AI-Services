import cv2
import numpy as np


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw JPEG/PNG bytes into a BGR numpy array."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image bytes.")
    return frame


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB (required by MediaPipe)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height))
