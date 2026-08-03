import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import deque
from pathlib import Path

from utils.image_utils import to_rgb

_MODEL_PATH       = Path(__file__).parent.parent / "weights" / "glasses_hf" / "glasses_model.onnx"
_INPUT_SIZE       = (224, 224)
_SMOOTHING_WINDOW = 10
_VOTE_RATIO       = 0.55


class SpectaclesDetector:
    """Detects glasses using a pre-trained ONNX classifier (joze2344/glasses-detector)."""
    # output[0] = glasses logit, output[1] = no-glasses logit

    def __init__(self):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Glasses detector model not found at {_MODEL_PATH}. "
                "Run: python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('joze2344/glasses-detector', 'glasses_model.onnx', local_dir='weights/glasses_hf')\""
            )
        self._sess = ort.InferenceSession(str(_MODEL_PATH), providers=['CPUExecutionProvider'])
        self._inp  = self._sess.get_inputs()[0].name
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._history: deque[bool] = deque(maxlen=_SMOOTHING_WINDOW)

    def _get_face_crop(self, frame) -> np.ndarray | None:
        h, w = frame.shape[:2]
        res = self._face_det.process(to_rgb(frame))
        if not res.detections:
            return None
        bb = res.detections[0].location_data.relative_bounding_box
        x1 = int(max(0, bb.xmin * w))
        y1 = int(max(0, bb.ymin * h))
        x2 = int(min(w, (bb.xmin + bb.width)  * w))
        y2 = int(min(h, (bb.ymin + bb.height) * h))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return frame[y1:y2, x1:x2]

    def _infer(self, face_crop: np.ndarray) -> float:
        resized = cv2.resize(face_crop, _INPUT_SIZE)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob    = rgb.transpose(2, 0, 1)[np.newaxis]
        logits  = self._sess.run(None, {self._inp: blob})[0][0]
        exp     = np.exp(logits - logits.max())
        probs   = exp / exp.sum()
        return float(probs[0])

    def predict(self, frame) -> dict:
        null_result = {
            "specsDetected": False, "confidence": None,
            "glareScore": 0.0, "edgeScore": 0.0,
        }

        face_crop = self._get_face_crop(frame)
        if face_crop is None:
            return null_result

        confidence   = self._infer(face_crop)
        raw_detected = confidence >= 0.5

        self._history.append(raw_detected)
        smoothed = (sum(self._history) / len(self._history)) >= _VOTE_RATIO

        return {
            "specsDetected": smoothed,
            "confidence":    round(confidence, 4),
            "glareScore":    0.0,
            "edgeScore":     0.0,
        }

    def close(self):
        self._face_det.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
