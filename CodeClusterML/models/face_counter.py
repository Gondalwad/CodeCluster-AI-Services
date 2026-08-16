import mediapipe as mp

from utils.image_utils import to_rgb


class FaceCounter:
    _STATUS_MAP = {0: "MISSING", 1: "OK"}

    def __init__(self):
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    def predict(self, frame) -> dict:
        rgb     = to_rgb(frame)
        results = self._detector.process(rgb)
        count   = len(results.detections) if results.detections else 0
        status  = self._STATUS_MAP.get(count, "VIOLATION")
        return {
            "faceCount": count,
            "detectedFaceCount": count,
            "faceStatus": status,
        }

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
