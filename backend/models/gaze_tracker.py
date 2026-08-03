import mediapipe as mp
import numpy as np

from utils.image_utils import to_rgb

_LEFT_IRIS  = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]
_LEFT_EYE   = [33, 133]
_RIGHT_EYE  = [362, 263]


class GazeTracker:
    """
    Estimates gaze direction from iris position relative to eye corners.
    Uses MediaPipe FaceMesh iris landmarks — no model training needed.
    """

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _iris_ratio(self, landmarks, iris_ids: list, corner_ids: list) -> float:
        iris_x  = np.mean([landmarks[i].x for i in iris_ids])
        left_x  = landmarks[corner_ids[0]].x
        right_x = landmarks[corner_ids[1]].x
        width   = right_x - left_x
        if width == 0:
            return 0.5
        return (iris_x - left_x) / width

    def predict(self, frame) -> dict:
        rgb     = to_rgb(frame)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"gazeDirection": "OFF_SCREEN", "leftIrisRatio": None, "rightIrisRatio": None}

        landmarks = results.multi_face_landmarks[0].landmark

        left_ratio  = self._iris_ratio(landmarks, _LEFT_IRIS,  _LEFT_EYE)
        right_ratio = self._iris_ratio(landmarks, _RIGHT_IRIS, _RIGHT_EYE)
        avg_ratio   = (left_ratio + right_ratio) / 2.0

        if avg_ratio < 0.40:
            direction = "LEFT"
        elif avg_ratio > 0.60:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        return {
            "gazeDirection":  direction,
            "leftIrisRatio":  round(left_ratio,  3),
            "rightIrisRatio": round(right_ratio, 3),
        }

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
