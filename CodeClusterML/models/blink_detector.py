import numpy as np
import mediapipe as mp

from utils.image_utils import to_rgb

_EAR_THRESHOLD = 0.20
_BLINK_FRAMES  = 2

_LEFT_EYE_PTS  = [33, 159, 158, 133, 153, 145]
_RIGHT_EYE_PTS = [362, 386, 385, 263, 380, 374]


class BlinkDetector:
    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._closed_frames = 0
        self._blink_count   = 0

    def _ear(self, landmarks, eye_pts: list) -> float:
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_pts])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    def predict(self, frame) -> dict:
        rgb     = to_rgb(frame)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"eyesClosed": False, "blinkCount": self._blink_count, "earScore": None}

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear  = self._ear(landmarks, _LEFT_EYE_PTS)
        right_ear = self._ear(landmarks, _RIGHT_EYE_PTS)
        avg_ear   = (left_ear + right_ear) / 2.0

        eyes_closed = avg_ear < _EAR_THRESHOLD

        if eyes_closed:
            self._closed_frames += 1
        else:
            if self._closed_frames >= _BLINK_FRAMES:
                self._blink_count += 1
            self._closed_frames = 0

        return {
            "eyesClosed": eyes_closed,
            "isBlinking": eyes_closed,
            "blinkCount": self._blink_count,
            "earScore":   round(avg_ear, 4),
        }

    def reset(self):
        self._closed_frames = 0
        self._blink_count   = 0

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
