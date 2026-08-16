import cv2
import numpy as np
import mediapipe as mp

from utils.image_utils import to_rgb
from config import HEAD_YAW_THRESHOLD_DEG, HEAD_PITCH_THRESHOLD_DEG

_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0,  170.0, -135.0),
    (225.0,   170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)

_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _camera_matrix(self, h: int, w: int) -> np.ndarray:
        focal = w
        return np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)

    def predict(self, frame) -> dict:
        h, w  = frame.shape[:2]
        rgb   = to_rgb(frame)
        results = self._face_mesh.process(rgb)

        null_result = {
            "yawAngle": None, "pitchAngle": None,
            "rollAngle": None, "isHeadTurned": False, "isNodding": False,
        }

        if not results.multi_face_landmarks:
            return null_result

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in _LANDMARK_IDS
        ], dtype=np.float64)

        camera_matrix = self._camera_matrix(h, w)
        dist_coeffs   = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return null_result

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch_raw = float(angles[0])
        yaw       = float(angles[1])
        roll      = float(angles[2])

        if pitch_raw > 90:
            pitch = pitch_raw - 180.0
        elif pitch_raw < -90:
            pitch = pitch_raw + 180.0
        else:
            pitch = pitch_raw

        return {
            "yawAngle":     round(yaw,   2),
            "pitchAngle":   round(pitch, 2),
            "rollAngle":    round(roll,  2),
            "isHeadTurned": abs(yaw)   > HEAD_YAW_THRESHOLD_DEG,
            "isNodding":    abs(pitch) > HEAD_PITCH_THRESHOLD_DEG,
        }

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
