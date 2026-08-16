from collections import deque
import threading
import time
import numpy as np

from models.gaze_tracker   import GazeTracker
from models.head_pose      import HeadPoseEstimator
from models.blink_detector import BlinkDetector
from models.face_counter   import FaceCounter


class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()
        self._event = threading.Event()

    def write(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
        self._event.set()  # always set — reader picks up latest

    def read(self, timeout: float = 6.0):
        signalled = self._event.wait(timeout=timeout)
        if not signalled:
            return None
        self._event.clear()
        with self._lock:
            return self._frame


from config import HEAD_YAW_THRESHOLD_DEG

_TARGET_FPS     = 10
_FRAME_INTERVAL = 1.0 / _TARGET_FPS

# Require all N yaw samples to exceed threshold before marking head as turned
_YAW_SMOOTH_WINDOW = 2
_YAW_THRESHOLD     = HEAD_YAW_THRESHOLD_DEG


class VideoMonitor:
    def __init__(self, frame_buffer: FrameBuffer):
        self._buf      = frame_buffer
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True, name="VideoMonitor")

        self._gaze       = GazeTracker()
        self._head_pose  = HeadPoseEstimator()
        self._blink      = BlinkDetector()
        self._face_count = FaceCounter()

        # Rolling yaw window — smooths out MediaPipe instability at extreme angles
        self._yaw_window = deque(maxlen=_YAW_SMOOTH_WINDOW)

        self._lock   = threading.Lock()
        self._latest = {
            "gaze":       {"gazeDirection": "CENTER", "leftIrisRatio": None, "rightIrisRatio": None},
            "head":       {"yawAngle": None, "pitchAngle": None, "rollAngle": None, "isHeadTurned": False, "isNodding": False},
            "blink":      {"eyesClosed": False, "blinkCount": 0, "earScore": None},
            "face_count": {"faceCount": 0, "faceStatus": "MISSING"},
        }

    def start(self):
        self._stop_evt.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="VideoMonitor")
            self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        for model in (self._gaze, self._head_pose, self._blink, self._face_count):
            try:
                model.close()
            except Exception:
                pass

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def reset(self):
        self._blink.reset()
        self._yaw_window.clear()
        with self._lock:
            self._latest = {
                "gaze":       {"gazeDirection": "CENTER", "leftIrisRatio": None, "rightIrisRatio": None},
                "head":       {"yawAngle": None, "pitchAngle": None, "rollAngle": None, "isHeadTurned": False, "isNodding": False},
                "blink":      {"eyesClosed": False, "blinkCount": 0, "earScore": None},
                "face_count": {"faceCount": 0, "faceStatus": "MISSING"},
            }

    def _run(self):
        while not self._stop_evt.is_set():
            frame = self._buf.read(timeout=6.0)
            if frame is None:
                continue

            t0 = time.time()

            gaze       = self._gaze.predict(frame)
            head       = self._head_pose.predict(frame)
            blink      = self._blink.predict(frame)
            face_count = self._face_count.predict(frame)

            # Smooth yaw over last N frames to eliminate per-frame flipping
            yaw = head["yawAngle"]
            if yaw is not None:
                self._yaw_window.append(abs(yaw))

            smoothed_turned = (
                len(self._yaw_window) == _YAW_SMOOTH_WINDOW
                and all(y > _YAW_THRESHOLD for y in self._yaw_window)
            )
            head = {**head, "isHeadTurned": smoothed_turned}

            with self._lock:
                self._latest["gaze"]       = gaze
                self._latest["head"]       = head
                self._latest["blink"]      = blink
                self._latest["face_count"] = face_count

            elapsed = time.time() - t0
            sleep_t = _FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
