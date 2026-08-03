from collections import deque
import threading
import queue
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
        self._event.set()

    def read(self, timeout: float = 6.0):
        signalled = self._event.wait(timeout=timeout)
        if not signalled:
            return None
        with self._lock:
            self._event.clear()
            return self._frame


_TARGET_FPS     = 10
_FRAME_INTERVAL = 1.0 / _TARGET_FPS

_GAZE_AWAY_FRAMES    = 8
_HEAD_TURNED_FRAMES  = 6
_FACE_MISSING_FRAMES = 10
_MULTI_FACE_FRAMES   = 5
_EYES_CLOSED_FRAMES  = 8

# Require all N yaw samples to exceed threshold before marking head as turned
_YAW_SMOOTH_WINDOW = 3
_YAW_THRESHOLD     = 60.0


class VideoMonitor:
    def __init__(self, frame_buffer: FrameBuffer, violation_queue: queue.Queue):
        self._buf      = frame_buffer
        self._queue    = violation_queue
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True, name="VideoMonitor")

        self._gaze       = GazeTracker()
        self._head_pose  = HeadPoseEstimator()
        self._blink      = BlinkDetector()
        self._face_count = FaceCounter()

        self._gaze_away_count    = 0
        self._head_turned_count  = 0
        self._face_missing_count = 0
        self._multi_face_count   = 0
        self._eyes_closed_count  = 0

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
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        self._gaze.close()
        self._head_pose.close()
        self._blink.close()
        self._face_count.close()

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def reset(self):
        self._blink.reset()
        self._gaze_away_count    = 0
        self._head_turned_count  = 0
        self._face_missing_count = 0
        self._multi_face_count   = 0
        self._eyes_closed_count  = 0
        self._yaw_window.clear()

    def _push(self, violation_type: str, data: dict):
        self._queue.put({
            "type":      violation_type,
            "timestamp": int(time.time()),
            "data":      data,
        })

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

            if gaze["gazeDirection"] in ("LEFT", "RIGHT", "OFF_SCREEN"):
                self._gaze_away_count += 1
                if self._gaze_away_count == _GAZE_AWAY_FRAMES:
                    self._push("LOOKING_AWAY", gaze)
            else:
                self._gaze_away_count = 0

            if smoothed_turned:
                self._head_turned_count += 1
                if self._head_turned_count == _HEAD_TURNED_FRAMES:
                    self._push("HEAD_TURNED", head)
            else:
                self._head_turned_count = 0

            if face_count["faceStatus"] == "MISSING":
                self._face_missing_count += 1
                if self._face_missing_count == _FACE_MISSING_FRAMES:
                    self._push("FACE_MISSING", face_count)
            else:
                self._face_missing_count = 0

            if face_count["faceStatus"] == "VIOLATION":
                self._multi_face_count += 1
                if self._multi_face_count == _MULTI_FACE_FRAMES:
                    self._push("MULTIPLE_FACES", face_count)
            else:
                self._multi_face_count = 0

            if blink["eyesClosed"]:
                self._eyes_closed_count += 1
                if self._eyes_closed_count == _EYES_CLOSED_FRAMES:
                    self._push("EYES_CLOSED", blink)
            else:
                self._eyes_closed_count = 0

            elapsed = time.time() - t0
            sleep_t = _FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
