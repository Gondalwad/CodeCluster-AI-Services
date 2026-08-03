import time
import queue
import logging
import threading
import numpy as np

from models.object_detector import ObjectDetector
from models.face_auth import FaceAuthenticator
from models.spectacles_detector import SpectaclesDetector
from pipeline.audio_monitor import AudioMonitor
from pipeline.video_monitor import VideoMonitor, FrameBuffer

logger = logging.getLogger(__name__)


def _native(v):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if v is None:
        return None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


class ProctoringPipeline:
    """
    Aggregates all ML models into a single process_frame() call.
    """

    def __init__(self, candidate_id: str = "UNKNOWN"):
        self.candidate_id = candidate_id

        self._violation_queue = queue.Queue()
        self._frame_buffer = FrameBuffer()
        self._process_lock = threading.Lock()

        self._frame_number = 0

        self._objects = ObjectDetector()
        self._face_auth = FaceAuthenticator()
        self._specs = SpectaclesDetector()

        self._video_monitor = VideoMonitor(
            self._frame_buffer,
            self._violation_queue,
        )

        self._audio_monitor = AudioMonitor(
            self._violation_queue,
        )

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def register_candidate(self, frame) -> bool:
        return self._face_auth.register(frame)

    def start(self):
        self._video_monitor.start()
        self._audio_monitor.start()

    def stop(self):
        self._video_monitor.stop()
        self._audio_monitor.stop()

    def reset_session(self):
        self._video_monitor.reset()

        while not self._violation_queue.empty():
            try:
                self._violation_queue.get_nowait()
            except queue.Empty:
                break

    # ==========================================================
    # Main Snapshot
    # ==========================================================

    def process_frame(self, frame):

        with self._process_lock:
            self._frame_number += 1
            self._frame_buffer.write(frame)

            # ---------------------------------------------
            # Continuous Models (background thread results)
            # ---------------------------------------------

            video = self._video_monitor.get_latest()
            audio = self._audio_monitor.get_latest()

            gaze = video["gaze"]
            head = video["head"]
            blink = video["blink"]
            face_count = video["face_count"]

            objects = self._objects.predict(frame)
            specs = self._specs.predict(frame)

            # ---------------------------------------------
            # Pose Aware Face Authentication (every 5 frames)
            # ---------------------------------------------

            should_authenticate = (
                self._frame_number % 5 == 0
                and face_count["faceStatus"] == "OK"
                and not head["isHeadTurned"]
                and abs(head["yawAngle"] or 0) < 20
            )

            face_auth = self._face_auth.predict(frame) if should_authenticate else {
                "faceMatched": True, "similarityScore": None
            }

            # ---------------------------------------------
            # Continuous Violations
            # ---------------------------------------------

            queued_violations = []
            seen = set()
            while True:
                try:
                    event = self._violation_queue.get_nowait()
                    if event["type"] not in seen:
                        queued_violations.append(event["type"])
                        seen.add(event["type"])
                except queue.Empty:
                    break

            # ---------------------------------------------
            # Snapshot Violations
            # ---------------------------------------------

            snapshot_violations = []

            if head["isHeadTurned"]:
                snapshot_violations.append("HEAD_TURNED")

            if face_count["faceStatus"] == "MISSING":
                snapshot_violations.append("FACE_MISSING")
            elif face_count["faceStatus"] == "VIOLATION":
                snapshot_violations.append("MULTIPLE_FACES")

            if face_auth["faceMatched"] is False and face_auth["similarityScore"] is not None:
                snapshot_violations.append("IDENTITY_MISMATCH")

            for obj in objects["detectedObjects"]:
                snapshot_violations.append(
                    f"BANNED_OBJECT:{obj['label'].upper().replace(' ', '_')}"
                )

            if audio["isHumanSpeech"]:
                snapshot_violations.append("SPEECH_DETECTED")

            if specs["specsDetected"]:
                snapshot_violations.append("SPECTACLES_DETECTED")

            all_violations = list(dict.fromkeys(queued_violations + snapshot_violations))

            # ---------------------------------------------
            # Debug output
            # ---------------------------------------------

            obj_labels = [o['label'] for o in objects['detectedObjects']]
            logger.info(
                "[FRAME %d] Yaw=%.1f° | HeadTurn=%s | Face=%s | "
                "Specs=%s | Speech=%s | Objects=%s | "
                "ContViolations=%s | SnapViolations=%s",
                self._frame_number,
                head["yawAngle"] or 0.0,
                head["isHeadTurned"],
                face_count["faceStatus"],
                specs["specsDetected"],
                audio["isHumanSpeech"],
                obj_labels if obj_labels else "none",
                queued_violations if queued_violations else "none",
                snapshot_violations if snapshot_violations else "none",
            )

            # ---------------------------------------------
            # Response
            # ---------------------------------------------

            return {

                "timestamp": int(time.time()),

                "candidateId": self.candidate_id,

                "visionAnalysis": {

                    "gazeDirection": gaze["gazeDirection"],

                    "isHeadTurned": _native(
                        head["isHeadTurned"]
                    ),

                    "yawAngle": _native(
                        head["yawAngle"]
                    ),

                    "pitchAngle": _native(
                        head["pitchAngle"]
                    ),

                    "faceCount": _native(
                        face_count["faceCount"]
                    ),

                    "faceStatus": face_count["faceStatus"],

                    "faceMatched": _native(
                        face_auth["faceMatched"]
                    ),

                    "similarityScore": _native(
                        face_auth["similarityScore"]
                    ),

                    "detectedObjects": objects["detectedObjects"],

                    "eyesClosed": _native(
                        blink["eyesClosed"]
                    ),

                    "blinkCount": _native(
                        blink["blinkCount"]
                    ),

                    "specsDetected": _native(
                        specs["specsDetected"]
                    ),

                    "specsConfidence": _native(
                        specs["confidence"]
                    ),
                },

                "audioAnalysis": {

                    "isHumanSpeech": _native(
                        audio["isHumanSpeech"]
                    ),

                    "speechProbability": _native(
                        audio["speechProbability"]
                    ),
                },

                "continuousViolations": queued_violations,

                "snapshotViolations": snapshot_violations,

                "violations": all_violations,

                "systemStatus": "SUCCESS",
            }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
