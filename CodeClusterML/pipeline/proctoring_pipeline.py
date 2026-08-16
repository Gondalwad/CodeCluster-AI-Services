import time
import logging
import threading
import numpy as np

from models.gaze_tracker import GazeTracker
from models.head_pose import HeadPoseEstimator
from models.blink_detector import BlinkDetector
from models.face_counter import FaceCounter
from models.object_detector import ObjectDetector
from models.face_auth import FaceAuthenticator
from models.spectacles_detector import SpectaclesDetector
from pipeline.audio_monitor import AudioMonitor

logger = logging.getLogger(__name__)


def _native(v):
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
    def __init__(self, candidate_id: str = "UNKNOWN"):
        self.candidate_id = candidate_id

        self._process_lock = threading.Lock()
        self._frame_number = 0

        self._gaze = GazeTracker()
        self._head_pose = HeadPoseEstimator()
        self._blink = BlinkDetector()
        self._face_count = FaceCounter()
        self._objects = ObjectDetector()
        self._face_auth = FaceAuthenticator()
        self._specs = SpectaclesDetector()

        self._audio_monitor = AudioMonitor()
        self._running = False

    def register_candidate(self, frame) -> bool:
        return self._face_auth.register(frame)

    def start(self):
        self.reset_session()
        self._audio_monitor.start()
        self._running = True

    def is_running(self):
        return self._running

    def stop(self):
        self._running = False
        self._audio_monitor.stop()

    def close(self):
        self.stop()
        for model in (self._gaze, self._head_pose, self._blink, self._face_count):
            try:
                model.close()
            except Exception:
                pass

    def reset_session(self):
        self._frame_number = 0
        self._audio_monitor.reset()
        self._specs.reset()

    def process_frame(self, frame):
        with self._process_lock:
            self._frame_number += 1

            gaze = self._gaze.predict(frame)
            head = self._head_pose.predict(frame)
            blink = self._blink.predict(frame)
            face_count = self._face_count.predict(frame)
            objects = self._objects.predict(frame)
            specs = self._specs.predict(frame)
            audio = self._audio_monitor.get_latest()

            should_authenticate = (
                self._frame_number % 5 == 0
                and face_count["faceStatus"] == "OK"
                and not head["isHeadTurned"]
                and abs(head["yawAngle"] or 0) < 20
            )

            face_auth = self._face_auth.predict(frame) if should_authenticate else {
                "faceMatched": True, "similarityScore": None
            }

            violations = []

            if head["isHeadTurned"]:
                violations.append("HEAD_TURNED")

            if gaze["gazeDirection"] in ("LEFT", "RIGHT") and not head["isHeadTurned"]:
                violations.append("LOOKING_AWAY")

            if face_count["faceStatus"] == "MISSING":
                violations.append("FACE_MISSING")
            elif face_count["faceStatus"] == "VIOLATION":
                violations.append("MULTIPLE_FACES")

            if face_auth["faceMatched"] is False and face_auth["similarityScore"] is not None:
                violations.append("IDENTITY_MISMATCH")

            for obj in objects["detectedObjects"]:
                violations.append(
                    f"BANNED_OBJECT:{obj['label'].upper().replace(' ', '_')}"
                )

            if audio["isHumanSpeech"]:
                violations.append("SPEECH_DETECTED")

            if specs["specsDetected"]:
                violations.append("SPECTACLES_DETECTED")

            all_violations = list(dict.fromkeys(violations))

            obj_labels = [f"{o['label']}({o['confidence']:.2f})" for o in objects['detectedObjects']]
            logger.info(
                "[FRAME %d] Yaw=%.1f° | HeadTurn=%s | Gaze=%s | Face=%s | "
                "Specs=%s | Speech=%s | Objects=%s | Violations=%s",
                self._frame_number,
                head["yawAngle"] or 0.0,
                head["isHeadTurned"],
                gaze["gazeDirection"],
                face_count["faceStatus"],
                specs["specsDetected"],
                audio["isHumanSpeech"],
                obj_labels if obj_labels else "none",
                all_violations if all_violations else "none",
            )

            return {
                "timestamp": int(time.time()),
                "candidateId": self.candidate_id,
                "systemStatus": "SUCCESS",
                "violations": all_violations,
                "activeViolations": all_violations,
                "continuousViolations": all_violations,
                "snapshotViolations": [],
                "visionAnalysis": {
                    "gazeDirection": gaze.get("gazeDirection", "CENTER"),
                    "isHeadTurned": _native(head.get("isHeadTurned", False)),
                    "headPose": {
                        "yaw": _native(head.get("yawAngle")),
                        "pitch": _native(head.get("pitchAngle")),
                        "roll": _native(head.get("rollAngle")),
                    },
                    "yawAngle": _native(head.get("yawAngle")),
                    "pitchAngle": _native(head.get("pitchAngle")),
                    "eyesClosed": _native(blink.get("eyesClosed", False)),
                    "isBlinking": _native(blink.get("isBlinking", False)),
                    "blinkCount": _native(blink.get("blinkCount", 0)),
                    "faceCount": _native(face_count.get("faceCount", 1)),
                    "detectedFaceCount": _native(face_count.get("detectedFaceCount", 1)),
                    "faceStatus": face_count.get("faceStatus", "OK"),
                    "faceAuth": {
                        "isMatched": _native(face_auth.get("faceMatched", True)),
                        "similarity": _native(face_auth.get("similarityScore")),
                    },
                    "faceMatched": _native(face_auth.get("faceMatched", True)),
                    "similarityScore": _native(face_auth.get("similarityScore")),
                    "detectedObjects": objects.get("detectedObjects", []),
                    "spectacles": {
                        "isDetected": _native(specs.get("specsDetected", False)),
                        "confidence": _native(specs.get("confidence")),
                    },
                    "specsDetected": _native(specs.get("specsDetected", False)),
                    "specsConfidence": _native(specs.get("confidence")),
                },
                "audioAnalysis": {
                    "isHumanSpeech": _native(audio.get("isHumanSpeech", False)),
                    "speechProbability": _native(audio.get("speechProbability", 0.0)),
                    "confidence": _native(audio.get("speechProbability", 0.0)),
                    "decibels": _native(audio.get("decibels", 0.0)),
                },
            }
