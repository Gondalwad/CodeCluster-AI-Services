import re
import numpy as np
import logging

import proctor_pb2
import proctor_pb2_grpc

from pipeline.proctoring_pipeline import ProctoringPipeline

logger = logging.getLogger(__name__)

_CANDIDATE_ID_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def _safe_id(candidate_id: str) -> str:
    return re.sub(r'[\r\n\t]', '', candidate_id)


def _validate_id(candidate_id: str) -> bool:
    return bool(_CANDIDATE_ID_RE.match(candidate_id))


def _jpeg_to_frame(jpeg_bytes: bytes) -> np.ndarray:
    import cv2

    if jpeg_bytes is None:
        return None

    try:
        payload = bytes(jpeg_bytes)
    except (TypeError, ValueError):
        return None

    if len(payload) < 8:
        logger.warning(
            "Rejecting JPEG payload: too short (%d bytes)", len(payload))
        return None

    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        logger.warning("Rejecting JPEG payload: empty byte array")
        return None

    try:
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        logger.warning("Rejecting JPEG payload: OpenCV decode failed: %s", exc)
        return None

    if frame is None or frame.size == 0:
        logger.warning(
            "Rejecting JPEG payload: cv2.imdecode returned empty frame")
        return None

    return frame


def _build_frame_response(result: dict) -> proctor_pb2.FrameResponse:
    va = result["visionAnalysis"]
    aa = result["audioAnalysis"]

    detected_objects = [
        proctor_pb2.DetectedObject(
            label=obj["label"],
            confidence=float(obj["confidence"]),
            bbox=[int(x) for x in obj["bbox"]],
        )
        for obj in va["detectedObjects"]
    ]

    vision = proctor_pb2.VisionAnalysis(
        gaze_direction=va["gazeDirection"],
        is_head_turned=bool(va["isHeadTurned"]),
        yaw_angle=float(va["yawAngle"] or 0.0),
        pitch_angle=float(va["pitchAngle"] or 0.0),
        face_count=int(va["faceCount"]),
        face_status=va["faceStatus"],
        face_matched=bool(va["faceMatched"]),
        similarity_score=float(va["similarityScore"] or 0.0),
        eyes_closed=bool(va["eyesClosed"]),
        blink_count=int(va["blinkCount"]),
        specs_detected=bool(va["specsDetected"]),
        specs_confidence=float(va["specsConfidence"] or 0.0),
        detected_objects=detected_objects,
    )

    audio = proctor_pb2.AudioAnalysis(
        is_human_speech=bool(aa["isHumanSpeech"]),
        speech_probability=float(aa["speechProbability"]),
    )

    return proctor_pb2.FrameResponse(
        timestamp=result["timestamp"],
        candidate_id=result["candidateId"],
        vision_analysis=vision,
        audio_analysis=audio,
        violations=result["violations"],
        continuous_violations=result["continuousViolations"],
        snapshot_violations=result["snapshotViolations"],
        system_status=result["systemStatus"],
    )


class ProctoringServicer(proctor_pb2_grpc.ProctoringServiceServicer):
    """One ProctoringPipeline instance per active candidate session."""

    def __init__(self):
        self._sessions: dict[str, ProctoringPipeline] = {}

    def RegisterCandidate(self, request, context):
        candidate_id = request.candidate_id
        if not _validate_id(candidate_id):
            return proctor_pb2.RegisterResponse(success=False, message="Invalid candidate_id format")
        logger.info("[%s] RegisterCandidate called", _safe_id(candidate_id))

        frame = _jpeg_to_frame(request.face_image)
        if frame is None:
            return proctor_pb2.RegisterResponse(success=False, message="Invalid JPEG image")

        if candidate_id not in self._sessions:
            self._sessions[candidate_id] = ProctoringPipeline(
                candidate_id=candidate_id)

        ok = self._sessions[candidate_id].register_candidate(frame)
        if not ok:
            return proctor_pb2.RegisterResponse(success=False, message="No face detected in registration image")

        logger.info("[%s] Face registered successfully",
                    _safe_id(candidate_id))
        return proctor_pb2.RegisterResponse(success=True, message="OK")

    def StartSession(self, request, context):
        candidate_id = request.candidate_id
        if not _validate_id(candidate_id):
            return proctor_pb2.SessionResponse(success=False, message="Invalid candidate_id format")
        logger.info("[%s] StartSession called", _safe_id(candidate_id))

        if candidate_id not in self._sessions:
            return proctor_pb2.SessionResponse(success=False, message="Candidate not registered")

        self._sessions[candidate_id].start()
        logger.info("[%s] Monitoring threads started", _safe_id(candidate_id))
        return proctor_pb2.SessionResponse(success=True, message="OK")

    def AnalyzeFrame(self, request, context):
        candidate_id = request.candidate_id
        if not _validate_id(candidate_id):
            return proctor_pb2.FrameResponse(
                candidate_id=candidate_id,
                system_status="ERROR",
                error_message="Invalid candidate_id format",
            )

        if candidate_id not in self._sessions:
            return proctor_pb2.FrameResponse(
                candidate_id=candidate_id,
                system_status="ERROR",
                error_message="No active session for this candidate",
            )

        frame = _jpeg_to_frame(request.frame_jpeg)
        if frame is None:
            return proctor_pb2.FrameResponse(
                candidate_id=candidate_id,
                system_status="ERROR",
                error_message="Failed to decode JPEG frame",
            )

        try:
            result = self._sessions[candidate_id].process_frame(frame)
            if request.audio_chunk:
                self._sessions[candidate_id]._audio_monitor.push_audio(
                    request.audio_chunk)
            logger.debug("[%s] violations=%s", _safe_id(
                candidate_id), result["violations"])
            return _build_frame_response(result)
        except Exception as e:
            logger.exception("[%s] AnalyzeFrame error: %s",
                             _safe_id(candidate_id), e)
            return proctor_pb2.FrameResponse(
                candidate_id=candidate_id,
                system_status="ERROR",
                error_message=str(e),
            )

    def EndSession(self, request, context):
        candidate_id = request.candidate_id
        if not _validate_id(candidate_id):
            return proctor_pb2.SessionResponse(success=False, message="Invalid candidate_id format")
        logger.info("[%s] EndSession called", _safe_id(candidate_id))

        pipeline = self._sessions.pop(candidate_id, None)
        if pipeline is None:
            return proctor_pb2.SessionResponse(success=False, message="No active session found")

        pipeline.stop()
        logger.info("[%s] Session ended, threads stopped",
                    _safe_id(candidate_id))
        return proctor_pb2.SessionResponse(success=True, message="OK")
