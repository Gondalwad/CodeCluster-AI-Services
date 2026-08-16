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
    va = result.get("visionAnalysis", {})
    aa = result.get("audioAnalysis", {})

    detected_objects = [
        proctor_pb2.DetectedObject(
            label=obj.get("label", ""),
            confidence=float(obj.get("confidence", 0.0)),
            bbox=[int(x) for x in obj.get("bbox", [])],
        )
        for obj in va.get("detectedObjects", [])
    ]

    vision = proctor_pb2.VisionAnalysis(
        gaze_direction=va.get("gazeDirection", "CENTER"),
        is_head_turned=bool(va.get("isHeadTurned", False)),
        yaw_angle=float(va.get("yawAngle") or 0.0),
        pitch_angle=float(va.get("pitchAngle") or 0.0),
        face_count=int(va.get("faceCount", 1)),
        face_status=va.get("faceStatus", "OK"),
        face_matched=bool(va.get("faceMatched", True)),
        similarity_score=float(va.get("similarityScore") or 0.0),
        eyes_closed=bool(va.get("eyesClosed", False)),
        blink_count=int(va.get("blinkCount", 0)),
        specs_detected=bool(va.get("specsDetected", False)),
        specs_confidence=float(va.get("specsConfidence") or 0.0),
        detected_objects=detected_objects,
    )

    audio = proctor_pb2.AudioAnalysis(
        is_human_speech=bool(aa.get("isHumanSpeech", False)),
        speech_probability=float(aa.get("speechProbability", 0.0)),
    )

    return proctor_pb2.FrameResponse(
        timestamp=result.get("timestamp", 0),
        candidate_id=result.get("candidateId", ""),
        vision_analysis=vision,
        audio_analysis=audio,
        violations=result.get("violations", []),
        continuous_violations=result.get("continuousViolations", []),
        snapshot_violations=result.get("snapshotViolations", []),
        system_status=result.get("systemStatus", "SUCCESS"),
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
            self._sessions[candidate_id] = ProctoringPipeline(candidate_id=candidate_id)
        else:
            self._sessions[candidate_id].reset_session()

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

        pipeline = self._sessions[candidate_id]
        if pipeline.is_running():
            pipeline.stop()
        pipeline.start()
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
        audio_only = not request.frame_jpeg

        if not audio_only and frame is None:
            return proctor_pb2.FrameResponse(
                candidate_id=candidate_id,
                system_status="ERROR",
                error_message="Failed to decode JPEG frame",
            )

        try:
            if request.audio_chunk:
                self._sessions[candidate_id]._audio_monitor.push_audio(
                    request.audio_chunk)

            if audio_only:
                return proctor_pb2.FrameResponse(
                    candidate_id=candidate_id,
                    system_status="AUDIO_ONLY",
                )

            result = self._sessions[candidate_id].process_frame(frame)
            logger.debug("[%s] violations=%s", _safe_id(
                candidate_id), result.get("violations", []))
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

        pipeline.close()
        logger.info("[%s] Session ended, threads stopped",
                    _safe_id(candidate_id))
        return proctor_pb2.SessionResponse(success=True, message="OK")
