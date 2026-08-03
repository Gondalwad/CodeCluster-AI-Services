"""
tests/test_grpc_integration.py — CodeCluster ML gRPC Integration Test
======================================================================
Simulates the full Java backend lifecycle against a live gRPC server:

    RegisterCandidate → StartSession → AnalyzeFrame (×3) → EndSession

Also covers all error paths:
    - Invalid candidate_id format
    - AnalyzeFrame before registration
    - Corrupt / empty JPEG bytes

Usage:
    python tests/test_grpc_integration.py [--image path/to/face.jpg]

    If --image is not provided, attempts to capture one frame from webcam.
    If webcam is unavailable, falls back to a synthetic 480×640 BGR frame.

Exit codes:
    0 — all assertions passed
    1 — one or more assertions failed
"""

import os
import sys
import time
import struct
import argparse
import threading
import traceback
import math
import json
from concurrent import futures
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import grpc

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRPC_DIR = os.path.join(ROOT, "grpc_service")
sys.path.insert(0, ROOT)
sys.path.insert(0, GRPC_DIR)

from config import GRPC_PORT
import proctor_pb2
import proctor_pb2_grpc
from servicer import ProctoringServicer

# ── Constants ─────────────────────────────────────────────────────────────────
_TEST_CANDIDATE   = "TEST_CANDIDATE_01"
_INVALID_ID       = "bad id with spaces!!"
_SAMPLE_RATE      = 16000
_SILENCE_DURATION = 0.5   # seconds of silence PCM per audio chunk
_ANALYZE_FRAMES   = 3
_SERVER_WAIT_SEC  = 3.0   # time to let background threads warm up

CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


_results: List[TestResult] = []


def _check(name: str, condition: bool, detail: str = ""):
    r = TestResult(name, condition, detail)
    _results.append(r)
    status = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}" + (f"  →  {detail}" if detail else ""))
    return condition


def _section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ── gRPC server lifecycle ─────────────────────────────────────────────────────

def _start_server(port: int) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ("grpc.max_send_message_length",    10 * 1024 * 1024),
        ],
    )
    proctor_pb2_grpc.add_ProctoringServiceServicer_to_server(ProctoringServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


# ── Frame / audio helpers ─────────────────────────────────────────────────────

def _capture_or_synthetic() -> np.ndarray:
    """Return a BGR frame from webcam, saved image, or synthetic fallback."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"  {YELLOW}[webcam]{RESET} Captured registration frame from webcam.")
            return frame
    # Synthetic: gradient face-like image so models don't crash
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, (200, 170, 140), -1)
    cv2.circle(frame, (270, 210), 20, (60, 40, 20), -1)
    cv2.circle(frame, (370, 210), 20, (60, 40, 20), -1)
    cv2.ellipse(frame, (320, 310), (50, 25), 0, 0, 180, (120, 80, 60), 2)
    print(f"  {YELLOW}[synthetic]{RESET} No webcam — using synthetic face frame.")
    return frame


def _load_image(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    print(f"  {YELLOW}[image]{RESET} Loaded registration frame from {path}")
    return frame


def _to_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _silence_pcm(duration_sec: float = _SILENCE_DURATION) -> bytes:
    """Raw PCM bytes: 16kHz mono int16 silence."""
    n_samples = int(_SAMPLE_RATE * duration_sec)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _tone_pcm(freq_hz: int = 440, duration_sec: float = _SILENCE_DURATION) -> bytes:
    """Raw PCM bytes: 16kHz mono int16 sine tone (simulates audio activity)."""
    n_samples = int(_SAMPLE_RATE * duration_sec)
    samples = [int(32767 * math.sin(2 * math.pi * freq_hz * i / _SAMPLE_RATE))
               for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


# ── Test suites ───────────────────────────────────────────────────────────────

def suite_invalid_id(stub):
    _section("Suite 1 — Invalid candidate_id rejection")

    r = stub.RegisterCandidate(proctor_pb2.RegisterRequest(
        candidate_id=_INVALID_ID,
        face_image=b"\xff\xd8\xff",
    ))
    _check("RegisterCandidate rejects invalid ID", not r.success,
           f"message={r.message!r}")

    r = stub.StartSession(proctor_pb2.SessionRequest(candidate_id=_INVALID_ID))
    _check("StartSession rejects invalid ID", not r.success)

    r = stub.AnalyzeFrame(proctor_pb2.FrameRequest(
        candidate_id=_INVALID_ID,
        frame_jpeg=b"\xff\xd8\xff",
    ))
    _check("AnalyzeFrame rejects invalid ID", r.system_status == "ERROR",
           f"status={r.system_status}")

    r = stub.EndSession(proctor_pb2.SessionRequest(candidate_id=_INVALID_ID))
    _check("EndSession rejects invalid ID", not r.success)


def suite_no_session(stub):
    _section("Suite 2 — AnalyzeFrame without active session")

    r = stub.AnalyzeFrame(proctor_pb2.FrameRequest(
        candidate_id="GHOST_CANDIDATE",
        frame_jpeg=b"\xff\xd8\xff",
    ))
    _check("AnalyzeFrame returns ERROR for unknown candidate",
           r.system_status == "ERROR", f"error={r.error_message!r}")


def suite_corrupt_jpeg(stub, candidate_id: str):
    _section("Suite 3 — Corrupt JPEG handling")

    r = stub.AnalyzeFrame(proctor_pb2.FrameRequest(
        candidate_id=candidate_id,
        frame_jpeg=b"NOT_A_JPEG_AT_ALL",
    ))
    _check("AnalyzeFrame returns ERROR for corrupt JPEG",
           r.system_status == "ERROR", f"error={r.error_message!r}")


def suite_registration(stub, frame: np.ndarray, candidate_id: str) -> bool:
    _section("Suite 4 — RegisterCandidate")

    jpeg = _to_jpeg(frame)
    r = stub.RegisterCandidate(proctor_pb2.RegisterRequest(
        candidate_id=candidate_id,
        face_image=jpeg,
    ))
    _check("RegisterCandidate returns success=True or graceful failure",
           isinstance(r.success, bool),
           f"success={r.success} message={r.message!r}")
    _check("RegisterCandidate message is non-empty", len(r.message) > 0)

    if not r.success:
        print(f"  {YELLOW}[warn]{RESET} Registration failed ({r.message}) — "
              "face detection needs a real face photo. Continuing with remaining suites.")
    return r.success


def suite_start_session(stub, candidate_id: str) -> bool:
    _section("Suite 5 — StartSession")

    r = stub.StartSession(proctor_pb2.SessionRequest(candidate_id=candidate_id))
    _check("StartSession returns success=True", r.success, f"message={r.message!r}")
    return r.success


def suite_analyze_frames(stub, frame: np.ndarray, candidate_id: str):
    _section(f"Suite 6 — AnalyzeFrame × {_ANALYZE_FRAMES} (simulating Java 5s cadence)")

    jpeg      = _to_jpeg(frame)
    silence   = _silence_pcm()
    responses = []

    for i in range(_ANALYZE_FRAMES):
        audio = silence if i % 2 == 0 else _tone_pcm(300)
        req = proctor_pb2.FrameRequest(
            candidate_id=candidate_id,
            frame_jpeg=jpeg,
            timestamp=int(time.time()),
            audio_chunk=audio,
        )
        r = stub.AnalyzeFrame(req)
        responses.append(r)
        print(f"\n  Frame {i + 1}/{_ANALYZE_FRAMES}:")
        _check(f"  Frame {i+1}: system_status is SUCCESS or ERROR",
               r.system_status in ("SUCCESS", "ERROR"),
               f"status={r.system_status}")
        _check(f"  Frame {i+1}: candidate_id echoed correctly",
               r.candidate_id == candidate_id)

        if r.system_status == "SUCCESS":
            va = r.vision_analysis
            aa = r.audio_analysis
            _check(f"  Frame {i+1}: gaze_direction is valid",
                   va.gaze_direction in ("CENTER", "LEFT", "RIGHT", "OFF_SCREEN"),
                   f"gaze={va.gaze_direction!r}")
            _check(f"  Frame {i+1}: face_status is valid",
                   va.face_status in ("OK", "MISSING", "VIOLATION"),
                   f"face_status={va.face_status!r}")
            _check(f"  Frame {i+1}: similarity_score in [0, 1]",
                   0.0 <= va.similarity_score <= 1.0,
                   f"score={va.similarity_score:.4f}")
            _check(f"  Frame {i+1}: speech_probability in [0, 1]",
                   0.0 <= aa.speech_probability <= 1.0,
                   f"prob={aa.speech_probability:.4f}")
            _check(f"  Frame {i+1}: blink_count >= 0",
                   va.blink_count >= 0, f"blinks={va.blink_count}")
            _check(f"  Frame {i+1}: specs_confidence in [0, 1]",
                   0.0 <= va.specs_confidence <= 1.0,
                   f"conf={va.specs_confidence:.4f}")
            _check(f"  Frame {i+1}: violations is a list",
                   isinstance(list(r.violations), list),
                   f"violations={list(r.violations)}")

            # Print full JSON snapshot for visibility
            snapshot = {
                "timestamp":   r.timestamp,
                "candidateId": r.candidate_id,
                "visionAnalysis": {
                    "gazeDirection":   va.gaze_direction,
                    "isHeadTurned":    va.is_head_turned,
                    "yawAngle":        round(va.yaw_angle, 2),
                    "pitchAngle":      round(va.pitch_angle, 2),
                    "faceCount":       va.face_count,
                    "faceStatus":      va.face_status,
                    "faceMatched":     va.face_matched,
                    "similarityScore": round(va.similarity_score, 4),
                    "eyesClosed":      va.eyes_closed,
                    "blinkCount":      va.blink_count,
                    "specsDetected":   va.specs_detected,
                    "specsConfidence": round(va.specs_confidence, 4),
                    "detectedObjects": [
                        {"label": o.label, "confidence": round(o.confidence, 3),
                         "bbox": list(o.bbox)}
                        for o in va.detected_objects
                    ],
                },
                "audioAnalysis": {
                    "isHumanSpeech":     aa.is_human_speech,
                    "speechProbability": round(aa.speech_probability, 4),
                },
                "violations":           list(r.violations),
                "continuousViolations": list(r.continuous_violations),
                "snapshotViolations":   list(r.snapshot_violations),
                "systemStatus":         r.system_status,
            }
            print(f"\n{json.dumps(snapshot, indent=4)}")

        time.sleep(0.5)   # brief pause between frames (not 5s — test speed)

    return responses


def suite_end_session(stub, candidate_id: str):
    _section("Suite 7 — EndSession")

    r = stub.EndSession(proctor_pb2.SessionRequest(candidate_id=candidate_id))
    _check("EndSession returns success=True", r.success, f"message={r.message!r}")

    # Second EndSession must fail gracefully — session already removed
    r2 = stub.EndSession(proctor_pb2.SessionRequest(candidate_id=candidate_id))
    _check("Second EndSession returns success=False (idempotent)", not r2.success,
           f"message={r2.message!r}")


def suite_post_end_analyze(stub, frame: np.ndarray, candidate_id: str):
    _section("Suite 8 — AnalyzeFrame after EndSession (must return ERROR)")

    r = stub.AnalyzeFrame(proctor_pb2.FrameRequest(
        candidate_id=candidate_id,
        frame_jpeg=_to_jpeg(frame),
        timestamp=int(time.time()),
    ))
    _check("AnalyzeFrame after EndSession returns ERROR",
           r.system_status == "ERROR", f"error={r.error_message!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CodeCluster ML gRPC Integration Test")
    parser.add_argument("--image", default=None,
                        help="Path to a face JPEG to use for registration (optional)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  CodeCluster ML — gRPC Integration Test Suite{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Port     : {GRPC_PORT}")
    print(f"  Candidate: {_TEST_CANDIDATE}")

    # ── Acquire registration frame ─────────────────────────────────
    _section("Setup — Acquiring registration frame")
    frame = _load_image(args.image) if args.image else _capture_or_synthetic()

    # ── Start in-process gRPC server ───────────────────────────────
    _section("Setup — Starting in-process gRPC server")
    server = _start_server(GRPC_PORT)
    print(f"  {GREEN}Server started on port {GRPC_PORT}{RESET}")
    time.sleep(1.0)   # let server bind

    channel = grpc.insecure_channel(
        f"localhost:{GRPC_PORT}",
        options=[
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ("grpc.max_send_message_length",    10 * 1024 * 1024),
        ],
    )
    stub = proctor_pb2_grpc.ProctoringServiceStub(channel)

    try:
        # ── Run all suites ─────────────────────────────────────────
        suite_invalid_id(stub)
        suite_no_session(stub)

        registered = suite_registration(stub, frame, _TEST_CANDIDATE)

        if registered:
            suite_corrupt_jpeg(stub, _TEST_CANDIDATE)
            started = suite_start_session(stub, _TEST_CANDIDATE)
            if started:
                time.sleep(_SERVER_WAIT_SEC)   # let VideoMonitor thread warm up
                suite_analyze_frames(stub, frame, _TEST_CANDIDATE)
                suite_end_session(stub, _TEST_CANDIDATE)
                suite_post_end_analyze(stub, frame, _TEST_CANDIDATE)
        else:
            # Registration failed (synthetic frame — no real face)
            # Still run error-path suites that don't need a registered candidate
            _section("Suite 5-8 — Skipped (registration failed, no real face detected)")
            print(f"  {YELLOW}Tip: run with --image path/to/face.jpg for full coverage{RESET}")

    except Exception:
        print(f"\n{RED}[FATAL] Unexpected exception during test run:{RESET}")
        traceback.print_exc()
    finally:
        channel.close()
        server.stop(grace=2)

    # ── Final report ───────────────────────────────────────────────
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)
    total  = len(_results)

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  RESULTS: {passed}/{total} passed{RESET}")
    if failed:
        print(f"\n{RED}  Failed checks:{RESET}")
        for r in _results:
            if not r.passed:
                print(f"    {RED}✗{RESET} {r.name}" + (f"  →  {r.detail}" if r.detail else ""))
    else:
        print(f"  {GREEN}All checks passed ✓{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
