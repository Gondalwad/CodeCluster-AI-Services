"""
tests/test_fixes.py
Automated tests for all 5 industrial-level fixes.
No webcam, no microphone, no gRPC server needed.
Run: python tests/test_fixes.py
"""

import sys
import os
import time
import threading
import queue
import struct
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    _results.append((name, condition))


# ----------------------------------------------------------------
# FIX 1 — spectacles_detector model path is absolute
# ----------------------------------------------------------------
def test_spectacles_model_path():
    print("\n-- Fix 1: SpectaclesDetector absolute model path --")
    from pathlib import Path
    # Import the module and read _MODEL_PATH directly
    import models.spectacles_detector as sd_mod
    path = sd_mod._MODEL_PATH
    check("_MODEL_PATH is absolute", path.is_absolute(),
          str(path))
    check("_MODEL_PATH resolves to correct filename", path.name == "glasses_model.onnx",
          path.name)


# ----------------------------------------------------------------
# FIX 2 — FaceAuthenticator thread safety
# ----------------------------------------------------------------
def test_face_auth_thread_safety():
    print("\n-- Fix 2: FaceAuthenticator thread safety --")
    from models.face_auth import FaceAuthenticator
    import threading

    auth = FaceAuthenticator()
    errors = []

    # Synthetic 640x480 black frame — InsightFace will find no face, that's fine
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    def register_loop():
        for _ in range(5):
            try:
                auth.register(blank)
            except Exception as e:
                errors.append(f"register: {e}")

    def predict_loop():
        # Manually set embedding so predict() doesn't raise RuntimeError
        auth._registered_embedding = np.random.rand(512).astype(np.float32)
        for _ in range(5):
            try:
                auth.predict(blank)
            except RuntimeError:
                pass  # "No registered embedding" is fine before register sets it
            except Exception as e:
                errors.append(f"predict: {e}")

    t1 = threading.Thread(target=register_loop)
    t2 = threading.Thread(target=predict_loop)
    t1.start(); t2.start()
    t1.join(); t2.join()

    check("No race condition errors during concurrent register+predict",
          len(errors) == 0, str(errors) if errors else "clean")
    check("FaceAuthenticator has _lock attribute", hasattr(auth, "_lock"))


# ----------------------------------------------------------------
# FIX 3 — FrameBuffer: each frame returned only once (no stale reuse)
# ----------------------------------------------------------------
def test_frame_buffer_no_stale_reuse():
    print("\n-- Fix 3: FrameBuffer stale frame prevention --")
    from pipeline.video_monitor import FrameBuffer

    buf = FrameBuffer()
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_b = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Write frame_a, read it — should get frame_a
    buf.write(frame_a)
    got = buf.read(timeout=1.0)
    check("read() returns written frame", got is not None and np.array_equal(got, frame_a))

    # Read again without a new write — should timeout and return None
    got2 = buf.read(timeout=0.3)
    check("read() returns None when no new frame written (no stale reuse)", got2 is None,
          f"got: {type(got2)}")

    # Write frame_b, read — should get frame_b not frame_a
    buf.write(frame_b)
    got3 = buf.read(timeout=1.0)
    check("read() returns new frame after second write",
          got3 is not None and np.array_equal(got3, frame_b))


# ----------------------------------------------------------------
# FIX 4 — ProctoringPipeline process_frame concurrency lock
# ----------------------------------------------------------------
def test_process_frame_lock():
    print("\n-- Fix 4: ProctoringPipeline process_frame concurrency lock --")
    from pipeline.proctoring_pipeline import ProctoringPipeline

    pipeline = ProctoringPipeline(candidate_id="TEST_LOCK")
    check("_process_lock exists", hasattr(pipeline, "_process_lock"))
    check("_process_lock is a threading.Lock",
          isinstance(pipeline._process_lock, type(threading.Lock())))

    # Simulate two threads trying to call process_frame simultaneously
    # Both should complete without error (lock serialises them)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    # Manually set registered embedding so face_auth.predict() doesn't raise
    pipeline._face_auth._registered_embedding = np.random.rand(512).astype(np.float32)

    errors = []
    results = []

    def call_process():
        try:
            r = pipeline.process_frame(blank)
            results.append(r["systemStatus"])
        except Exception as e:
            errors.append(str(e))

    t1 = threading.Thread(target=call_process)
    t2 = threading.Thread(target=call_process)
    t1.start(); t2.start()
    t1.join(); t2.join()

    check("Both concurrent process_frame calls completed", len(results) == 2,
          f"results={len(results)}, errors={errors}")
    check("No errors during concurrent process_frame", len(errors) == 0,
          str(errors) if errors else "clean")


# ----------------------------------------------------------------
# FIX 5 — AudioMonitor push_audio (no local mic, no sounddevice)
# ----------------------------------------------------------------
def test_audio_monitor_push_audio():
    print("\n-- Fix 5: AudioMonitor push_audio (no local mic) --")
    from pipeline.audio_monitor import AudioMonitor

    vq = queue.Queue()
    monitor = AudioMonitor(vq)

    check("AudioMonitor has push_audio method", hasattr(monitor, "push_audio"))
    check("AudioMonitor has _audio_queue", hasattr(monitor, "_audio_queue"))

    monitor.start()

    # Generate 512ms of silence (16kHz, mono, int16) — VAD should say no speech
    silence = np.zeros(8192, dtype=np.int16).tobytes()
    monitor.push_audio(silence)
    time.sleep(0.5)  # let thread process it

    latest = monitor.get_latest()
    check("get_latest() returns dict with isHumanSpeech key",
          "isHumanSpeech" in latest)
    check("Silence correctly classified as no speech",
          latest["isHumanSpeech"] is False,
          f"isHumanSpeech={latest['isHumanSpeech']}, prob={latest['speechProbability']}")

    # Generate synthetic speech-like audio (sine wave at 300Hz — VAD may or may not flag,
    # but the important thing is push_audio doesn't crash and processes it)
    t = np.linspace(0, 0.512, 8192, dtype=np.float32)
    sine = (np.sin(2 * np.pi * 300 * t) * 16000).astype(np.int16)
    monitor.push_audio(sine.tobytes())
    time.sleep(0.5)

    latest2 = monitor.get_latest()
    check("push_audio processed sine wave without crash",
          "speechProbability" in latest2,
          f"prob={latest2['speechProbability']}")

    monitor.stop()
    check("AudioMonitor stopped cleanly", not monitor._thread.is_alive())


# ----------------------------------------------------------------
# FIX 5b — servicer.py wires audio_chunk to push_audio
# ----------------------------------------------------------------
def test_servicer_audio_wiring():
    print("\n-- Fix 5b: servicer.py audio_chunk wiring --")
    import inspect
    # proctor_pb2 lives inside grpc_service/ — add it to path before importing servicer
    grpc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grpc_service"))
    if grpc_dir not in sys.path:
        sys.path.insert(0, grpc_dir)
    import grpc_service.servicer as svc_mod
    src = inspect.getsource(svc_mod.ProctoringServicer.AnalyzeFrame)
    check("AnalyzeFrame calls push_audio", "push_audio" in src)
    check("AnalyzeFrame checks request.audio_chunk", "audio_chunk" in src)


# ----------------------------------------------------------------
# BONUS — candidate_id validation in servicer
# ----------------------------------------------------------------
def test_candidate_id_validation():
    print("\n-- Bonus: candidate_id validation --")
    grpc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grpc_service"))
    if grpc_dir not in sys.path:
        sys.path.insert(0, grpc_dir)
    import grpc_service.servicer as svc_mod

    check("_validate_id accepts valid ID",
          svc_mod._validate_id("CANDIDATE_001") is True)
    check("_validate_id rejects newline injection",
          svc_mod._validate_id("abc\nFAKE_LOG") is False)
    check("_validate_id rejects >64 chars",
          svc_mod._validate_id("a" * 65) is False)
    check("_validate_id rejects empty string",
          svc_mod._validate_id("") is False)
    check("_safe_id strips control chars",
          "\n" not in svc_mod._safe_id("abc\ninjected"))


# ----------------------------------------------------------------
# Run all
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CodeCluster ML — Industrial Fixes Test Suite")
    print("=" * 60)

    test_spectacles_model_path()
    test_face_auth_thread_safety()
    test_frame_buffer_no_stale_reuse()
    test_process_frame_lock()
    test_audio_monitor_push_audio()
    test_servicer_audio_wiring()
    test_candidate_id_validation()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in _results if ok)
    failed = sum(1 for _, ok in _results if not ok)
    print(f"  Results: {passed} passed, {failed} failed out of {len(_results)} checks")
    if failed:
        print("  FAILED checks:")
        for name, ok in _results:
            if not ok:
                print(f"    FAIL: {name}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
