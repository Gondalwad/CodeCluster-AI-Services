import threading
import queue
import time
import numpy as np

from models.speech_detector import SpeechDetector

_SAMPLE_RATE  = 16000
_CHUNK_FRAMES = 8192


class AudioMonitor:
    """
    Processes audio chunks pushed from the gRPC layer on a background thread.
    Runs Silero VAD continuously and pushes SPEECH_DETECTED events to the
    shared violation queue when speech is detected.
    """

    def __init__(self, violation_queue: queue.Queue):
        self._queue       = violation_queue
        self._detector    = SpeechDetector()
        self._stop_evt    = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True, name="AudioMonitor")
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._lock          = threading.Lock()
        self._latest_result = {"isHumanSpeech": False, "speechProbability": 0.0}

    def start(self):
        self._stop_evt.clear()
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        self._detector.reset()

    def push_audio(self, pcm_bytes: bytes):
        """Called by the gRPC servicer for each incoming audio chunk. Non-blocking."""
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest_result)

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                pcm_bytes = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self._detector.predict(pcm_bytes)
            except Exception:
                result = {"isHumanSpeech": False, "speechProbability": 0.0}

            with self._lock:
                self._latest_result = result

            if result["isHumanSpeech"]:
                self._queue.put({
                    "type":      "SPEECH_DETECTED",
                    "timestamp": int(time.time()),
                    "data":      result,
                })
