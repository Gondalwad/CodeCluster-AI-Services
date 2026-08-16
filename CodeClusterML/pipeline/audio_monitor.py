import threading
import queue
import time

from models.speech_detector import SpeechDetector

_SAMPLE_RATE  = 16000
_CHUNK_FRAMES = 8192


class AudioMonitor:
    def __init__(self):
        self._detector    = SpeechDetector()
        self._stop_evt    = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True, name="AudioMonitor")
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._lock          = threading.Lock()
        self._latest_result = {"isHumanSpeech": False, "speechProbability": 0.0}

    def start(self):
        self._stop_evt.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="AudioMonitor")
            self._thread.start()

    def reset(self):
        with self._lock:
            self._latest_result = {"isHumanSpeech": False, "speechProbability": 0.0}
        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass
        self._detector.reset()

    def stop(self):
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._detector.reset()

    def push_audio(self, pcm_bytes: bytes):
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
