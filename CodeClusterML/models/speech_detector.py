import torch
import numpy as np

from config import SPEECH_CONFIDENCE_THRESHOLD

_SAMPLE_RATE = 16000
_CHUNK_SIZE  = 512
_TARGET_RMS  = 0.1
_MIN_ENERGY  = 1e-6
_WHISPER_RMS = 0.09


class SpeechDetector:
    """Detects human speech in raw PCM audio using Silero VAD."""

    def __init__(self):
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()

    def _bytes_to_chunks(self, audio_bytes: bytes) -> tuple:
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        pcm /= 32768.0

        raw_rms = float(np.sqrt(np.mean(pcm ** 2)))
        if raw_rms < _MIN_ENERGY:
            return [], 0.0

        pcm_norm = np.clip(pcm * (_TARGET_RMS / raw_rms), -1.0, 1.0)
        total_chunks = len(pcm_norm) // _CHUNK_SIZE
        chunks = [
            torch.from_numpy(pcm_norm[i * _CHUNK_SIZE:(i + 1) * _CHUNK_SIZE])
            for i in range(total_chunks)
        ]
        return chunks, raw_rms

    def predict(self, audio_bytes: bytes) -> dict:
        chunks, raw_rms = self._bytes_to_chunks(audio_bytes)
        if not chunks:
            return {"isHumanSpeech": False, "speechProbability": 0.0}

        with torch.no_grad():
            probs = [self._model(chunk, _SAMPLE_RATE).item() for chunk in chunks]

        self._model.reset_states()

        vad_prob   = float(np.max(probs))
        is_whisper = raw_rms > _WHISPER_RMS and vad_prob < SPEECH_CONFIDENCE_THRESHOLD
        final_prob = max(vad_prob, 0.60 if is_whisper else 0.0)

        return {
            "isHumanSpeech":     final_prob >= SPEECH_CONFIDENCE_THRESHOLD,
            "speechProbability": round(final_prob, 4),
        }

    def reset(self):
        self._model.reset_states()
