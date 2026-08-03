import sys
import os
import sounddevice as sd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.speech_detector import SpeechDetector

_SAMPLE_RATE  = 16000
_CHUNK_FRAMES = 3072  # 6 x 512 samples
_DEVICE       = 1     # Realtek Microphone Array (system default)
_CHANNELS     = 2     # Realtek array mic is stereo — mix to mono before passing to VAD
_RESET_EVERY  = 10    # reset VAD state every N chunks to prevent state drift


def main():
    print("[INFO] Speech Detector test started. Speak into your mic. Press Ctrl+C to quit.")
    print("[INFO] Loading Silero VAD from PyTorch Hub (downloads on first run)...")

    detector = SpeechDetector()
    detector.reset()

    print("[INFO] Model ready. Listening...\n")

    chunk_count = 0
    try:
        while True:
            audio_chunk = sd.rec(
                frames=_CHUNK_FRAMES,
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="int16",
                device=_DEVICE,
                blocking=True,
            )
            # mix stereo → mono by averaging channels
            mono        = audio_chunk.mean(axis=1).astype(np.int16)
            audio_bytes = mono.tobytes()
            result      = detector.predict(audio_bytes)

            chunk_count += 1
            if chunk_count % _RESET_EVERY == 0:
                detector.reset()  # prevent VAD state drift

            indicator = "SPEECH" if result["isHumanSpeech"] else "SILENT"
            print(f"\r{indicator}  prob: {result['speechProbability']:.4f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] Test complete.")


if __name__ == "__main__":
    main()
