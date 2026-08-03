import threading
import numpy as np
from insightface.app import FaceAnalysis

from utils.image_utils import to_rgb
from config import FACE_MATCH_THRESHOLD


class FaceAuthenticator:
    """
    Verifies candidate identity using InsightFace buffalo_l embeddings.
    Call register() once at exam start, then predict() on each snapshot.
    """

    def __init__(self):
        self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        self._registered_embedding: np.ndarray | None = None
        self._lock = threading.Lock()

    def _get_embedding(self, frame) -> np.ndarray | None:
        faces = self._app.get(to_rgb(frame))
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def register(self, frame) -> bool:
        embedding = self._get_embedding(frame)
        if embedding is None:
            return False
        with self._lock:
            self._registered_embedding = embedding
        return True

    def predict(self, frame) -> dict:
        with self._lock:
            registered = self._registered_embedding

        if registered is None:
            raise RuntimeError("No registered embedding. Call register() first.")

        live_embedding = self._get_embedding(frame)
        if live_embedding is None:
            return {"faceMatched": False, "similarityScore": None}

        score = self._cosine_similarity(registered, live_embedding)
        return {
            "faceMatched":     score >= FACE_MATCH_THRESHOLD,
            "similarityScore": round(score, 4),
        }
