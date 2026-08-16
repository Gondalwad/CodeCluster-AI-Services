import asyncio
import time
import grpc

from protos.proctor_pb2 import (
    FrameRequest,
    RegisterRequest,
    SessionRequest,
)
from protos.proctor_pb2_grpc import ProctoringServiceStub


class AIClient:
    def __init__(self, host="127.0.0.1:50051"):
        channel = grpc.insecure_channel(host)
        try:
            self.stub = ProctoringServiceStub(channel)
        except Exception:
            channel.close()
            raise
        self.channel = channel

    # ── Sync helpers (run inside thread pool) ──────────────

    def _register_candidate_sync(self, candidate_id: str, face_image: bytes):
        request = RegisterRequest(
            candidate_id=candidate_id,
            face_image=face_image,
        )
        return self.stub.RegisterCandidate(request)

    def _start_session_sync(self, candidate_id: str):
        request = SessionRequest(
            candidate_id=candidate_id,
        )
        return self.stub.StartSession(request)

    def _analyze_frame_sync(self, candidate_id: str, frame: bytes):
        request = FrameRequest(
            candidate_id=candidate_id,
            frame_jpeg=frame,
            timestamp=int(time.time() * 1000),
            audio_chunk=b"",
        )
        return self.stub.AnalyzeFrame(request)

    def _end_session_sync(self, candidate_id: str):
        request = SessionRequest(
            candidate_id=candidate_id,
        )
        return self.stub.EndSession(request)

    def _push_audio_sync(self, candidate_id: str, audio: bytes):
        request = FrameRequest(
            candidate_id=candidate_id,
            frame_jpeg=b"",
            timestamp=int(time.time() * 1000),
            audio_chunk=audio,
        )
        return self.stub.AnalyzeFrame(request)

    # ── Async wrappers (safe for the event loop) ──────────

    async def register_candidate(self, candidate_id: str, face_image: bytes):
        return await asyncio.to_thread(
            self._register_candidate_sync, candidate_id, face_image
        )

    async def start_session(self, candidate_id: str):
        return await asyncio.to_thread(
            self._start_session_sync, candidate_id
        )

    async def analyze_frame(self, candidate_id: str, frame: bytes):
        return await asyncio.to_thread(
            self._analyze_frame_sync, candidate_id, frame
        )

    async def end_session(self, candidate_id: str):
        return await asyncio.to_thread(
            self._end_session_sync, candidate_id
        )

    async def push_audio(self, candidate_id: str, audio: bytes):
        return await asyncio.to_thread(
            self._push_audio_sync, candidate_id, audio
        )


ai_client = AIClient()
