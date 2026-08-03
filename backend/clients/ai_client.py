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
        self.channel = grpc.insecure_channel(host)
        self.stub = ProctoringServiceStub(self.channel)

    def register_candidate(self, candidate_id: str, face_image: bytes):
        request = RegisterRequest(
            candidate_id=candidate_id,
            face_image=face_image,
        )
        return self.stub.RegisterCandidate(request)

    def start_session(self, candidate_id: str):
        request = SessionRequest(
            candidate_id=candidate_id,
        )
        return self.stub.StartSession(request)

    def analyze_frame(
        self,
        candidate_id: str,
        frame: bytes,
        audio: bytes = b"",
    ):
        request = FrameRequest(
            candidate_id=candidate_id,
            frame_jpeg=frame,
            timestamp=int(time.time() * 1000),
            audio_chunk=audio,
        )
        return self.stub.AnalyzeFrame(request)

    def end_session(self, candidate_id: str):
        request = SessionRequest(
            candidate_id=candidate_id,
        )
        return self.stub.EndSession(request)


ai_client = AIClient()
