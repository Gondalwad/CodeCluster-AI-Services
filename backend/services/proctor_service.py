import logging
import grpc

from clients.ai_client import ai_client

logger = logging.getLogger(__name__)


def _grpc_unavailable(e: Exception) -> bool:
    return isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.UNAVAILABLE


class ProctorService:
    def __init__(self):
        self.active_sessions = set()

    async def process_frame(
        self,
        candidate_id: str,
        frame: bytes,
        audio: bytes = b"",
    ):
        from protos.proctor_pb2 import FrameResponse

        def _waiting():
            r = FrameResponse()
            r.candidate_id = candidate_id
            r.system_status = "WAITING_FOR_REGISTRATION"
            return r

        # Register and start the ML session only once, retry until valid frame
        if candidate_id not in self.active_sessions:

            if len(frame) < 100:
                logger.debug("Skipping registration for %s — frame too small (%d bytes)", candidate_id, len(frame))
                return _waiting()

            logger.info("Registering candidate=%s", candidate_id)

            try:
                register_response = ai_client.register_candidate(
                    candidate_id=candidate_id,
                    face_image=frame,
                )
            except Exception as e:
                logger.error("gRPC register failed for %s: %s", candidate_id, e)
                return _waiting()

            logger.info("Register Response -> success=%s, message=%s",
                        register_response.success, register_response.message)

            if not register_response.success:
                logger.warning("Registration failed for %s: %s — will retry", candidate_id, register_response.message)
                return _waiting()

            try:
                session_response = ai_client.start_session(candidate_id)
            except Exception as e:
                logger.error("gRPC start_session failed for %s: %s", candidate_id, e)
                return _waiting()

            logger.info("Start Session -> success=%s, message=%s",
                        session_response.success, session_response.message)

            if not session_response.success:
                raise ValueError(session_response.message)

            self.active_sessions.add(candidate_id)
            logger.info("Candidate session initialized successfully for %s", candidate_id)

        # Analyze current frame
        try:
            prediction = ai_client.analyze_frame(
                candidate_id=candidate_id,
                frame=frame,
                audio=audio,
            )
        except Exception as e:
            logger.error("gRPC analyze_frame failed for %s: %s", candidate_id, e)
            # Drop session so next frame triggers fresh registration
            self.active_sessions.discard(candidate_id)
            r = FrameResponse()
            r.candidate_id = candidate_id
            r.system_status = "WAITING_FOR_REGISTRATION"
            return r

        return prediction

    def push_audio(self, candidate_id: str, audio: bytes):
        """Push a raw PCM audio chunk directly to the ML pipeline."""
        if candidate_id not in self.active_sessions:
            return
        try:
            ai_client.analyze_frame(
                candidate_id=candidate_id,
                frame=b"",
                audio=audio,
            )
        except Exception as e:
            logger.debug("push_audio gRPC error for %s: %s", candidate_id, e)

    def end_session(self, candidate_id: str):
        if candidate_id in self.active_sessions:
            logger.info("Ending session for %s", candidate_id)
            try:
                response = ai_client.end_session(candidate_id)
                logger.info("End Session -> success=%s, message=%s",
                            response.success, response.message)
            except Exception as e:
                logger.warning("end_session gRPC error for %s: %s", candidate_id, e)
            finally:
                self.active_sessions.discard(candidate_id)


proctor_service = ProctorService()
