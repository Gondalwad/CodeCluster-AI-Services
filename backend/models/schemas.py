from pydantic import BaseModel
from typing import Literal, Optional


class FrameMessage(BaseModel):
    type: Literal["frame"]
    payload: str
    timestamp: int


class PredictionResult(BaseModel):
    face_detected: bool
    authenticated: bool
    gaze: str
    blink: bool
    head_pose: str
    phone_detected: bool
    person_count: int
    speech_detected: bool


class ProctorEvent(BaseModel):
    type: Literal["warning", "terminate", "heartbeat"]
    message: str
    timestamp: Optional[int] = None
