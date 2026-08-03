import cv2
import grpc
import time

from protos.proctor_pb2 import (
    RegisterRequest,
    SessionRequest,
    FrameRequest,
)
from protos.proctor_pb2_grpc import ProctoringServiceStub


CHANNEL = grpc.insecure_channel("localhost:50051")
stub = ProctoringServiceStub(CHANNEL)

candidate_id = "test123"

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if not ret:
    print("Camera not found")
    exit()

_, jpeg = cv2.imencode(".jpg", frame)
jpeg_bytes = jpeg.tobytes()

print("Registering candidate...")

response = stub.RegisterCandidate(
    RegisterRequest(
        candidate_id=candidate_id,
        face_image=jpeg_bytes,
    )
)

print(response)

print("Starting session...")

response = stub.StartSession(
    SessionRequest(
        candidate_id=candidate_id,
    )
)

print(response)

for i in range(10):

    ret, frame = cap.read()
    cv2.imwrite("registration.jpg", frame)
print("Saved registration.jpg")

_, jpeg = cv2.imencode(".jpg", frame)

prediction = stub.AnalyzeFrame(
    FrameRequest(
        candidate_id=candidate_id,
        frame_jpeg=jpeg.tobytes(),
        timestamp=int(time.time() * 1000),
        audio_chunk=b"",
    )
)

print("=" * 40)
print(prediction)

time.sleep(1)

print("Ending session...")

response = stub.EndSession(
    SessionRequest(
        candidate_id=candidate_id,
    )
)

print(response)

cap.release()
