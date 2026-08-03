import time

import cv2
import grpc

from protos.proctor_pb2 import (
    RegisterRequest,
    SessionRequest,
    FrameRequest,
)
from protos.proctor_pb2_grpc import ProctoringServiceStub

from services.rule_engine import rule_engine
from services.warning_manager import warning_manager


# ==========================================================
# Configuration
# ==========================================================

CHANNEL = grpc.insecure_channel("localhost:50051")
stub = ProctoringServiceStub(CHANNEL)

candidate_id = "test123"


# ==========================================================
# Initialize Camera
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to open webcam.")
    exit()

print("📷 Webcam started.")


# ==========================================================
# Registration
# ==========================================================

print("\nLook at the camera...")
time.sleep(2)

ret, frame = cap.read()

if not ret:
    print("❌ Failed to capture registration image.")
    cap.release()
    exit()

_, jpeg = cv2.imencode(".jpg", frame)

print("\nRegistering candidate...")

response = stub.RegisterCandidate(
    RegisterRequest(
        candidate_id=candidate_id,
        face_image=jpeg.tobytes(),
    )
)

print(response)

if not response.success:
    cap.release()
    exit()


# ==========================================================
# Start Session
# ==========================================================

print("\nStarting session...")

response = stub.StartSession(
    SessionRequest(
        candidate_id=candidate_id,
    )
)

print(response)

if not response.success:
    cap.release()
    exit()

print("\n===================================================")
print("        LIVE PROCTORING TEST STARTED")
print("          Press Q to Quit")
print("===================================================\n")


# ==========================================================
# Live Monitoring
# ==========================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        _, jpeg = cv2.imencode(".jpg", frame)

        prediction = stub.AnalyzeFrame(
            FrameRequest(
                candidate_id=candidate_id,
                frame_jpeg=jpeg.tobytes(),
                timestamp=int(time.time() * 1000),
                audio_chunk=b"",
            )
        )

        # ==================================================
        # Rule Engine
        # ==================================================

        rule_result = rule_engine.evaluate(prediction)

        # ==================================================
        # Warning Manager
        # ==================================================

        warning_result = warning_manager.update(
            candidate_id,
            rule_result["violations"],
        )

        response = {
            **rule_result,
            **warning_result,
        }

        # ==================================================
        # Console Output (clean exam-style summary)
        # ==================================================

        if warning_result["warning"] or warning_result["terminate"]:
            print("\n" + "=" * 60)
            print("⚠ EXAM ALERT")
            print("-" * 60)
            print(
                f"Status     : {rule_result['system_status']}"
            )
            print(
                f"Warnings   : {warning_result['warning_count']} / "
                f"{warning_manager.max_warnings}"
            )
            if warning_result["new_warnings"]:
                print(
                    "Issue      : "
                    + ", ".join(warning_result["new_warnings"])
                )
            print(f"Terminate  : {warning_result['terminate']}")
            print("=" * 60)
        elif int(time.time() * 1000) % 5000 < 100:
            print(
                f"Monitor: status={rule_result['system_status']} | "
                f"warnings={warning_result['warning_count']} / "
                f"{warning_manager.max_warnings}"
            )

        # ==================================================
        # Show Webcam
        # ==================================================

        cv2.imshow("Live Proctor Test", frame)

        # ==================================================
        # End Exam
        # ==================================================

        if response["terminate"]:
            print("\n🚨 EXAM TERMINATED 🚨")
            break

        # Quit manually
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

# ==========================================================
# Cleanup
# ==========================================================

print("\nEnding session...")

response = stub.EndSession(
    SessionRequest(
        candidate_id=candidate_id,
    )
)

print(response)

warning_manager.reset(candidate_id)

cap.release()
cv2.destroyAllWindows()

print("\n✅ Live proctor test finished.")
