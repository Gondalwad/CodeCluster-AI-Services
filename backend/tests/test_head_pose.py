import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.head_pose import HeadPoseEstimator


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Head Pose test started. Press 'q' to quit.")

    with HeadPoseEstimator() as estimator:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = estimator.predict(frame)

            lines = [
                f"Yaw:   {result['yawAngle']}  Turned: {result['isHeadTurned']}",
                f"Pitch: {result['pitchAngle']}  Nodding: {result['isNodding']}",
                f"Roll:  {result['rollAngle']}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Head Pose Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
