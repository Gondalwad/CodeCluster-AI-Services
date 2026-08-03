import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gaze_tracker import GazeTracker


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Gaze Tracker test started. Press 'q' to quit.")

    with GazeTracker() as tracker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.predict(frame)

            label = f"Gaze: {result['gazeDirection']}  L:{result['leftIrisRatio']}  R:{result['rightIrisRatio']}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gaze Tracker Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
