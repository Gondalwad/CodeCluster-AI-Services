import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.spectacles_detector import SpectaclesDetector


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Spectacles Detector test started. Press 'q' to quit.")
    print("[INFO] Wear glasses then remove them to validate detection.")

    with SpectaclesDetector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.predict(frame)

            color = (0, 0, 255) if result["specsDetected"] else (0, 255, 0)
            lines = [
                f"Specs Detected: {result['specsDetected']}",
                f"Confidence:     {result['confidence']}",
                f"Edge Score:     {result['edgeScore']}",
                f"Glare Score:    {result['glareScore']}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.imshow("Spectacles Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
