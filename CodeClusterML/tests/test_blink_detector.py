import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.blink_detector import BlinkDetector


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Blink Detector test started. Press 'q' to quit.")

    with BlinkDetector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.predict(frame)

            color = (0, 0, 255) if result["eyesClosed"] else (0, 255, 0)
            lines = [
                f"Eyes Closed: {result['eyesClosed']}",
                f"Blink Count: {result['blinkCount']}",
                f"EAR Score:   {result['earScore']}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (20, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.imshow("Blink Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
