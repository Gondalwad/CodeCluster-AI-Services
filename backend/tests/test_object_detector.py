import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.object_detector import ObjectDetector


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Object Detector test started. Press 'q' to quit.")
    print(f"[INFO] Downloading yolov8n.pt on first run if not cached...")

    detector = ObjectDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.predict(frame)

        for obj in result["detectedObjects"]:
            x1, y1, x2, y2 = obj["bbox"]
            label = f"{obj['label']} {obj['confidence']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status = f"Banned items detected: {len(result['detectedObjects'])}"
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Object Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
