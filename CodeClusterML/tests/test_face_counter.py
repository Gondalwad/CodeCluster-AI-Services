import cv2
import sys
import os
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.face_counter import FaceCounter

_drawing = mp.solutions.drawing_utils
_styles  = mp.solutions.drawing_styles


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Face Counter test started. Press 'q' to quit.")

    with FaceCounter() as counter:
        # keep a raw detector reference just for drawing bboxes in the test
        raw_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = counter.predict(frame)

            # draw detections for visual feedback
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = raw_detector.process(rgb).detections or []
            for det in detections:
                _drawing.draw_detection(frame, det)

            color = (0, 255, 0) if result["faceStatus"] == "OK" else (0, 0, 255)
            label = f"Faces: {result['faceCount']}  Status: {result['faceStatus']}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Counter Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        raw_detector.close()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
