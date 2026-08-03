import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.face_auth import FaceAuthenticator


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Face Auth test started.")
    print("[INFO] Press 'r' to register your face. Press 'q' to quit.")

    auth      = FaceAuthenticator()
    registered = False
    result     = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            success = auth.register(frame)
            registered = success
            print(f"[INFO] Registration {'successful' if success else 'failed — no face detected'}.")

        if registered:
            result = auth.predict(frame)
            color  = (0, 255, 0) if result["faceMatched"] else (0, 0, 255)
            label  = f"Matched: {result['faceMatched']}  Score: {result['similarityScore']}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Press 'r' to register", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Face Auth Test", frame)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test complete.")


if __name__ == "__main__":
    main()
