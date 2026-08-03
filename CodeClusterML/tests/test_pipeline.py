import cv2
import sys
import json
import time
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.proctoring_pipeline import ProctoringPipeline


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] ProctoringPipeline test started.")
    print("[INFO] Press 'r' to register face (auto-starts monitoring), 'q' to quit.")
    print("[INFO] IMPORTANT: Click the webcam window first so key presses are detected.")

    pipeline    = ProctoringPipeline(cap, candidate_id="TEST_CANDIDATE_01")
    registered  = False
    started     = False
    last_snap   = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not registered:
            if pipeline.register_candidate(frame):
                registered = True
                pipeline.start()
                started    = True
                last_snap  = time.time()
                print("[INFO] Candidate registered. Monitoring started. First snapshot in 5s.")
            else:
                print("[WARN] No face found — try again.")

        if key == ord('q'):
            break

        # Snapshot every 5s
        if started and (time.time() - last_snap) >= 5.0:
            last_snap   = time.time()
            last_result = pipeline.process_frame(frame)
            print("\n" + "=" * 60)
            print(json.dumps(last_result, indent=2))
            print("=" * 60)

        # ── Overlay ────────────────────────────────────────────────
        display = frame.copy()

        if not registered:
            cv2.putText(display, "Click window, then press 'r' to register",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Registered + Monitoring ACTIVE",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Countdown to next snapshot
            remaining = max(0, 5.0 - (time.time() - last_snap))
            cv2.putText(display, f"Next snapshot in: {remaining:.1f}s",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if last_result:
            violations = last_result.get("violations", [])
            v_color = (0, 0, 255) if violations else (0, 255, 0)
            cv2.putText(display, f"Violations: {violations or 'None'}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, v_color, 1)

            va = last_result["visionAnalysis"]
            aa = last_result["audioAnalysis"]
            lines = [
                f"Gaze: {va['gazeDirection']}  HeadTurned: {va['isHeadTurned']}",
                f"Faces: {va['faceCount']} ({va['faceStatus']})  Match: {va['faceMatched']} ({va['similarityScore']})",
                f"Specs: {va['specsDetected']}  Blinks: {va['blinkCount']}  EyesClosed: {va['eyesClosed']}",
                f"Speech: {aa['isHumanSpeech']} ({aa['speechProbability']})",
                f"Objects: {[o['label'] for o in va['detectedObjects']]}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 130 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        cv2.imshow("ProctoringPipeline", display)

    if started:
        pipeline.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
