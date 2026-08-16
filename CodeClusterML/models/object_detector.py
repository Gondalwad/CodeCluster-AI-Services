from ultralytics import YOLO

from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, BANNED_CLASSES


class ObjectDetector:
    def __init__(self):
        self._model  = YOLO(YOLO_MODEL_PATH)
        self._banned = set(BANNED_CLASSES)

    def predict(self, frame) -> dict:
        results = self._model.predict(
            source=frame,
            imgsz=320,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        detected = []
        for box in results.boxes:
            label = self._model.names[int(box.cls)]
            if label not in self._banned:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detected.append({
                "label":      label,
                "confidence": round(float(box.conf), 3),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            })

        return {"detectedObjects": detected}
