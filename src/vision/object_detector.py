from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="src/models/yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, stream=True, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                if box.conf[0] < self.conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                detections.append({
                    "label": label,
                    "bbox": (x1, y1, x2, y2)
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detections