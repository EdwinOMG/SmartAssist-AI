import cv2
from src.vision.hand_tracker import HandTracker
from src.vision.object_detector import ObjectDetector

class FusionTracker:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.object_detector = ObjectDetector()
        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, [], None

        # Detect objects and hands
        objects = self.object_detector.detect(frame)
        hands = self.hand_tracker.track(frame)

        return frame, objects, hands

    def display_instruction(self, frame, text):
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AI Task Assistant", frame)

    def is_hand_near(self, target_object_name):
        # TODO: implement logic that checks proximity between hand and object
        # Placeholder for now
        return True