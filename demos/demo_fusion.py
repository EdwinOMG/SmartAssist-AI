import cv2
import mediapipe as mp
from ultralytics import YOLO

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

model = YOLO('yolov8n.pt')

def fingertip_xy(landmarks, w, h):
    tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return int(tip.x * w), int(tip.y * h)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_res = hands.process(rgb)
    obj_res = model(frame, conf=0.35)

    h, w = frame.shape[:2]
    fingertips = []

    if hand_res.multi_hand_landmarks:
        for hand in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            fingertips.append(fingertip_xy(hand, w, h))

    for r in obj_res:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            color = (0, 255, 0)

            for fx, fy in fingertips:
                if x1 < fx < x2 and y1 < fy < y2:
                    color = (0, 0, 255)
                    label += " (interacting)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Fusion Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()