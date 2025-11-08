"""
SmartAssist v2 - Water Bottle Interaction Demo
Features:
 - Detect water bottle with ObjectDetector
 - Hand tracking with HandTracker
 - Shaking, opening, and closing detection
 - Scene recognition for context-aware guidance
 - Voice instructions with pyttsx3
 - On-screen overlay instructions

Requires:
 - src/vision/hand_tracker.py
 - src/vision/object_detector.py
 - src/utils/voice_feedback.py
 - pyttsx3, opencv-python, numpy, torch, torchvision
"""

import time
from collections import deque
import math

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# --- Import your modules ---
from src.vision.hand_tracker import HandTracker
from src.vision.object_detector import ObjectDetector
from src.utils.voice_feedback import speak

# ----- Parameters -----
FPS_SMOOTH = 10
HISTORY_SECONDS = 2.5
MAX_HISTORY = 30
PICKUP_Y_THRESHOLD = -30
PICKUP_MIN_FRAMES = 3
SHAKE_SPEED_THRESHOLD = 8.0
SHAKE_STD_THRESHOLD = 4.5
OPEN_ROTATION_THRESHOLD = 10
PROXIMITY_PADDING = 50
LABEL_TO_TRACK = "bottle"

# ----- Helper functions -----
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_bbox(pt, bbox, pad=0):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

def angle_between(v1, v2):
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosang = max(min(cosang, 1.0), -1.0)
    return math.degrees(math.acos(cosang))

# ----- Scene Recognition -----
class SceneRecognizer:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Placeholder labels
        self.scene_labels = ["kitchen", "office", "living room", "street", "parking lot", "unknown"]

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            _, pred = torch.max(out, 1)
        idx = int(pred.item()) % len(self.scene_labels)
        return self.scene_labels[idx]

# ----- Main Demo -----
class BottleInteractionDemoV2:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = ObjectDetector()
        self.hand = HandTracker()
        self.scene_recognizer = SceneRecognizer()
        self.history = deque(maxlen=MAX_HISTORY)
        self.state = "idle"
        self.last_state_change = time.time()
        self.bottle_bbox = None
        self.shake_cooldown_until = 0
        self.opened = False
        self.closed = False
        self.last_scene = None
        self.scene_update_interval = 3.0
        self.last_scene_update = 0

    def _extract_primary_bottle(self, detections):
        bottles = [d for d in detections if d["label"].lower() == LABEL_TO_TRACK]
        if not bottles:
            return None
        best = max(bottles, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
        return best["bbox"]

    def _get_hand_point(self, hand_entry, frame_shape):
        h, w = frame_shape[:2]
        if isinstance(hand_entry, tuple) and len(hand_entry)==2:
            return hand_entry
        if isinstance(hand_entry, list) and len(hand_entry)>0:
            first = hand_entry[0]
            if isinstance(first, tuple):
                return first
        if isinstance(hand_entry, dict) and "landmarks" in hand_entry:
            lm = hand_entry["landmarks"]
            if len(lm) > 8:
                l = lm[8]
                return int(l[0]*w), int(l[1]*h)
        return None

    def _append_history(self, bottle_center, hand_point):
        self.history.append({"t":time.time(), "bottle_center":bottle_center, "hand_point":hand_point})

    def _compute_motion_stats(self):
        pts = [h["bottle_center"] for h in self.history if h["bottle_center"] is not None]
        if len(pts)<2: return {"speeds":[], "mean_speed":0.0, "std_x":0.0}
        vecs = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(pts[:-1],pts[1:])]
        speeds = [math.hypot(v[0],v[1]) for v in vecs]
        xs = [v[0] for v in vecs]
        return {"speeds":speeds,"mean_speed":float(np.mean(speeds)),"std_x":float(np.std(xs)),"vecs":vecs}

    def update_state_machine(self, bottle_bbox, hand_point):
        now = time.time()
        bottle_center = bbox_center(bottle_bbox) if bottle_bbox else None
        self._append_history(bottle_center, hand_point)
        hand_near = bottle_bbox is not None and hand_point is not None and point_in_bbox(hand_point, bottle_bbox, pad=PROXIMITY_PADDING)
        motion = self._compute_motion_stats()

        
        # --- State Machine ---
        if self.state=="idle" and bottle_bbox:
            self.state="bottle_detected"
            speak("Bottle detected. Bring your hand near it.")
        elif self.state=="bottle_detected" and hand_near:
            self.state="hand_near_bottle"
            speak("Hand detected near the bottle. You can pick it up now.")
        elif self.state == "hand_near_bottle" and hand_near:
            # only keep non-None centers
            centers = [h["bottle_center"] for h in self.history if h["bottle_center"] is not None]
            if len(centers) >= PICKUP_MIN_FRAMES:
                # compare early to latest for upward motion
                if centers[-1][1] - centers[-PICKUP_MIN_FRAMES][1] <= PICKUP_Y_THRESHOLD:
                    self.state = "bottle_picked_up"
                    self.last_state_change = now
                    speak("Bottle picked up. Now shake it.")
        elif self.state=="bottle_picked_up":
            if motion["mean_speed"]>SHAKE_SPEED_THRESHOLD and motion["std_x"]>SHAKE_STD_THRESHOLD and now>self.shake_cooldown_until:
                self.state="shaking"
                self.shake_cooldown_until=now+1.5
                speak("Shaking detected. Try opening the cap.")
        elif self.state=="shaking" and bottle_bbox and hand_point:
            x1,y1,x2,y2=bottle_bbox
            cap_region=(x1,y1,x2,int(y1+0.35*(y2-y1)))
            if point_in_bbox(hand_point,cap_region,pad=20):
                hand_pts=[h["hand_point"] for h in self.history if h["hand_point"]]
                if len(hand_pts)>=3:
                    v_prev=(hand_pts[-2][0]-bottle_center[0], hand_pts[-2][1]-bottle_center[1])
                    v_now=(hand_pts[-1][0]-bottle_center[0], hand_pts[-1][1]-bottle_center[1])
                    ang=angle_between(v_prev,v_now)
                    if ang>OPEN_ROTATION_THRESHOLD:
                        self.state="opening"
                        self.opened=True
                        self.last_state_change=now
                        speak("Opening detected. Wait a moment before closing.")
        elif self.state=="opening" and now - self.last_state_change>2.5 and bottle_bbox and hand_point:
            hand_pts=[h["hand_point"] for h in self.history if h["hand_point"]]
            if len(hand_pts)>=3:
                v_prev=(hand_pts[-2][0]-bottle_center[0],hand_pts[-2][1]-bottle_center[1])
                v_now=(hand_pts[-1][0]-bottle_center[0],hand_pts[-1][1]-bottle_center[1])
                ang=angle_between(v_prev,v_now)
                if ang>OPEN_ROTATION_THRESHOLD:
                    self.state="closing"
                    self.closed=True
                    self.last_state_change=now
                    speak("Closing detected. Great! Demo complete.")

        # Final done state after short pause
        if self.state=="closing":
            if now - self.last_state_change > 1.0:
                self.state="done"

    def draw_overlay(self, frame, bottle_bbox, hand_point):
        h, w = frame.shape[:2]

        # Draw bottle bbox
        if bottle_bbox:
            x1, y1, x2, y2 = bottle_bbox
            color = (0, 255, 0)  # green
            if self.state in ("hand_near_bottle", "bottle_picked_up", "shaking", "opening", "closing"):
                color = (0, 165, 255)  # orange
            if self.state=="done":
                color=(0, 200, 0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cx,cy=bbox_center(bottle_bbox)
            cv2.circle(frame,(cx,cy),4,color,-1)
            cv2.putText(frame,f"Bottle ({self.state})",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # Draw hand
        if hand_point:
            cv2.circle(frame, hand_point, 6, (255,0,0), -1)
            cv2.putText(frame,"Hand",(hand_point[0]+8, hand_point[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

        # Draw bottle path
        pts=[h["bottle_center"] for h in self.history if h["bottle_center"]]
        for i in range(1,len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (200,200,50),2)

        # Display state and scene
        cv2.putText(frame, f"State: {self.state}", (10, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        if self.last_scene:
            cv2.putText(frame, f"Scene: {self.last_scene}", (10,h-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    def run(self):
        speak("Bottle demo starting. Show a water bottle to the camera.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read camera frame")
                break

            # Detect objects and hands
            detections=self.detector.detect(frame)
            hand_entries=self.hand.track(frame)

            # Select primary bottle and first hand point
            bottle_bbox=self._extract_primary_bottle(detections)
            hand_point=None
            if hand_entries:
                candidate=hand_entries[0]
                p=self._get_hand_point(candidate,frame.shape)
                if p:
                    hand_point=p

            # Update state machine
            self.update_state_machine(bottle_bbox, hand_point)

            # Draw overlays
            self.draw_overlay(frame,bottle_bbox,hand_point)
            cv2.imshow("Water Bottle Interaction Demo", frame)

            # Key handling
            if cv2.waitKey(1) & 0xFF==27:  # ESC
                break

            # Finish if done
            if self.state=="done":
                speak("Demo finished. Good job!")
                time.sleep(1.0)
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ----- Run demo -----
if __name__=="__main__":
    demo=BottleInteractionDemoV2(camera_index=0)
    demo.run()