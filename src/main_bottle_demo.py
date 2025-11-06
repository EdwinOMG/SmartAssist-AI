# src/main_bottle_demo.py
"""
Water bottle interaction demo:
 - Detect a 'bottle' with your object detector
 - Watch for hand proximity -> pick up
 - Detect shaking (oscillatory motion)
 - Detect open/close (heuristic: rotational motion near cap)
Requires:
 - src/vision/hand_tracker.py   (HandTracker class with track(frame) -> list)
 - src/vision/object_detector.py (ObjectDetector class with detect(frame) -> list of {"label","bbox"})
 - src/utils/voice_feedback.py  (speak(text) function)
"""

import time
from collections import deque
import math

import cv2
import numpy as np

# Import your modules (adjust paths if you used different names)
from src.vision.hand_tracker import HandTracker
from src.vision.object_detector import ObjectDetector
from src.utils.voice_feedback import speak

# ----- Parameters -----
FPS_SMOOTH = 10
HISTORY_SECONDS = 2.5
MAX_HISTORY = 30  # number of frames to keep (approx)
PICKUP_Y_THRESHOLD = -30       # upward motion (pixels) to count as "picked up"
PICKUP_MIN_FRAMES = 3
SHAKE_SPEED_THRESHOLD = 8.0    # mean speed threshold for shaking
SHAKE_STD_THRESHOLD = 4.5      # std of lateral motion indicating oscillation
OPEN_ROTATION_THRESHOLD = 10   # degrees of rotation change to detect open/close
PROXIMITY_PADDING = 50        # px around bbox to consider "near"
LABEL_TO_TRACK = "bottle"      # YOLO label for water bottle (COCO: "bottle")

# ----- Helper functions -----
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_bbox(pt, bbox, pad=0):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def angle_between(v1, v2):
    # v1 and v2 are (x,y)
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosang = max(min(cosang, 1.0), -1.0)
    return math.degrees(math.acos(cosang))

# ----- Main demo class -----
class BottleInteractionDemo:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = ObjectDetector()     # expects model path inside
        self.hand = HandTracker()
        self.history = deque(maxlen=MAX_HISTORY)  # entries: dict with timestamp, bottle_center, hand_point
        self.state = "idle"
        self.pickup_candidate_frames = 0
        self.last_state_change = time.time()
        self.bottle_bbox = None
        self.prev_bottle_center = None
        self.prev_hand_vector = None
        self.shake_cooldown_until = 0
        self.opened = False
        self.closed = False

    def _extract_primary_bottle(self, detections):
        # Choose the largest bottle bbox if multiple
        bottles = [d for d in detections if d["label"].lower() == LABEL_TO_TRACK]
        if not bottles:
            return None
        # largest by area
        best = max(bottles, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        return best["bbox"]

    def _get_hand_point(self, hand_entry, frame_shape):
        # Accepts variety of hand tracker outputs:
        # - if entry is (x,y) tuple, return it
        # - if dict with 'landmarks' (MediaPipe), extract index fingertip
        # - if list of tuples, take first
        h, w = frame_shape[:2]
        if isinstance(hand_entry, tuple) and len(hand_entry) == 2:
            return hand_entry
        if isinstance(hand_entry, list) and len(hand_entry) > 0:
            first = hand_entry[0]
            if isinstance(first, tuple):
                return first
        if isinstance(hand_entry, dict) and "landmarks" in hand_entry:
            # landmarks expected normalized 0..1 like MediaPipe; try to extract INDEX_FINGER_TIP (id 8)
            lm = hand_entry["landmarks"]
            if len(lm) > 8:
                l = lm[8]
                return int(l[0] * w), int(l[1] * h)
        # Fallback: return None
        return None

    def _append_history(self, bottle_center, hand_point):
        self.history.append({
            "t": time.time(),
            "bottle_center": bottle_center,
            "hand_point": hand_point
        })

    def _compute_motion_stats(self):
        # compute motion vectors of bottle center across history
        pts = [h["bottle_center"] for h in self.history if h["bottle_center"] is not None]
        if len(pts) < 2:
            return {"speeds": [], "mean_speed": 0.0, "std_x": 0.0}
        vecs = []
        for a, b in zip(pts[:-1], pts[1:]):
            vecs.append((b[0] - a[0], b[1] - a[1]))
        speeds = [math.hypot(v[0], v[1]) for v in vecs]
        xs = [v[0] for v in vecs] if vecs else [0]
        return {"speeds": speeds, "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "std_x": float(np.std(xs)) if xs else 0.0,
                "vecs": vecs}

    def update_state_machine(self, bottle_bbox, hand_point):
        now = time.time()
        bottle_center = bbox_center(bottle_bbox) if bottle_bbox is not None else None
        self._append_history(bottle_center, hand_point)

        # compute basic observations
        hand_near = False
        if bottle_bbox is not None and hand_point is not None:
            hand_near = point_in_bbox(hand_point, bottle_bbox, pad=PROXIMITY_PADDING)

        motion = self._compute_motion_stats()

        # State transitions
        if self.state == "idle":
            if bottle_bbox is not None:
                self.state = "bottle_detected"
                self.bottle_bbox = bottle_bbox
                self.last_state_change = now
                speak("Bottle detected. Please bring your hand near the bottle.")
        elif self.state == "bottle_detected":
            if hand_near:
                self.state = "hand_near_bottle"
                self.last_state_change = now
                speak("Hand detected near the bottle. You can pick it up now.")
        elif self.state == "hand_near_bottle":
            # detect pickup: upward motion while hand near
            # check last few bottle centers
            centers = [h["bottle_center"] for h in self.history if h["bottle_center"] is not None]
            if len(centers) >= PICKUP_MIN_FRAMES and hand_near:
                # compare early to latest: if bottle moved up by threshold (y decreased)
                if centers[-1][1] - centers[-PICKUP_MIN_FRAMES][1] <= PICKUP_Y_THRESHOLD:
                    self.state = "bottle_picked_up"
                    self.last_state_change = now
                    speak("Bottle picked up. Now you can shake it.")
        elif self.state == "bottle_picked_up":
            # detect shaking: lateral oscillation or high mean speed in recent frames
            ms = motion
            if ms["mean_speed"] > SHAKE_SPEED_THRESHOLD and ms["std_x"] > SHAKE_STD_THRESHOLD and now > self.shake_cooldown_until:
                self.state = "shaking"
                self.last_state_change = now
                self.shake_cooldown_until = now + 1.5  # small cooldown so we don't re-trigger too fast
                speak("I see shaking. Now try opening the bottle cap.")
        elif self.state == "shaking":
            # look for open gesture: hand near cap region + rotational change
            # cap region ~ top 30% of bbox
            if bottle_bbox is not None and hand_point is not None:
                x1, y1, x2, y2 = bottle_bbox
                cap_region = (x1, y1, x2, int(y1 + 0.35 * (y2 - y1)))
                if point_in_bbox(hand_point, cap_region, pad=20):
                    # compute finger vector relative to bottle center across history
                    # approximate hand vector using last two hand points
                    hand_pts = [h["hand_point"] for h in self.history if h["hand_point"] is not None]
                    if len(hand_pts) >= 3:
                        v_prev = (hand_pts[-2][0] - bottle_center[0], hand_pts[-2][1] - bottle_center[1])
                        v_now = (hand_pts[-1][0] - bottle_center[0], hand_pts[-1][1] - bottle_center[1])
                        ang = angle_between(v_prev, v_now)
                        print(f"[DEBUG] Rotation angle: {ang:.2f}")
                        if ang > OPEN_ROTATION_THRESHOLD:
                            self.state = "opening"
                            self.last_state_change = now
                            self.opened = True
                            speak("Opening detected. Now close the cap when done.")
        elif self.state == "opening":
            # Wait a few seconds before allowing closing detection
            if now - self.last_state_change < 2.5:
                return  # prevent immediate transition
            if bottle_bbox is not None and hand_point is not None:
                hand_pts = [h["hand_point"] for h in self.history if h["hand_point"] is not None]
                if len(hand_pts) >= 3:
                    v_prev = (hand_pts[-2][0] - bottle_center[0], hand_pts[-2][1] - bottle_center[1])
                    v_now = (hand_pts[-1][0] - bottle_center[0], hand_pts[-1][1] - bottle_center[1])
                    ang = angle_between(v_prev, v_now)
                    print(f"[DEBUG] Rotation angle: {ang:.2f}")
                    if ang > OPEN_ROTATION_THRESHOLD:
                        self.state = "closing"
                        self.last_state_change = now
                        self.closed = True
                        speak("Close detected. Great! Demo complete.")
        if self.state == "closing":
            # allow a short pause then mark done
            if now - self.last_state_change > 1.0:
                self.state = "done"

    def draw_overlay(self, frame, bottle_bbox, hand_point):
        h, w = frame.shape[:2]
        # draw bottle bbox
        if bottle_bbox is not None:
            x1, y1, x2, y2 = bottle_bbox
            color = (0, 255, 0)
            if self.state in ("hand_near_bottle", "bottle_picked_up", "shaking", "opening", "closing"):
                color = (0, 165, 255)  # orange-ish
            if self.state == "done":
                color = (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = bbox_center(bottle_bbox)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"Bottle ({self.state})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw hand point
        if hand_point is not None:
            cv2.circle(frame, hand_point, 6, (255, 0, 0), -1)
            cv2.putText(frame, "Hand", (hand_point[0] + 8, hand_point[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # draw recent bottle path
        pts = [h["bottle_center"] for h in self.history if h["bottle_center"] is not None]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (200, 200, 50), 2)

        # status text
        cv2.putText(frame, f"State: {self.state}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def run(self):
        speak("Bottle demo starting. Show a water bottle to the camera.")
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read camera frame")
                break

            # detect objects and hands
            detections = self.detector.detect(frame)  # expect list of {"label","bbox"}
            hand_entries = self.hand.track(frame)     # may be list of tuples or dicts

            # pick primary bottle and first hand point
            bottle_bbox = self._extract_primary_bottle(detections)
            hand_point = None
            if hand_entries:
                # try to get a sensible hand point from first hand
                candidate = hand_entries[0]
                p = self._get_hand_point(candidate, frame.shape)
                if p is not None:
                    hand_point = p

            # update logic
            self.update_state_machine(bottle_bbox, hand_point)

            # draw overlays
            self.draw_overlay(frame, bottle_bbox, hand_point)
            cv2.imshow("Water Bottle Interaction Demo", frame)

            # fps cap and key handling
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

            # finish if done
            if self.state == "done":
                speak("Demo finished. Good job!")
                time.sleep(1.0)
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ----- Run demo -----
if __name__ == "__main__":
    demo = BottleInteractionDemo(camera_index=0)
    demo.run()