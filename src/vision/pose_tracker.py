import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def track(self, frame):
        """
        Returns a list of 33 keypoints: each keypoint is (x, y, z, visibility)
        Normalized 0..1
        """
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
            return keypoints
        return None