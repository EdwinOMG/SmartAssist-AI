import cv2
import os

video_path = "data/raw/punches/punch1.mp4"
output_dir = "data/processed/punches"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames")