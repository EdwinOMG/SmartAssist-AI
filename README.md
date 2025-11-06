# SmartAssist AI

A real time computer vision system that combines hand tracking, object detection, and interaction recognition.

-Hand tracking with MediaPipe
-Object detection with Ultralytics YOLOv8 model
-Fusion module combining both to see interaction

This is the base of a hopefully larger project. Lets see where this goes!

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/vision-assisted-ai.git
cd vision-assisted-ai
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python hand_tracking/demo_hand_tracking.py
python object_detection/demo_object_detection.py
python fusion_demo/fusion_app.py