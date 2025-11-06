from src.vision.fusion import FusionTracker
from src.tasks.tire_change import TireChangeTask
from src.utils.voice_feedback import speak
import cv2
def main():
    fusion = FusionTracker()
    task = TireChangeTask()

    speak("Starting AI Task Assistant. Let's begin changing the tire.")

    while True:
        frame, detected_objects, hand_position = fusion.process_frame()
        if frame is None:
            break

        current_instruction = task.get_current_instruction()
        fusion.display_instruction(frame, current_instruction)

        # Example rule for step advancement
        if "tire iron" in detected_objects and fusion.is_hand_near("tire iron"):
            task.next_step()
            speak(f"Good job! {task.get_current_instruction()}")

        if task.is_complete():
            speak("Tire change complete!")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    fusion.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()