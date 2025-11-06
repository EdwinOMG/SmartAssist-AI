steps = [
    "Locate the tire iron.",
    "Loosen the lug nuts slightly.",
    "Place the jack under the car frame and lift the car.",
    "Remove the lug nuts and take off the tire.",
    "Align and mount the new tire.",
    "Tighten the lug nuts firmly in a cross pattern.",
]

class TireChangeTask:
    def __init__(self):
        self.current_step = 0

    def get_current_instruction(self):
        return steps[self.current_step]

    def next_step(self):
        if self.current_step < len(steps) - 1:
            self.current_step += 1

    def is_complete(self):
        return self.current_step == len(steps) - 1