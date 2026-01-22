import cv2
import numpy as np
from src.display import DisplayHandler

# 1. Create a dummy YOLO-like box object to mimic Ultralytics output
class MockBox:
    def __init__(self, x1, y1, x2, y2):
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.xywh = [np.array([0, 0, x2-x1, y2-y1])] # Width calculation

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.microns = [] # This gets populated by our logic

def test_display():
    ui = DisplayHandler()
    
    # Create a blank "frame" (1080p black image)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create mock detections
    # One for quality, one for contamination
    quality_boxes = MockResult([MockBox(500, 400, 700, 600)])
    quality_boxes.microns = [80.5] # Fake micron measurement
    
    contamination_boxes = MockResult([MockBox(200, 200, 250, 250)])
    contamination_boxes.microns = [12.2]
    
    results = {
        'quality': quality_boxes,
        'contamination': contamination_boxes
    }

    # Render the frame with 1000x magnification
    processed_frame = ui.draw_interface(frame, results, "1000x")
    
    # Show it to the user
    print("Displaying test window. Press any key to close.")
    cv2.imshow('UI Test Calibration', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_display()