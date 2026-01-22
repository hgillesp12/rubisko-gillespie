import cv2
import numpy as np
from .utils import below_threshold, load_config, get_physical_metrics

class TechAssistant:
    """
    Orchestrates the microscope analysis workflow, including motion detection,
    magnification estimation, and result enrichment.
    """
    def __init__(self, handler, stream_url, mode='manual', config_path='config.json'):
        """
        Initializes the Assistant with hardware and AI components.

        Args:
            handler (ModelsHandler): The object managing the YOLO models.
            stream_url (str/int): The RTSP URL for the EP50 or local camera index.
            mode (str): Operational mode, either 'manual' or 'auto'.
            config_path (str): Path to the JSON file containing calibration data.
        """
        self.handler = handler
        self.config = load_config(config_path)
        self.cap = cv2.VideoCapture(stream_url)
        self.prev_frame = None
        self.mode = mode 
        self.current_mag = "400x" 

    def get_frame(self):
        """
        Captures a single frame from the microscope camera stream.

        Returns:
            numpy.ndarray: The captured BGR image frame, or None if capture fails.
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def is_stationary(self, frame) -> bool:
        """
        Determines if the microscope slide is currently still using Optical Flow.

        Args:
            frame (numpy.ndarray): The current video frame.

        Returns:
            bool: True if motion is below the defined threshold, False otherwise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stationary = below_threshold(gray, self.prev_frame)
        self.prev_frame = gray
        return stationary

    def _estimate_magnification(self, quality_results):
        """
        Infers the current objective lens (400x vs 1000x) based on cell pixel-width.

        Args:
            quality_results (ultralytics.engine.results.Results): YOLO output.

        Returns:
            str: The estimated magnification string ("400x" or "1000x").
        """
        if len(quality_results.boxes) == 0:
            return self.current_mag

        widths = quality_results.boxes.xywh[:, 2].cpu().numpy()
        median_width = np.median(widths)
        threshold = self.config['calibration']['1000x']['mag_detection_threshold_min']
        
        return "1000x" if median_width >= threshold else "400x"

    def run_scan(self, frame):
        """
        Executes a full AI analysis pass, updates magnification, and converts 
        pixel detections to physical microns.

        Args:
            frame (numpy.ndarray): The image frame to analyze.

        Returns:
            tuple: (Original frame, Dict of enriched Results, Current magnification).
        """
        analysis = self.handler.run_inference(frame)
        self.current_mag = self._estimate_magnification(analysis['quality'])
        
        for key in analysis:
            analysis[key].microns = [
                get_physical_metrics(box.xywh[0][2].item(), self.current_mag, self.config)
                for box in analysis[key].boxes
            ]

        return frame, analysis, self.current_mag