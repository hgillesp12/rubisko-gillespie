from ultralytics import YOLO
import logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class ModelsHandler:
    """
    Manages the lifecycle and concurrent execution of specialized YOLO models.
    """
    def __init__(self, model_dir='models/'):
        """
        Loads the weight files for all four biological analysis models.

        Args:
            model_dir (str): Directory where .pt files are stored.
        """
        self.contamination = YOLO(f'{model_dir}contamination.pt')
        self.quality = YOLO(f'{model_dir}quality.pt')
        self.stage = YOLO(f'{model_dir}stage.pt')
        self.ratios = YOLO(f'{model_dir}ratios.pt')
        logging.info("Loaded all YOLO models successfully.")

    def run_inference(self, frame):
        """
        Performs object detection using all four models on a single frame.

        Args:
            frame (numpy.ndarray): Image to process.

        Returns:
            dict: Dictionary mapping model names to their respective Result objects.
        """
        results = {}

        results['contamination'] = self.contamination(frame, verbose=False)[0]
        results['quality'] = self.quality(frame, verbose=False)[0]
        results['stage'] = self.stage(frame, verbose=False)[0]
        results['ratios'] = self.ratios(frame, verbose=False)[0]
    
        return results