import json
import cv2

def load_config(path):
    """
    Loads the system configuration and calibration data from a JSON file.

    Args:
        path (str): File path to config.json.

    Returns:
        dict: Parsed configuration settings.
    """
    with open(path, 'r') as f:
        return json.load(f)

def get_physical_metrics(pixel_width, magnification, config):
    """
    Translates pixel measurements into real-world microns using calibration ratios.

    Args:
        pixel_width (float): The width of the object in pixels.
        magnification (str): The current lens level ("400x"/"1000x").
        config (dict): The configuration dictionary containing ppm values.

    Returns:
        float: The calculated size in microns (um).
    """
    ppm = config['calibration'][magnification]['pixels_per_micron']
    return round(pixel_width / ppm, 2)

def below_threshold(current_gray, prev_gray, threshold=5000):
    """
    Calculates the absolute difference between frames to detect motion.

    Args:
        current_gray (numpy.ndarray): Current frame in grayscale.
        prev_gray (numpy.ndarray): Previous frame in grayscale.
        threshold (int): Sensitivity level for motion detection.

    Returns:
        bool: True if the frame is stationary (difference < threshold).
    """
    if prev_gray is None: return False
    diff = cv2.absdiff(current_gray, prev_gray)
    return cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]) < threshold