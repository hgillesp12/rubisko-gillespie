import cv2
import logging
import json
from src.assistant import TechAssistant
from src.models_handler import ModelsHandler
from src.display import DisplayHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Load configuration for hardware-specific settings
with open('config.json', 'r') as f:
    config = json.load(f)

def main(mode='manual'):
    """
    The main execution loop for the RUBISKO AI Assistant.
    
    This function initializes the hardware stream, loads the AI models, and 
    manages the real-time interaction loop between the camera and the user. 
    It handles keyboard interrupts for mode switching and manual scanning.

    Args:
        mode (str): Starting mode for the assistant ('manual' or 'auto'). 
                    In 'auto', the system uses optical flow to trigger scans.
                    Defaults to 'manual'.

    Returns:
        None
    """
    logging.info("Starting RUBISKO AI Assistant")
    
    # Initialize components
    handler = ModelsHandler(model_dir='models/')
    
    # Note: For the Olympus EP50, stream_url would typically be 'rtsp://<IP_ADDRESS>:554/live'
    assistant = TechAssistant(handler, stream_url=0, mode=mode, config_path='config.json') 
    ui = DisplayHandler()
    
    # Persistent state for the UI
    last_results = {}
    mag_level = "400x" 

    print("--- RUBISKO LIVE ---")
    print("Controls: [SPACE] to Scan, [M] Toggle Auto/Manual, [Q] to Quit")

    while True:
        frame = assistant.get_frame()
        if frame is None:
            logging.error("Failed to grab frame from camera.")
            break

        key = cv2.waitKey(1) & 0xFF
        
        # 1. Trigger Logic
        # Unpack the 3 values: frame, analysis results, and the detected magnification
        if (assistant.mode == 'auto' and assistant.is_stationary(frame)) or (key == ord(' ')):
            _, last_results, mag_level = assistant.run_scan(frame)
            logging.info(f"Scan complete. Detected Magnification: {mag_level}")

        # 2. Mode Toggle (Bonus functionality for better UX)
        if key == ord('m'):
            assistant.mode = 'auto' if assistant.mode == 'manual' else 'manual'
            logging.info(f"Switched mode to: {assistant.mode}")

        # 3. UI Rendering
        # We pass the results and the mag_level to the display handler
        processed_frame = ui.draw_interface(
            frame, 
            last_results, 
            mag_level
        )

        cv2.imshow('RUBISKO AI Assistant', processed_frame)
        
        if key == ord('q'): 
            break

    assistant.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(mode='manual')