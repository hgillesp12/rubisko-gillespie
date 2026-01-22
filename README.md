## RUBISKO: Seaweed Seed Quality AI Assistant

This repository contains an automated computer vision pipeline designed to replace manual microscope examinations of seaweed seed material. By integrating with an Olympus CKX41 Microscope and EP50 Camera, the system standardizes quality scoring, contamination detection, and biological analysis.

### Project Architecture
The codebase is modularized to separate camera hardware abstraction, AI inference, and the technician's user interface.

### Directory Structure
- **main.py**: The entry point. Manages the high-level application loop and user input (Keyboard/Mouse)
- **config.json**: Centralized configuration for hardware calibration and AI thresholds.
- **.src/**: The core application logic.
    - **assistant.py**: The `TechAssistant` class. Orchestrates the "state" of the examination (Stationary vs. Moving) and triggers scans.
    - **models_handler.py**: The `ModelsHandler` class. Encapsulates the four YOLO models, managing concurrent inference and resource allocation.
    - **display.py**: The `DisplayHandler` class. Manages the OpenCV-based UI, overlays, and status sidebars.
    - **utils.py**: Contains helper functions for optical flow, motion detection, and image pre-processing.
- **models/**: Stores the serialized `.pt` (PyTorch) weights for the four specialized AI models.
- **training/**:
    - **train_models.py**: Script for fine-tuning YOLOv8 on custom seaweed datasets.
    - ***.yaml**: Configuration files (Contamination, Quality, Stage, Sex Ratio) defining dataset paths and class names.
- **datasets/**: Local storage for training images and labels formatted in YOLO format.
- **requirements.txt**: Project dependencies (Ultralytics, OpenCV, Torch).

### The AI Models
The system utilizes four distinct models to provide a comprehensive quality assessment:
- Contamination: Detects bacteria, fungi, and microalgae.
- Quality Scoring: Assesses tissue health (healthy vs. necrotic).
- Developmental Stage: Categorizes lifecycle progress (Spore to Mature).
- Sex Ratios: Identifies male vs. female reproductive structures.

### Getting Started 
1. Installation
```
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

2. Running the Assistant

Ensure the EP50 camera is connected via Ethernet/WLAN and the RTSP stream is accessible.
```
python main.py
```
**Controls**:
- **m**: Toggle between AUTO (motion-detected) and MANUAL mode.
- **SPACE**: Trigger a Full Deep Scan (Manual mode).
- **q**: Quit the application.

### Training Pipeline
To retrain models with new lab data:
1. Add images to datasets/<model_name>/images.
2. Add labels (YOLO format) to datasets/<model_name>/labels
3. Update the corresponding .yaml file in the training/ folder.
4. Run the training script:
```python
python training/train_models.py
```

### User Interface
The DisplayHandler provides a real-time augmented view of the microscope feed:
- **Dynamic Overlays**: Bounding boxes are color-coded (e.g., Red for Contamination, Green for Health).
- **Physical Metrics**: Labels display the classification and the calculated physical size in $\mu m$.
- **Status HUD**: Displays the current operating mode (Auto/Manual) and detected magnification level.
- **Scale Bar**: A digital scale bar is rendered in the bottom corner, providing a visual reference that updates automatically as the technician switches lenses.

### Engineering Considerations
- **Multi-Modal Triggering**: Two modalities are available to suit the lab workflow:
    - **Manual**: The technician positions the slide and triggers a scan. Results are displayed for immediate review and approval.
    - **Auto**: Data is collected continuously. High-accuracy scans are filtered via the is_stationary heuristic (using Optical Flow), ensuring analysis only triggers when the slide is still.
- **Automated Magnification Inference**: The system automatically distinguishes between $400\times$ and $1000\times$ magnification by analyzing the median pixel-width of detected cells.
- **Objective Calibration**: Using `config.json`, the system converts pixel measurements into microns ($\mu m$). This ensures that data is hardware-agnostic and provides scientifically valid metrics regardless of the camera resolution.
- **Training Data**: RUBISKO "Golden Datasets" are utilized across all four models to ensure consistency and high precision in specialized biological environments. 

### Infrastructure Still Needed
1. Robust Testing: 
Additional testing is required to ensure the system is reliable and production-ready. This should include unit tests (for core functions such as frame differencing and thresholding), functional tests (validating end-to-end behaviour across typical technician workflows), and integration tests (ensuring compatibility between the computer vision pipeline, model inference, and output logging). A lightweight framework such as Pytest would be a strong option due to its quick integration with Python projects, clear test structure, and ability to scale coverage as the codebase grows.

2. Data Pipeline: 
The datasets used for training and evaluation would ideally be constructed via a dedicated data pipeline. This pipeline would automate the ingestion of raw microscope imagery and associated labels (e.g., stationarity state, quality flags, technician annotations), then normalise the data into a consistent format for training and inference. A key output would be a reproducible 80:20 train/validation split, along with dataset versioning to support model comparison and traceability.

3. Docker Deployment: 
To scale this system across multiple machines and ensure consistent environments, the project would benefit from containerisation. A lightweight abstraction tool such as Docker would allow the full runtime dependencies (OpenCV, NumPy, model weights, and Python environment) to be packaged in a portable and reproducible way. This would reduce setup errors across technician devices and allow the system to be deployed reliably across compute environments (e.g., developer machines, shared lab workstations, or cloud-based inference).

4. User Feedback: 
A user feedback layer should be incorporated to support real-world technician workflows and improve data quality. This would enable the technician to review, adjust, or override detected outcomes (e.g., “stationary” vs “moving”, label confidence, or frame quality) before the information is saved. This step would help reduce incorrect labels, support trust in the system, and provide a mechanism for creating higher-quality training data over time.

