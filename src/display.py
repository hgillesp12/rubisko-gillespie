import cv2

class DisplayHandler:
    """
    Handles the rendering of the Augmented Reality (AR) overlay on the 
    live microscope feed.
    """
    def __init__(self):
        """Initializes display constants like fonts and color schemes."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (0, 255, 0)   # Green for info
        self.alert_color = (0, 0, 255)     # Red for contamination
        self.accent_color = (255, 255, 0)  # Cyan for measurements

    def draw_interface(self, frame, results, magnification):
        """
        Draws bounding boxes, labels, micron sizes, and the HUD onto the frame.

        Args:
            frame (numpy.ndarray): The raw BGR frame from the camera.
            results (dict): Enriched YOLO results containing micron data.
            magnification (str): The current magnification level to display.

        Returns:
            numpy.ndarray: The frame with all visual overlays applied.
        """
        # 1. Magnification Badge (Top-Left)
        cv2.rectangle(frame, (10, 10), (180, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"MAG: {magnification}", (20, 40), 
                    self.font, 0.8, self.primary_color, 2)

        # 2. Iterate through all model results (quality, contamination, etc.)
        for category, data in results.items():
            color = self.alert_color if category == 'contamination' else self.primary_color
            
            # YOLO results objects store boxes in .boxes
            for i, box in enumerate(data.boxes):
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 3. Micron Label (Calculated in TechAssistant)
                # We pull the micron value we calculated during the scan
                if hasattr(data, 'microns') and data.microns[i] is not None:
                    label = f"{category.upper()}: {data.microns[i]}um"
                else:
                    label = f"{category.upper()}"

                # Draw Label Background
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7), 
                            self.font, 0.5, (255, 255, 255), 1)

        # 4. Scale Bar (Bottom-Right) - Visual reference
        self._draw_scale_bar(frame, magnification)
        
        return frame

    def _draw_scale_bar(self, frame, magnification):
        """
        Renders a physical scale bar (e.g., 100um) for visual reference.

        Args:
            frame (numpy.ndarray): Frame to draw on.
            magnification (str): Used to determine pixel-length of the bar.
        """
        bar_width_px = 250 if magnification == "1000x" else 100 
        cv2.line(frame, (frame.shape[1]-300, frame.shape[1]-50), 
                 (frame.shape[1]-300+bar_width_px, frame.shape[1]-50), (255, 255, 255), 3)
        cv2.putText(frame, "100um", (frame.shape[1]-300, frame.shape[1]-65), 
                    self.font, 0.6, (255, 255, 255), 1)