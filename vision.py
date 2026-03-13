import cv2
from deepface import DeepFace
import numpy as np
import base64

class VisionModule:
    """
    Handles capturing frames from the webcam and generating facial embeddings.
    """
    def __init__(self, model_name="Facenet"):
        self.model_name = model_name
        print(f"VisionModule initialized using {self.model_name} model.")

    def capture_frame(self):
        """
        Captures a single frame from the default webcam.
        Returns:
            frame: Captured image frame (numpy array) or None if failed.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None

        # Warm up the camera
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame
        else:
            print("Error: Could not read frame from webcam.")
            return None

    def generate_embeddings(self, frame) -> list:
        """
        Generates facial embeddings for all detected faces in the given frame.
        Args:
            frame: Numpy array representing the image.
        Returns:
            list: A list of dicts. Each dict contains:
                  - 'embedding': list of floats
                  - 'box': dict with 'x', 'y', 'w', 'h'
                  Returns empty list if no faces detected.
        """
        if frame is None:
            return []

        try:
            # DeepFace.represent returns a list of dictionaries (one for each face).
            representations = DeepFace.represent(
                img_path=frame, 
                model_name=self.model_name, 
                enforce_detection=False
            )
            
            faces = []
            for rep in representations:
                # Calculate w and h from x, y, left_eye, right_eye if DeepFace doesn't provide standard w/h natively
                # Actually, DeepFace provides `facial_area` containing x, y, w, h
                area = rep.get("facial_area", {})
                faces.append({
                    "embedding": rep["embedding"],
                    "box": {
                        "x": area.get("x", 0),
                        "y": area.get("y", 0),
                        "w": area.get("w", 0),
                        "h": area.get("h", 0)
                    }
                })
            return faces
                
        except ValueError as e:
            # Usually happens if no face is detected
            print(f"Face detection failed: {e}")
            return []

    def generate_embeddings_from_base64(self, base64_string) -> list:
        """
        Decodes a base64 encoded image string and generates facial embeddings for all faces.
        Args:
            base64_string: The base64 string (e.g., from a canvas data URI).
        Returns:
            list: A list of dicts containing embeddings and bounding boxes.
        """
        if not base64_string:
            return []
            
        try:
            # Strip standard data URI prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return self.generate_embeddings(frame)
        except Exception as e:
            print(f"Failed to process base64 image: {e}")
            return []

# Simple test block
if __name__ == "__main__":
    vision = VisionModule()
    print("Capturing frame...")
    frame = vision.capture_frame()
    if frame is not None:
        print("Frame captured. Generating embeddings...")
        faces = vision.generate_embeddings(frame)
        if faces:
            print(f"Successfully generated {len(faces)} face(s).")
            for i, f in enumerate(faces):
                print(f"Face {i+1} at {f['box']}")
        else:
            print("Failed to generate embedding or no faces detected.")
