import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import MODEL_PATH, RAW_DATA_DIR
from src.model.architecture import SignifyVGG16
from src.data.augment import get_transforms
from src.data.dataset import SignifyDataset

class GestureDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_transforms('val')
        
        # Load classes dynamically if possible, else hardcode or load from saved meta
        # For this implementation, we re-instantiate the dataset just to get classes
        # In prod, we should save class names to json
        dummy_dataset = SignifyDataset(root_dir=RAW_DATA_DIR)
        self.classes = dummy_dataset.classes
        print(f"Classes: {self.classes}")
        
        self.model = SignifyVGG16(num_classes=len(self.classes)).to(self.device)
        
        path = model_path if model_path else MODEL_PATH
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")
        else:
            print(f"Warning: Model path {path} does not exist. Using random weights.")
            
        self.model.eval()

    def preprocess(self, frame, roi):
        """
        Extracts ROI, converts to PIL, applies transforms.
        """
        x, y, w, h = roi
        hand_img = frame[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_img = Image.fromarray(hand_img)
        
        # Apply transforms (resize, normalize, tensor)
        input_tensor = self.transform(pil_img)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, frame, roi):
        """
        Runs inference on the ROI of the frame.
        """
        input_tensor = self.preprocess(frame, roi)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        idx = preds.item()
        label = self.classes[idx]
        confidence = conf.item()
        
        return label, confidence

def run_realtime_inference():
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    
    # Define ROI (Region of Interest) box coordinates (top-left)
    roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Draw ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        
        # Run inference on ROI
        label, conf = detector.predict(frame, (roi_x, roi_y, roi_w, roi_h))
        
        # Display result
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('SignifyAI - Real-time Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()
