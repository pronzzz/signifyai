
import os
import numpy as np
import cv2
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import RAW_DATA_DIR

def generate_synthetic_data(num_classes=3, images_per_class=10):
    print(f"Generating synthetic data in {RAW_DATA_DIR}...")
    
    classes = [f"gesture_{i}" for i in range(num_classes)]
    
    for cls in classes:
        cls_dir = RAW_DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_class):
            # Create a random image
            img = np.random.randint(0, 255, (240, 640, 3), dtype=np.uint8)
            
            # Draw class name on it to make it "distinct"
            cv2.putText(img, cls, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            save_path = cls_dir / f"img_{i}.png"
            cv2.imwrite(str(save_path), img)
            
    print("Synthetic data generation complete.")

if __name__ == "__main__":
    generate_synthetic_data()
