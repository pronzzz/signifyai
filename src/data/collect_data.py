import cv2
import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import RAW_DATA_DIR

def collect_data(class_name, num_samples=100):
    """
    Collects images from webcam for a specific class.
    
    Args:
        class_name (str): Name of the gesture class (e.g., 'thumbs_up')
        num_samples (int): Number of images to collect
    """
    
    # Create directory for the class
    class_dir = RAW_DATA_DIR / class_name
    os.makedirs(class_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    print(f"Starting data collection for class '{class_name}'...")
    print("Press 's' to start/stop saving images.")
    print("Press 'q' to quit.")
    
    count = len(list(class_dir.glob("*.jpg")))
    target = count + num_samples
    recording = False
    
    # ROI coordinates
    roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Draw ROI
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        
        # Status text
        status_color = (0, 0, 255) if recording else (0, 255, 255)
        status_text = "RECORDING" if recording else "PAUSED (Press 's')"
        cv2.putText(display_frame, f"{status_text} | Count: {count}/{target}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('Data Collector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            recording = not recording
            
        if recording and count < target:
            # Save frame inside ROI
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi.size > 0:
                # Save with timestamp to avoid overwrite
                timestamp = int(time.time() * 1000)
                file_path = class_dir / f"{class_name}_{timestamp}.jpg"
                cv2.imwrite(str(file_path), roi)
                count += 1
                time.sleep(0.1) # Limit capture rate
                
        if count >= target:
            print(f"Collected {num_samples} samples for {class_name}.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect custom gesture data.")
    parser.add_argument("--class_name", type=str, required=True, help="Name of the gesture class")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    
    args = parser.parse_args()
    collect_data(args.class_name, args.samples)
