import os
import kaggle
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import RAW_DATA_DIR

def download_dataset():
    """
    Downloads the Hand Gesture Recognition Dataset from Kaggle.
    """
    dataset_name = "gti-upm/leapgestrecog" # Validated Kaggle dataset name
    
    print(f"Downloading {dataset_name} to {RAW_DATA_DIR}...")
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=RAW_DATA_DIR, 
            unzip=True
        )
        print("Download complete and unzipped.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have your kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")

if __name__ == "__main__":
    download_dataset()
