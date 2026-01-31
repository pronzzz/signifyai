import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import RAW_DATA_DIR, BATCH_SIZE, MODEL_PATH
from src.data.dataset import SignifyDataset
from src.data.augment import get_transforms
from src.model.architecture import SignifyVGG16

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    # Assuming we want to evaluate on a hold-out set or just validation for now
    # Ideally should use a specific test set
    if not any(RAW_DATA_DIR.iterdir()):
         print(f"No data found in {RAW_DATA_DIR}.")
         return

    dataset = SignifyDataset(root_dir=RAW_DATA_DIR, transform=get_transforms('val'))
    
    # Simple split for demo purposes (last 20% again if we don't have separate folders)
    # In a real scenario, we'd load a specific 'test' directory
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Load Model
    num_classes = len(dataset.classes)
    model = SignifyVGG16(num_classes=num_classes).to(device)
    
    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
