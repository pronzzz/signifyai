import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import torch
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import RAW_DATA_DIR
from src.data.dataset import SignifyDataset
from src.data.augment import get_transforms

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_training_history():
    """Reads history.json and plots Loss and Accuracy curves."""
    history_path = PROJECT_ROOT / "checkpoints" / "history.json"
    if not history_path.exists():
        print("No training history found. Please train the model first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'training_loss.png')
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'training_accuracy.png')
    plt.close()
    
    print(f"Training plots saved to {OUTPUT_DIR}")

def plot_class_distribution():
    """Plots a bar chart of class distribution in the dataset."""
    if not any(RAW_DATA_DIR.iterdir()):
        print("No dataset found.")
        return

    dataset = SignifyDataset(root_dir=RAW_DATA_DIR)
    
    # Count samples per class
    class_counts = {cls: 0 for cls in dataset.classes}
    for _, idx in dataset.samples:
        class_name = dataset.classes[idx]
        class_counts[class_name] += 1
        
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.title('Dataset Class Distribution')
    plt.xlabel('Gesture Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.png')
    plt.close()
    
    print(f"Class distribution plot saved to {OUTPUT_DIR}")

def visualize_samples(num_samples=9):
    """Visualizes a grid of random samples from the dataset."""
    if not any(RAW_DATA_DIR.iterdir()):
        print("No dataset found.")
        return

    # Use raw transforms (resize only) for visualization clarity
    dataset = SignifyDataset(root_dir=RAW_DATA_DIR, transform=get_transforms('val'))
    
    if len(dataset) == 0:
        print("Empty dataset.")
        return
        
    # Get random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    grid_size = math.ceil(math.sqrt(num_samples))
    plt.figure(figsize=(12, 12))
    
    for i, idx in enumerate(indices):
        image, label_idx = dataset[idx]
        class_name = dataset.classes[label_idx]
        
        # Un-normalize for display if using normalized transform
        # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img_np = image.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img_np)
        plt.title(class_name)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_gestures.png')
    plt.close()
    
    print(f"Sample visualization saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Generating visualizations...")
    plot_training_history()
    plot_class_distribution()
    visualize_samples()
    print("Done.")
