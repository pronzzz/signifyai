import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import copy
import time
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import DATA_DIR, RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH, NUM_CLASSES
from src.data.dataset import SignifyDataset
from src.data.augment import get_transforms
from src.model.architecture import SignifyVGG16

def train_model():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset Preparation
    # Ideally should point to split directories, but for now assuming flat structure or subfolders in RAW_DATA_DIR
    # We might need to adjust this depending on how the downloaded data is structured
    # For now, let's assume raw data has class folders
    
    # Check if we have data to train on
    data_path = RAW_DATA_DIR
    if not any(data_path.iterdir()):
         print(f"No data found in {data_path}. Please run src/data/download_data.py or add images manually.")
         return

    full_dataset = SignifyDataset(root_dir=data_path, transform=get_transforms('train'))
    
    # Update num_classes based on dataset
    num_classes = len(full_dataset.classes)
    print(f"Detected {num_classes} classes: {full_dataset.classes}")
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to val_dataset (This is tricky with random_split, 
    # but for simplicity we rely on the generic transform. 
    # Ideally, we'd implement a wrapper to override transform for val subset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model Setup
    model = SignifyVGG16(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # History storage
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Make sure to convert tensor scalar to float/int for JSON serialization
            epoch_acc_val = epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc_val:.4f}")
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc_val)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc_val)
            
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"New best model saved to {MODEL_PATH}")

    # Save history
    import json
    history_path = PROJECT_ROOT / "checkpoints" / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
                    
    print(f"Best val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()
