import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image

class SignifyDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            classes (list, optional): List of class names.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = classes if classes else sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Walk through the directory and collect images
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
                
            for img_path in cls_dir.glob("*.png"): # Adjust extension as needed (jpg, png, etc.)
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
                
            # Also check for jpg
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
