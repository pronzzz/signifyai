import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class SignifyVGG16(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Args:
            num_classes (int): Number of gesture classes.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(SignifyVGG16, self).__init__()
        
        # Load pre-trained VGG16
        weights = VGG16_Weights.DEFAULT
        self.vgg16 = vgg16(weights=weights)
        
        # Freeze early convolution layers
        # Freezing all features layers first
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
            
        # Optional: Unfreeze the last block of convolution features for fine-tuning
        # This is a common strategy for transfer learning
        for param in self.vgg16.features[-4:].parameters():
            param.requires_grad = True
            
        # Replace the classifier
        # VGG16 default classifier: (25088 -> 4096) -> (4096 -> 4096) -> (4096 -> 1000)
        # We will simplify it for our specific task to reduce parameters and prevent overfitting on smaller datasets
        
        num_features = self.vgg16.classifier[0].in_features
        
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.vgg16(x)
