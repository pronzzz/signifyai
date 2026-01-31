from torchvision import transforms

def get_transforms(stage='train'):
    """
    Returns the data transformation pipeline for training or validation.
    
    Args:
        stage (str): 'train' or 'val'/'test'
    """
    
    mean = [0.485, 0.456, 0.406] # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std
    
    if stage == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    return transform
