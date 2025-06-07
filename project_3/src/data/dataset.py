import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CatDataset(Dataset):
    """PyTorch Dataset for cat images."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        use_albumentations: bool = False,
        image_size: int = 256,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.data_dir.glob('*.png')) + 
                                 list(self.data_dir.glob('*.jpg')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(self.image_files):
            import random
            random.seed(42)
            self.image_files = random.sample(self.image_files, max_samples)
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # If no transform provided, use default
        if self.transform is None:
            self.transform = self.get_default_transform()
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.use_albumentations and hasattr(self.transform, '__call__'):
            # Convert to numpy for albumentations
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Use torchvision transforms
            image = self.transform(image)
        
        return image
    
    def get_default_transform(self) -> Callable:
        """Get default transformation pipeline."""
        if self.use_albumentations:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])


def get_augmentation_transform(
    image_size: int = 256,
    mode: str = 'train',
    use_albumentations: bool = True
) -> Callable:
    """Get augmentation transforms for different modes."""
    
    if use_albumentations:
        if mode == 'train':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:  # val/test
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
    else:
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:  # val/test
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])


class CombinedCatDogDataset(Dataset):
    """Dataset for combined cats and dogs training."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        use_albumentations: bool = False,
        image_size: int = 256,
        return_labels: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.image_size = image_size
        self.return_labels = return_labels
        
        # Get cat and dog files
        self.cat_files = sorted(list(self.data_dir.glob('cat*.jpg')))
        self.dog_files = sorted(list(self.data_dir.glob('dog*.jpg')))
        
        # Combine and create labels
        self.image_files = self.cat_files + self.dog_files
        self.labels = [0] * len(self.cat_files) + [1] * len(self.dog_files)
        
        print(f"Found {len(self.cat_files)} cats and {len(self.dog_files)} dogs")
        
        if self.transform is None:
            self.transform = get_augmentation_transform(
                image_size, 'train', use_albumentations
            )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.use_albumentations:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = self.transform(image)
        
        if self.return_labels:
            return image, label
        return image


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    use_albumentations: bool = True,
    pin_memory: bool = True,
    max_train_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = CatDataset(
        os.path.join(data_root, 'train'),
        transform=get_augmentation_transform(image_size, 'train', use_albumentations),
        use_albumentations=use_albumentations,
        image_size=image_size,
        max_samples=max_train_samples
    )
    
    val_dataset = CatDataset(
        os.path.join(data_root, 'val'),
        transform=get_augmentation_transform(image_size, 'val', use_albumentations),
        use_albumentations=use_albumentations,
        image_size=image_size
    )
    
    test_dataset = CatDataset(
        os.path.join(data_root, 'test'),
        transform=get_augmentation_transform(image_size, 'test', use_albumentations),
        use_albumentations=use_albumentations,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = CatDataset(
        "data/processed_cats/train",
        use_albumentations=True
    )
    
    # Test loading
    img = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")