import os
import shutil
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import random


class DataPreprocessor:
    """Handles data preprocessing for cat image generation project."""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (256, 256),
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, val, and test ratios must sum to 1.0"
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def analyze_dataset(self) -> dict:
        """Analyze the dataset to get statistics."""
        stats = {
            'total_images': 0,
            'resolutions': {},
            'aspect_ratios': [],
            'mean_resolution': None,
            'min_resolution': None,
            'max_resolution': None,
            'corrupted_files': []
        }
        
        image_files = list(self.source_dir.rglob('*.png')) + \
                     list(self.source_dir.rglob('*.jpg')) + \
                     list(self.source_dir.rglob('*.jpeg'))
        
        resolutions = []
        
        for img_path in tqdm(image_files, desc="Analyzing dataset"):
            try:
                with Image.open(img_path) as img:
                    stats['total_images'] += 1
                    resolution = img.size
                    resolutions.append(resolution)
                    
                    res_str = f"{resolution[0]}x{resolution[1]}"
                    stats['resolutions'][res_str] = stats['resolutions'].get(res_str, 0) + 1
                    
                    aspect_ratio = resolution[0] / resolution[1]
                    stats['aspect_ratios'].append(aspect_ratio)
                    
            except Exception as e:
                stats['corrupted_files'].append((str(img_path), str(e)))
        
        if resolutions:
            stats['mean_resolution'] = (
                int(np.mean([r[0] for r in resolutions])),
                int(np.mean([r[1] for r in resolutions]))
            )
            stats['min_resolution'] = (
                min(r[0] for r in resolutions),
                min(r[1] for r in resolutions)
            )
            stats['max_resolution'] = (
                max(r[0] for r in resolutions),
                max(r[1] for r in resolutions)
            )
        
        return stats
    
    def preprocess_and_split(self, analyze_first: bool = True):
        """Preprocess images and split into train/val/test sets."""
        if analyze_first:
            print("Analyzing dataset...")
            stats = self.analyze_dataset()
            print(f"\nDataset Statistics:")
            print(f"Total images: {stats['total_images']}")
            print(f"Mean resolution: {stats['mean_resolution']}")
            print(f"Min resolution: {stats['min_resolution']}")
            print(f"Max resolution: {stats['max_resolution']}")
            print(f"Corrupted files: {len(stats['corrupted_files'])}")
            print(f"Most common resolutions:")
            sorted_res = sorted(stats['resolutions'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
            for res, count in sorted_res:
                print(f"  {res}: {count} images")
        
        # Create output directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Get all valid image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(self.source_dir.rglob(ext)))
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        # Split files
        splits_dict = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Process and save images
        for split_name, files in splits_dict.items():
            print(f"\nProcessing {split_name} split ({len(files)} images)...")
            
            for idx, img_path in enumerate(tqdm(files, desc=f"Processing {split_name}")):
                try:
                    # Open and convert to RGB
                    img = Image.open(img_path).convert('RGB')
                    
                    # Resize with high-quality resampling
                    img_resized = self.resize_image(img)
                    
                    # Save with new name
                    output_path = self.output_dir / split_name / f"{split_name}_{idx:06d}.png"
                    img_resized.save(output_path, 'PNG', quality=95)
                    
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
        
        print("\nPreprocessing complete!")
        self._save_dataset_info()
    
    def resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio and center crop."""
        # Calculate resize dimensions to maintain aspect ratio
        width, height = img.size
        target_w, target_h = self.target_size
        
        # Calculate scale to fill the target size
        scale = max(target_w / width, target_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_w) // 2
        top = (new_height - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        return img_cropped
    
    def _save_dataset_info(self):
        """Save dataset information for reproducibility."""
        info = {
            'source_dir': str(self.source_dir),
            'target_size': self.target_size,
            'splits': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'seed': self.seed,
            'num_images': {
                split: len(list((self.output_dir / split).glob('*.png')))
                for split in ['train', 'val', 'test']
            }
        }
        
        import json
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)


def get_normalization_stats(data_dir: Path, num_samples: int = 1000) -> Tuple[List[float], List[float]]:
    """Calculate mean and std for normalization."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Sample random images
    image_files = list(data_dir.glob('*.png'))
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Calculate statistics
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for img_path in tqdm(sample_files, desc="Calculating normalization stats"):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        for c in range(3):
            mean[c] += img_tensor[c].mean()
            std[c] += img_tensor[c].std()
        total_samples += 1
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        source_dir="data/cat-dataset/Data",
        output_dir="data/processed_cats",
        target_size=(256, 256),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    preprocessor.preprocess_and_split(analyze_first=True)