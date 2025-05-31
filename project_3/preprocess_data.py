#!/usr/bin/env python3
"""
Script to preprocess the cat dataset for the image generation project.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.preprocessing import DataPreprocessor


def main():
    # Configuration
    configs = [
        {
            'name': 'cats_256',
            'source_dir': 'data/cat-dataset/Data',
            'output_dir': 'data/processed_cats_256',
            'target_size': (256, 256)
        },
        {
            'name': 'cats_128',
            'source_dir': 'data/cat-dataset/Data',
            'output_dir': 'data/processed_cats_128',
            'target_size': (128, 128)
        }
    ]
    
    # Process each configuration
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Processing {config['name']}...")
        print(f"{'='*50}")
        
        preprocessor = DataPreprocessor(
            source_dir=config['source_dir'],
            output_dir=config['output_dir'],
            target_size=config['target_size'],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )
        
        # Check if already processed
        output_path = Path(config['output_dir'])
        if output_path.exists() and len(list(output_path.rglob('*.png'))) > 0:
            print(f"Dataset {config['name']} already processed. Skipping...")
            continue
        
        # Process the dataset
        preprocessor.preprocess_and_split(analyze_first=True)
    
    print("\n" + "="*50)
    print("All preprocessing complete!")
    print("="*50)


if __name__ == "__main__":
    main()