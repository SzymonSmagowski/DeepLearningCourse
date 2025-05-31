#!/usr/bin/env python3
"""
Main training script for StyleGAN2.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.trainers.train_gan import main

if __name__ == '__main__':
    main()