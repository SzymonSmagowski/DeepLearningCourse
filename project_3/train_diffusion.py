#!/usr/bin/env python3
"""
Main training script for diffusion model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.trainers.train_diffusion import main

if __name__ == '__main__':
    main()