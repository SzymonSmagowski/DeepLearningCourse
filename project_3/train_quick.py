#!/usr/bin/env python3
"""
Quick training script for fast experimentation.
Uses smaller model, fewer samples, and faster sampling.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.trainers.train_diffusion import main

if __name__ == '__main__':
    # Override sys.argv to use fast config
    import sys
    sys.argv = ['train_quick.py', '--config', 'configs/diffusion_fast_test.yaml']
    main()