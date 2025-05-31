#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.models.diffusion_unet import UNet
        from src.models.ddpm import DDPM
        from src.models.stylegan2 import Generator, Discriminator, MappingNetwork
        from src.data.dataset import CatDataset, create_dataloaders
        from src.utils.config import Config, get_default_diffusion_config, get_default_gan_config
        from src.utils.helpers import setup_device
        from src.utils.ada import AdaptiveAugment
        print("✓ All imports successful!")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_data_loading():
    """Test if data loading works."""
    print("\nTesting data loading...")
    
    try:
        from src.data.dataset import CatDataset
        
        # Check if processed data exists
        data_paths = [
            "data/processed_cats_256/train",
            "data/processed_cats_128/train"
        ]
        
        for path in data_paths:
            if Path(path).exists():
                dataset = CatDataset(path, image_size=256)
                print(f"✓ Found {len(dataset)} images in {path}")
                
                # Test loading one image
                img = dataset[0]
                print(f"✓ Successfully loaded image with shape: {img.shape}")
                break
        else:
            print("✗ No processed data found. Please run preprocess_data.py first.")
            return False
            
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False
    
    return True


def test_model_creation():
    """Test if models can be created."""
    print("\nTesting model creation...")
    
    try:
        from src.models.diffusion_unet import UNet
        from src.models.ddpm import DDPM
        from src.utils.helpers import setup_device
        
        device = setup_device()
        
        # Create UNet
        unet = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=64,  # Small for testing
            num_res_blocks=1,
            attention_resolutions=[16],
            channel_mult=[1, 2],
            image_size=128
        ).to(device)
        
        # Create DDPM
        model = DDPM(
            unet=unet,
            num_timesteps=100,  # Small for testing
            device=device
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 128, 128).to(device)
        t = torch.randint(0, 100, (2,)).to(device)
        
        with torch.no_grad():
            output = unet(x, t)
        
        print(f"✓ Model created successfully!")
        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False
    
    return True


def test_config():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from src.utils.config import Config, get_default_diffusion_config
        
        # Test default config
        config_dict = get_default_diffusion_config()
        config = Config(config_dict)
        
        print(f"✓ Configuration loaded successfully!")
        print(f"  - Image size: {config.image_size}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Device: {config.device}")
        
        # Test loading from yaml
        config_path = Path("configs/diffusion_128.yaml")
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
            print(f"✓ Loaded configuration from {config_path}")
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    return True


def test_gan_model():
    """Test if GAN models can be created."""
    print("\nTesting GAN model creation...")
    
    try:
        from src.models.stylegan2 import Generator, Discriminator, MappingNetwork
        from src.utils.helpers import setup_device
        
        device = setup_device()
        
        # Create models
        generator = Generator(size=64, style_dim=128, n_mlp=4)
        # Get the number of w vectors needed from generator
        num_ws = generator.n_latent
        mapping = MappingNetwork(z_dim=128, w_dim=128, num_ws=num_ws)
        discriminator = Discriminator(size=64)
        
        # Move to device
        mapping = mapping.to(device)
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        # Test forward pass
        z = torch.randn(2, 128).to(device)
        w = mapping(z)
        # Split w into individual style vectors for each layer
        w_split = [w[:, i] for i in range(w.shape[1])]
        fake_img = generator(w_split)
        d_out = discriminator(fake_img)
        
        print(f"✓ GAN models created successfully!")
        print(f"✓ Generated image shape: {fake_img.shape}")
        print(f"✓ Discriminator output shape: {d_out.shape}")
        
        # Count parameters
        g_params = sum(p.numel() for p in generator.parameters())
        d_params = sum(p.numel() for p in discriminator.parameters())
        m_params = sum(p.numel() for p in mapping.parameters())
        print(f"✓ Total GAN parameters: {(g_params + d_params + m_params):,}")
        
    except Exception as e:
        print(f"✗ GAN model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Project Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_model_creation,
        test_gan_model,
        test_config
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Run training with: python train_diffusion.py --config configs/diffusion_128.yaml")
        print("2. Monitor training with: tensorboard --logdir outputs/logs")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 50)


if __name__ == "__main__":
    main()