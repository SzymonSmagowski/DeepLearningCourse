import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class Config:
    """Configuration class for managing experiment settings."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._process_config()
    
    def _process_config(self):
        """Process and validate configuration."""
        # Set device
        if not hasattr(self, 'device'):
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        # Create paths
        for key in ['checkpoint_dir', 'sample_dir', 'log_dir']:
            if hasattr(self, key):
                path = Path(getattr(self, key))
                path.mkdir(parents=True, exist_ok=True)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        return self._config.get(name, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._config.update(updates)
        self._process_config()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def get_default_diffusion_config() -> Dict[str, Any]:
    """Get default configuration for diffusion model."""
    return {
        # Model
        'model_type': 'ddpm',
        'image_size': 256,
        'in_channels': 3,
        'out_channels': 3,
        'model_channels': 128,
        'num_res_blocks': 2,
        'attention_resolutions': [16, 8],
        'dropout': 0.1,
        'channel_mult': [1, 2, 4, 8],
        'num_heads': 8,
        'use_scale_shift_norm': True,
        
        # Diffusion
        'num_timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        
        # Training
        'batch_size': 16,
        'learning_rate': 2e-4,
        'num_epochs': 1000,
        'gradient_accumulation_steps': 1,
        'ema_decay': 0.9999,
        'mixed_precision': True,
        
        # Data
        'data_root': 'data/processed_cats_256',
        'num_workers': 4,
        'pin_memory': True,
        
        # Logging
        'log_interval': 100,
        'sample_interval': 1000,
        'checkpoint_interval': 5000,
        'num_samples': 16,
        
        # Paths
        'checkpoint_dir': 'outputs/checkpoints/diffusion',
        'sample_dir': 'outputs/generated_samples/diffusion',
        'log_dir': 'outputs/logs/diffusion',
        
        # Device
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }


def get_default_gan_config() -> Dict[str, Any]:
    """Get default configuration for GAN."""
    return {
        # Model
        'model_type': 'stylegan2',
        'image_size': 256,
        'latent_dim': 512,
        'n_layers': 8,
        'channel_multiplier': 2,
        
        # Generator
        'g_channels': [512, 512, 512, 512, 256, 128, 64, 32],
        'style_dim': 512,
        'n_mlp': 8,
        
        # Discriminator
        'd_channels': [32, 64, 128, 256, 512, 512, 512, 512],
        'mbstd_group_size': 4,
        
        # Training
        'batch_size': 8,
        'g_lr': 0.002,
        'd_lr': 0.002,
        'r1_gamma': 10.0,
        'pl_weight': 2.0,
        'num_epochs': 1000,
        'g_reg_interval': 4,
        'd_reg_interval': 16,
        
        # Augmentation
        'ada_target': 0.6,
        'ada_length': 500 * 1000,
        'augment_p': 0.0,
        
        # Data
        'data_root': 'data/processed_cats_256',
        'num_workers': 4,
        'pin_memory': True,
        
        # Logging
        'log_interval': 100,
        'sample_interval': 1000,
        'checkpoint_interval': 5000,
        'num_samples': 16,
        
        # Paths
        'checkpoint_dir': 'outputs/checkpoints/gan',
        'sample_dir': 'outputs/generated_samples/gan',
        'log_dir': 'outputs/logs/gan',
        
        # Device
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }


def create_config_files():
    """Create default configuration files."""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # Diffusion configs
    diffusion_config = get_default_diffusion_config()
    
    # 256x256 config
    with open(configs_dir / 'diffusion_256.yaml', 'w') as f:
        yaml.dump(diffusion_config, f, default_flow_style=False)
    
    # 128x128 config
    diffusion_config_128 = diffusion_config.copy()
    diffusion_config_128.update({
        'image_size': 128,
        'batch_size': 32,
        'data_root': 'data/processed_cats_128',
        'checkpoint_dir': 'outputs/checkpoints/diffusion_128',
        'sample_dir': 'outputs/generated_samples/diffusion_128',
    })
    with open(configs_dir / 'diffusion_128.yaml', 'w') as f:
        yaml.dump(diffusion_config_128, f, default_flow_style=False)
    
    # GAN configs
    gan_config = get_default_gan_config()
    
    # 256x256 config
    with open(configs_dir / 'gan_256.yaml', 'w') as f:
        yaml.dump(gan_config, f, default_flow_style=False)
    
    # 128x128 config
    gan_config_128 = gan_config.copy()
    gan_config_128.update({
        'image_size': 128,
        'batch_size': 16,
        'data_root': 'data/processed_cats_128',
        'checkpoint_dir': 'outputs/checkpoints/gan_128',
        'sample_dir': 'outputs/generated_samples/gan_128',
    })
    with open(configs_dir / 'gan_128.yaml', 'w') as f:
        yaml.dump(gan_config_128, f, default_flow_style=False)


if __name__ == "__main__":
    create_config_files()
    print("Configuration files created in 'configs' directory")