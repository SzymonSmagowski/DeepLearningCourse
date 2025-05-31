import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
from pathlib import Path
import torchvision.utils as vutils


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = (-1, 1)
):
    """Save a grid of images."""
    if normalize and value_range is not None:
        # Denormalize from [-1, 1] to [0, 1]
        images = (images - value_range[0]) / (value_range[1] - value_range[0])
    
    # Clamp values
    images = torch.clamp(images, 0, 1)
    
    # Make grid
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL and save
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(save_path)


def visualize_batch(
    images: torch.Tensor,
    title: str = "Image Batch",
    nrow: int = 8,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
):
    """Visualize a batch of images."""
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Make grid
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_path: str,
    ema_model: Optional[torch.nn.Module] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    additional_info: Optional[dict] = None
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[torch.nn.Module] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def get_cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_logger(log_dir: str, name: str = "train") -> 'logging.Logger':
    """Create a logger for training."""
    import logging
    from datetime import datetime
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_path = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def setup_device(device_preference: Optional[str] = None) -> torch.device:
    """Setup and return the best available device."""
    if device_preference:
        return torch.device(device_preference)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def denormalize(images: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """Denormalize images."""
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
    return images * std + mean