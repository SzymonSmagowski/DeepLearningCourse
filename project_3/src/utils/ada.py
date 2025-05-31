import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import kornia.augmentation as K


class AdaptiveAugment:
    """Adaptive Discriminator Augmentation for StyleGAN2."""
    
    def __init__(
        self,
        ada_target: float = 0.6,
        ada_length: int = 500 * 1000,
        device: str = 'cpu',
        update_every: int = 4
    ):
        self.ada_target = ada_target
        self.ada_length = ada_length
        self.device = device
        self.update_every = update_every
        
        # Initialize augmentation probability
        self.p = torch.tensor(0.0, device=device)
        self.sign_sum = torch.tensor(0.0, device=device)
        self.sign_count = 0
    
    def update(self, real_pred_signs: float):
        """Update augmentation probability based on discriminator predictions."""
        self.sign_sum += real_pred_signs
        self.sign_count += 1
        
        if self.sign_count >= self.update_every:
            # Calculate current accuracy
            current_accuracy = self.sign_sum / self.sign_count
            
            # Adjust augmentation probability
            adjust = torch.sign(current_accuracy - self.ada_target) * (self.sign_count / self.ada_length)
            self.p = (self.p + adjust).clamp(0, 1)
            
            # Reset counters
            self.sign_sum = torch.tensor(0.0, device=self.device)
            self.sign_count = 0
    
    def get_augmentation_pipeline(self) -> 'AugmentPipe':
        """Get the augmentation pipeline."""
        return AugmentPipe(self.p, self.device)


class AugmentPipe(nn.Module):
    """Augmentation pipeline for StyleGAN2-ADA."""
    
    def __init__(self, p: torch.Tensor, device: str = 'cpu'):
        super().__init__()
        self.p = p
        self.device = device
        
        # Define augmentation operations
        self.transforms = nn.ModuleDict({
            'xflip': K.RandomHorizontalFlip(p=1.0),
            'rotate': K.RandomRotation(degrees=180, p=1.0),
            'xint': K.RandomAffine(degrees=0, translate=(0.125, 0), p=1.0),
            'scale': K.RandomAffine(degrees=0, scale=(0.8, 1.25), p=1.0),
            'aniso': self.AnisotropicScale(),
            'color': K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=1.0),
            'cutout': K.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        })
        
        # Probabilities for each transform (cumulative)
        self.transform_probs = {
            'xflip': 0.1,
            'rotate': 0.2,
            'xint': 0.3,
            'scale': 0.4,
            'aniso': 0.5,
            'color': 0.7,
            'cutout': 1.0
        }
    
    class AnisotropicScale(nn.Module):
        """Anisotropic scaling augmentation."""
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            
            # Random scale factors
            scale_x = torch.empty(batch_size, device=x.device).uniform_(0.8, 1.25)
            scale_y = torch.empty(batch_size, device=x.device).uniform_(0.8, 1.25)
            
            # Apply scaling
            grid = self._get_scaling_grid(x, scale_x, scale_y)
            return F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        
        def _get_scaling_grid(
            self,
            x: torch.Tensor,
            scale_x: torch.Tensor,
            scale_y: torch.Tensor
        ) -> torch.Tensor:
            batch_size, _, height, width = x.shape
            
            # Create base grid
            y, x_coord = torch.meshgrid(
                torch.linspace(-1, 1, height, device=x.device),
                torch.linspace(-1, 1, width, device=x.device),
                indexing='ij'
            )
            
            # Apply scaling
            x_coord = x_coord.unsqueeze(0) / scale_x.view(-1, 1, 1)
            y = y.unsqueeze(0) / scale_y.view(-1, 1, 1)
            
            # Stack to create grid
            grid = torch.stack([x_coord, y], dim=-1)
            
            return grid.expand(batch_size, -1, -1, -1)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply augmentations based on current probability."""
        # Sample whether to apply augmentation for this batch
        if self.p == 0 or torch.rand(1) > self.p:
            return images
        
        # Apply augmentations based on cumulative probabilities
        p_sample = torch.rand(1).item()
        
        for transform_name, cum_prob in self.transform_probs.items():
            if p_sample <= cum_prob * self.p.item():
                # Apply this transform
                return self.transforms[transform_name](images)
        
        return images


class GridSample(nn.Module):
    """Grid sample augmentation."""
    
    def __init__(self, use_fp16: bool = False):
        super().__init__()
        self.use_fp16 = use_fp16
    
    def forward(self, images: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        """Apply grid sampling."""
        if self.use_fp16:
            with torch.cuda.amp.autocast(enabled=False):
                return F.grid_sample(
                    images.float(),
                    grids.float(),
                    mode='bilinear',
                    align_corners=False
                ).to(images.dtype)
        else:
            return F.grid_sample(
                images,
                grids,
                mode='bilinear',
                align_corners=False
            )


def get_ada_augment(ada_target: float, ada_length: int, device: str = 'cpu') -> AdaptiveAugment:
    """Create adaptive augmentation instance."""
    return AdaptiveAugment(ada_target, ada_length, device)


class DiffAugment(nn.Module):
    """Differentiable Augmentation for GAN training (alternative to ADA)."""
    
    def __init__(self, policy: str = 'color,translation,cutout'):
        super().__init__()
        self.policy = policy.split(',')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for p in self.policy:
            if p == 'color':
                x = self._color_augment(x)
            elif p == 'translation':
                x = self._translation_augment(x)
            elif p == 'cutout':
                x = self._cutout_augment(x)
        return x
    
    def _color_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random color jittering."""
        batch_size = x.shape[0]
        
        # Brightness
        brightness = torch.rand(batch_size, 1, 1, 1, device=x.device) * 0.5 + 0.75
        x = x * brightness
        
        # Saturation
        mean = x.mean(dim=1, keepdim=True)
        saturation = torch.rand(batch_size, 1, 1, 1, device=x.device) * 2
        x = mean + (x - mean) * saturation
        
        return x.clamp(0, 1)
    
    def _translation_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random translation."""
        batch_size, _, height, width = x.shape
        
        # Random shifts
        shift_x = torch.randint(-width // 8, width // 8 + 1, (batch_size,), device=x.device)
        shift_y = torch.randint(-height // 8, height // 8 + 1, (batch_size,), device=x.device)
        
        # Apply shifts
        base_grid = self._get_base_grid(batch_size, height, width, x.device)
        shift_grid = torch.stack([
            base_grid[..., 0] + 2 * shift_x.float().view(-1, 1, 1) / width,
            base_grid[..., 1] + 2 * shift_y.float().view(-1, 1, 1) / height
        ], dim=-1)
        
        return F.grid_sample(x, shift_grid, mode='bilinear', align_corners=False)
    
    def _cutout_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random cutout."""
        batch_size, _, height, width = x.shape
        
        # Random cutout parameters
        cutout_size = height // 4
        offset_x = torch.randint(0, width - cutout_size + 1, (batch_size,), device=x.device)
        offset_y = torch.randint(0, height - cutout_size + 1, (batch_size,), device=x.device)
        
        # Create masks
        masks = torch.ones_like(x)
        for i in range(batch_size):
            masks[i, :, offset_y[i]:offset_y[i] + cutout_size, offset_x[i]:offset_x[i] + cutout_size] = 0
        
        return x * masks
    
    def _get_base_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get base grid for transformations."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)