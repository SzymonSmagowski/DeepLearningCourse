import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

from .diffusion_unet import UNet
from ..utils.helpers import get_linear_beta_schedule, get_cosine_beta_schedule


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        unet: UNet,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        loss_type: str = 'l2',
        device: str = 'cpu'
    ):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.device = device
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = get_linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = get_cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Move to device
        betas = betas.to(device)
        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)
        alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # Precompute values for x_0 prediction
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process - add noise to data."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_mean_variance(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean and variance for p(x_{t-1} | x_t)."""
        batch_size = x.shape[0]
        
        # Predict noise
        pred_noise = self.unet(x, t)
        
        # Get x_0 prediction
        x_recon = self._predict_xstart_from_eps(x, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1, 1)
        
        # Calculate posterior mean and variance
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * x_recon +
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t)."""
        mean, _, log_variance = self.p_mean_variance(x, t, clip_denoised)
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        channels: int = 3,
        height: int = 256,
        width: int = 256,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """Generate samples from the model."""
        device = next(self.parameters()).device
        shape = (batch_size, channels, height, width)
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        imgs = [img]
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            img = self.p_sample(
                img, 
                torch.full((batch_size,), t, device=device, dtype=torch.long),
                clip_denoised=True
            )
            if return_all_timesteps:
                imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return img
    
    def training_loss(self, x_start: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate training loss."""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        pred_noise = self.unet(x_noisy, t)
        
        # Calculate loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_noise, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            'loss': loss,
            'pred_noise': pred_noise,
            'noise': noise,
            'x_noisy': x_noisy,
            't': t
        }
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a 1D tensor for a batch of indices."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


class EMA_DDPM(DDPM):
    """DDPM with Exponential Moving Average."""
    
    def __init__(self, *args, ema_decay: float = 0.9999, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        
        # Create EMA model
        self.ema_unet = UNet(
            in_channels=self.unet.in_channels,
            out_channels=self.unet.out_channels,
            model_channels=self.unet.model_channels,
            num_res_blocks=self.unet.num_res_blocks,
            attention_resolutions=self.unet.attention_resolutions,
            dropout=self.unet.dropout,
            channel_mult=self.unet.channel_mult,
            num_heads=self.unet.num_heads,
            image_size=self.unet.image_size
        ).to(self.device)
        
        # Copy parameters
        self.ema_unet.load_state_dict(self.unet.state_dict())
        
        # Disable gradient computation for EMA model
        for param in self.ema_unet.parameters():
            param.requires_grad = False
    
    def update_ema(self):
        """Update EMA parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    @torch.no_grad()
    def sample(self, *args, use_ema: bool = True, **kwargs):
        """Sample using EMA model if specified."""
        if use_ema:
            # Temporarily replace unet with ema_unet
            original_unet = self.unet
            self.unet = self.ema_unet
            samples = super().sample(*args, **kwargs)
            self.unet = original_unet
            return samples
        else:
            return super().sample(*args, **kwargs)