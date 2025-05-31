import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
import numpy as np


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler for faster sampling."""
    
    def __init__(self, ddpm_model: nn.Module, num_sampling_steps: int = 50):
        self.model = ddpm_model
        self.num_sampling_steps = num_sampling_steps
        
        # Create sampling schedule
        self.sampling_timesteps = self._make_sampling_schedule()
    
    def _make_sampling_schedule(self):
        """Create a schedule for DDIM sampling."""
        # Use linear spacing in the original timestep space
        timesteps = np.linspace(0, self.model.num_timesteps - 1, self.num_sampling_steps)
        return torch.from_numpy(timesteps.astype(np.int64)).flip(0)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        channels: int = 3,
        height: int = 256,
        width: int = 256,
        eta: float = 0.0,  # eta=0 for deterministic sampling
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Fast sampling using DDIM."""
        shape = (batch_size, channels, height, width)
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        for i in tqdm(range(len(self.sampling_timesteps)), desc='DDIM Sampling'):
            t = self.sampling_timesteps[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            pred_noise = self.model.unet(x, t_batch)
            
            # DDIM update
            if i < len(self.sampling_timesteps) - 1:
                t_next = self.sampling_timesteps[i + 1]
            else:
                t_next = torch.tensor(0)
            
            x = self._ddim_step(x, pred_noise, t, t_next, eta)
        
        return x
    
    def _ddim_step(self, x_t, pred_noise, t, t_next, eta=0.0):
        """Single DDIM denoising step."""
        # Get alphas
        alpha_t = self.model.alphas_cumprod[t]
        alpha_t_next = self.model.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t_next) / (1 - alpha_t)) * pred_noise
        
        # Random noise
        noise = eta * torch.randn_like(x_t) * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t))
        
        # DDIM update
        x_t_next = torch.sqrt(alpha_t_next) * x_0_pred + dir_xt + noise
        
        return x_t_next