import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.stylegan2 import Generator, Discriminator, MappingNetwork
from src.data.dataset import create_dataloaders
from src.utils.helpers import save_checkpoint, load_checkpoint, save_image_grid, setup_device
from src.utils.config import Config
from src.utils.ada import AdaptiveAugment


class GANTrainer:
    """Trainer for StyleGAN2 with Adaptive Discriminator Augmentation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = setup_device(config.device)
        
        # Setup logging
        self.setup_logging()
        
        # Create models
        self.create_models()
        
        # Create optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.mapping.parameters()),
            lr=config.g_lr,
            betas=(0.0, 0.99),
            eps=1e-8
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(0.0, 0.99),
            eps=1e-8
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            pin_memory=config.pin_memory,
            max_train_samples=config.get('max_train_samples', None)
        )
        
        # Initialize tracking
        self.global_step = 0
        self.epoch = 0
        self.best_fid = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 2)
        
        # Setup tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        # Setup wandb if available
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="cat-generation-gan",
                config=config._config,
                name=f"stylegan2_{config.image_size}"
            )
        
        # Setup augmentation
        self.augment = AdaptiveAugment(config.ada_target, config.ada_length, config.device)
        self.augment_pipe = self.augment.get_augmentation_pipeline()
    
    def create_models(self):
        """Create Generator, Discriminator, and Mapping network."""
        # Generator first to get n_latent
        self.generator = Generator(
            size=self.config.image_size,
            style_dim=self.config.style_dim,
            n_mlp=self.config.n_mlp,
            channel_multiplier=self.config.channel_multiplier
        ).to(self.device)
        
        # Mapping network with correct num_ws
        self.mapping = MappingNetwork(
            z_dim=self.config.latent_dim,
            w_dim=self.config.style_dim,
            num_ws=self.generator.n_latent,
            num_layers=self.config.n_mlp
        ).to(self.device)
        
        # Discriminator
        self.discriminator = Discriminator(
            size=self.config.image_size,
            channel_multiplier=self.config.channel_multiplier
        ).to(self.device)
        
        # Log model info
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        m_params = sum(p.numel() for p in self.mapping.parameters())
        
        self.logger.info(f"Generator parameters: {g_params:,}")
        self.logger.info(f"Discriminator parameters: {d_params:,}")
        self.logger.info(f"Mapping parameters: {m_params:,}")
        self.logger.info(f"Total parameters: {(g_params + d_params + m_params):,}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # File handler
        log_file = Path(self.config.log_dir) / 'training.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def g_nonsaturating_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Generator loss (non-saturating)."""
        return F.softplus(-fake_pred).mean()
    
    def d_logistic_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Discriminator logistic loss."""
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        
        # Clamp to prevent extreme values
        real_loss = torch.clamp(real_loss, max=50.0)
        fake_loss = torch.clamp(fake_loss, max=50.0)
        
        return real_loss.mean() + fake_loss.mean()
    
    def r1_penalty(self, real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty."""
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    
    def path_length_regularization(
        self,
        fake_img: torch.Tensor,
        latents: torch.Tensor,
        mean_path_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Path length regularization."""
        noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad, = torch.autograd.grad(
            outputs=(fake_img * noise).sum(),
            inputs=latents,
            create_graph=True
        )
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        
        # Adaptive path length
        path_mean = mean_path_length + 0.01 * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        
        return path_penalty, path_mean.detach(), path_lengths
    
    def train_discriminator_step(self, real_img: torch.Tensor) -> Dict[str, float]:
        """Single discriminator training step."""
        batch_size = real_img.shape[0]
        real_img.requires_grad = True
        
        # Generate fake images
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        w = self.mapping(z)
        # Split w for each layer
        w_split = [w[:, i] for i in range(w.shape[1])]
        fake_img = self.generator(w_split, return_latents=False)
        
        # Augment if enabled
        if self.config.augment_p > 0:
            real_img_aug = self.augment_pipe(real_img)
            fake_img_aug = self.augment_pipe(fake_img.detach())
        else:
            real_img_aug = real_img
            fake_img_aug = fake_img.detach()
        
        # Discriminator predictions
        real_pred = self.discriminator(real_img_aug)
        fake_pred = self.discriminator(fake_img_aug)
        
        # Calculate loss
        d_loss = self.d_logistic_loss(real_pred, fake_pred)
        
        # Backward
        self.d_optimizer.zero_grad()
        d_loss.backward()
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
        
        # R1 regularization
        d_regularize = self.global_step % self.config.d_reg_interval == 0
        if d_regularize:
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = self.r1_penalty(real_pred, real_img)
            
            self.d_optimizer.zero_grad()
            (self.config.r1_gamma / 2 * r1_loss * self.config.d_reg_interval).backward()
        
        self.d_optimizer.step()
        
        # Update augmentation probability
        if self.config.augment_p > 0:
            self.augment.update(real_pred.sign().mean().item())
        
        return {
            'd_loss': d_loss.item(),
            'real_score': real_pred.mean().item(),
            'fake_score': fake_pred.mean().item(),
            'augment_p': self.augment.p.item() if self.config.augment_p > 0 else 0
        }
    
    def train_generator_step(self) -> Dict[str, float]:
        """Single generator training step."""
        batch_size = self.config.batch_size
        
        # Generate images
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        w = self.mapping(z)
        # Split w for each layer
        w_split = [w[:, i] for i in range(w.shape[1])]
        fake_img = self.generator(w_split, return_latents=False)
        
        # Augment if enabled
        if self.config.augment_p > 0:
            fake_img_aug = self.augment_pipe(fake_img)
        else:
            fake_img_aug = fake_img
        
        # Generator loss
        fake_pred = self.discriminator(fake_img_aug)
        g_loss = self.g_nonsaturating_loss(fake_pred)
        
        # Backward
        self.g_optimizer.zero_grad()
        g_loss.backward()
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(
            list(self.generator.parameters()) + list(self.mapping.parameters()), 
            max_norm=10.0
        )
        
        self.g_optimizer.step()
        
        # Path length regularization
        g_regularize = self.global_step % self.config.g_reg_interval == 0
        if g_regularize:
            z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
            w = self.mapping(z)
            # Detach and create new tensor for path length regularization
            w = w.detach().requires_grad_(True)
            # Split w for each layer
            w_split = [w[:, i] for i in range(w.shape[1])]
            fake_img = self.generator(w_split, return_latents=False)
            
            if not hasattr(self, 'mean_path_length'):
                self.mean_path_length = torch.zeros(1).to(self.device)
            
            path_loss, self.mean_path_length, path_lengths = self.path_length_regularization(
                fake_img, w, self.mean_path_length
            )
            
            self.g_optimizer.zero_grad()
            (path_loss * self.config.pl_weight * self.config.g_reg_interval).backward()
            self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'fake_score': fake_pred.mean().item()
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_d_losses = []
        epoch_g_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, real_img in enumerate(pbar):
            real_img = real_img.to(self.device)
            
            # Train discriminator
            d_metrics = self.train_discriminator_step(real_img)
            epoch_d_losses.append(d_metrics['d_loss'])
            
            # Train generator
            g_metrics = self.train_generator_step()
            epoch_g_losses.append(g_metrics['g_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'd_loss': d_metrics['d_loss'],
                'g_loss': g_metrics['g_loss'],
                'aug_p': d_metrics['augment_p']
            })
            
            # Logging
            if self.config.log_interval != 0:
                if self.global_step % self.config.log_interval == 0:
                    self.log_training_step(loss_dict)
            
            if self.config.sample_interval != 0:
                if self.global_step % self.config.sample_interval == 0:
                    self.generate_samples()
            
            if self.config.checkpoint_interval != 0:
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()
            
            self.global_step += 1
        
        return {
            'train_d_loss': np.mean(epoch_d_losses),
            'train_g_loss': np.mean(epoch_g_losses)
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate using FID score."""
        from src.utils.metrics import GenerativeMetrics
        
        self.generator.eval()
        metrics_calculator = GenerativeMetrics(self.device)
        
        # Generate samples for FID
        n_samples = min(2048, len(self.val_loader.dataset))
        batch_size = self.config.batch_size
        
        fake_images = []
        real_images = []
        
        with torch.no_grad():
            # Generate fake images
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                z = torch.randn(current_batch_size, self.config.latent_dim).to(self.device)
                w = self.mapping(z)
                # Split w for each layer
                w_split = [w[:, i] for i in range(w.shape[1])]
                fake_img = self.generator(w_split, return_latents=False)
                fake_images.append(fake_img.cpu())
            
            # Collect real images
            for i, real_img in enumerate(self.val_loader):
                real_images.append(real_img)
                if len(real_images) * batch_size >= n_samples:
                    break
        
        fake_images = torch.cat(fake_images, dim=0)[:n_samples]
        real_images = torch.cat(real_images, dim=0)[:n_samples]
        
        # Calculate FID
        fid_score = metrics_calculator.compute_fid(real_images, fake_images, batch_size)
        
        return {'fid': fid_score}
    
    def generate_samples(self, num_samples: Optional[int] = None):
        """Generate and save samples."""
        if num_samples is None:
            num_samples = self.config.num_samples
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
            w = self.mapping(z)
            # Split w for each layer
            w_split = [w[:, i] for i in range(w.shape[1])]
            samples = self.generator(w_split, return_latents=False)
        
        # Save samples
        save_path = Path(self.config.sample_dir) / f'samples_step_{self.global_step}.png'
        save_image_grid(samples, save_path, nrow=int(np.sqrt(num_samples)))
        
        # Log to tensorboard
        self.writer.add_images('samples', samples, self.global_step)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'samples': wandb.Image(save_path),
                'step': self.global_step
            })
        
        self.generator.train()
    
    def log_training_step(self, d_metrics: Dict[str, float], g_metrics: Dict[str, float]):
        """Log training metrics."""
        # Log to tensorboard
        self.writer.add_scalar('loss/d_loss', d_metrics['d_loss'], self.global_step)
        self.writer.add_scalar('loss/g_loss', g_metrics['g_loss'], self.global_step)
        self.writer.add_scalar('scores/real_score', d_metrics['real_score'], self.global_step)
        self.writer.add_scalar('scores/fake_score', d_metrics['fake_score'], self.global_step)
        self.writer.add_scalar('augment_p', d_metrics['augment_p'], self.global_step)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'd_loss': d_metrics['d_loss'],
                'g_loss': g_metrics['g_loss'],
                'real_score': d_metrics['real_score'],
                'fake_score': d_metrics['fake_score'],
                'augment_p': d_metrics['augment_p'],
                'step': self.global_step
            })
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_step_{self.global_step}.pt'
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'mapping_state_dict': self.mapping.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_fid': self.best_fid,
            'config': self.config._config
        }
        
        if hasattr(self, 'mean_path_length'):
            checkpoint['mean_path_length'] = self.mean_path_length
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.mapping.load_state_dict(checkpoint['mapping_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        self.best_fid = checkpoint.get('best_fid', float('inf'))
        
        if 'mean_path_length' in checkpoint:
            self.mean_path_length = checkpoint['mean_path_length']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch}: D loss = {train_metrics['train_d_loss']:.4f}, "
                           f"G loss = {train_metrics['train_g_loss']:.4f}")
            
            # Validate with FID
            if epoch % 5 == 0:  # Validate every 5 epochs (FID is expensive)
                val_metrics = self.validate()
                self.logger.info(f"Epoch {epoch}: FID = {val_metrics['fid']:.2f}")
                
                # Log epoch metrics
                self.writer.add_scalar('metrics/fid', val_metrics['fid'], epoch)
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'fid': val_metrics['fid']
                    })
                
                # Early stopping based on FID
                if val_metrics['fid'] < self.best_fid:
                    self.best_fid = val_metrics['fid']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best FID: {self.best_fid:.2f}")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    self.logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s)")
                    
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered! Best FID: {self.best_fid:.2f}")
                        break
            
            # Generate samples at end of epoch
            self.generate_samples()
        
        self.logger.info("Training completed!")
        self.writer.close()
        if self.use_wandb:
            wandb.finish()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train StyleGAN2')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
