import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import logging
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.diffusion_unet import UNet
from src.models.ddpm import DDPM, EMA_DDPM
from src.data.dataset import create_dataloaders
from src.utils.helpers import save_checkpoint, load_checkpoint, save_image_grid, setup_device
from src.utils.config import Config


class DiffusionTrainer:
    """Trainer for diffusion models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = setup_device(config.device)
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 1000,  # Approximate steps
            eta_min=1e-6
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
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 2)
        
        # Setup tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        # Setup wandb if available
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="cat-generation-diffusion",
                config=config._config,
                name=f"diffusion_{config.image_size}"
            )
    
    def create_model(self) -> nn.Module:
        """Create the diffusion model."""
        # Create U-Net
        unet = UNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            model_channels=self.config.model_channels,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            dropout=self.config.dropout,
            channel_mult=self.config.channel_mult,
            num_heads=self.config.num_heads,
            use_scale_shift_norm=self.config.use_scale_shift_norm,
            image_size=self.config.image_size
        )
        
        # Create DDPM
        if self.config.get('use_ema', True):
            model = EMA_DDPM(
                unet=unet,
                num_timesteps=self.config.num_timesteps,
                beta_schedule=self.config.beta_schedule,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
                loss_type=self.config.get('loss_type', 'l2'),
                device=self.device,
                ema_decay=self.config.get('ema_decay', 0.9999)
            )
        else:
            model = DDPM(
                unet=unet,
                num_timesteps=self.config.num_timesteps,
                beta_schedule=self.config.beta_schedule,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
                loss_type=self.config.get('loss_type', 'l2'),
                device=self.device
            )
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            loss_dict = self.model.training_loss(images)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update EMA
            if hasattr(self.model, 'update_ema'):
                self.model.update_ema()
            
            # Logging
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
            if self.global_step % self.config.log_interval == 0:
                self.log_training_step(loss_dict)
            
            if self.global_step % self.config.sample_interval == 0:
                self.generate_samples()
            
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                loss_dict = self.model.training_loss(images)
                val_losses.append(loss_dict['loss'].item())
        
        return {'val_loss': np.mean(val_losses)}
    
    def generate_samples(self, num_samples: Optional[int] = None):
        """Generate and save samples."""
        if num_samples is None:
            num_samples = self.config.num_samples
        
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(
                batch_size=num_samples,
                channels=self.config.in_channels,
                height=self.config.image_size,
                width=self.config.image_size,
                return_all_timesteps=False
            )
        
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
        
        self.model.train()
    
    def log_training_step(self, loss_dict: Dict[str, torch.Tensor]):
        """Log training metrics."""
        # Log to tensorboard
        self.writer.add_scalar('loss/train', loss_dict['loss'].item(), self.global_step)
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'loss': loss_dict['loss'].item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': self.global_step
            })
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_step_{self.global_step}.pt'
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            step=self.global_step,
            checkpoint_path=str(checkpoint_path),
            scheduler=self.scheduler,
            additional_info={
                'best_val_loss': self.best_val_loss,
                'config': self.config._config
            }
        )
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.epoch,
                step=self.global_step,
                checkpoint_path=str(best_path),
                scheduler=self.scheduler,
                additional_info={
                    'best_val_loss': self.best_val_loss,
                    'config': self.config._config
                }
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch}: Train loss = {train_metrics['train_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Epoch {epoch}: Val loss = {val_metrics['val_loss']:.4f}")
            
            # Log epoch metrics
            self.writer.add_scalar('loss/train_epoch', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('loss/val_epoch', val_metrics['val_loss'], epoch)
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss_epoch': train_metrics['train_loss'],
                    'val_loss_epoch': val_metrics['val_loss']
                })
            
            # Save best model and check early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True)
                self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s)")
                
                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
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
    
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Create trainer
    trainer = DiffusionTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()