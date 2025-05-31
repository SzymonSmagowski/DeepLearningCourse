import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, List
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
import warnings

# Try to import clean_fid, but make it optional
try:
    from clean_fid import fid
    HAS_CLEAN_FID = True
except ImportError:
    HAS_CLEAN_FID = False
    warnings.warn("clean_fid not available. Using built-in FID calculation instead.")


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network for feature extraction."""
    
    def __init__(self, output_blocks: List[int] = [3], resize_input: bool = True, normalize_input: bool = True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True)
        
        # Build feature extraction blocks
        self.blocks = nn.ModuleList()
        
        block_idx = 0
        for module in inception.children():
            if block_idx > self.last_needed_block:
                break
                
            if isinstance(module, nn.Sequential):
                for sub_module in module:
                    self.blocks.append(sub_module)
                    block_idx += 1
                    if block_idx > self.last_needed_block:
                        break
            else:
                self.blocks.append(module)
                block_idx += 1
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from InceptionV3."""
        outputs = []
        
        # Resize if necessary
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize if necessary
        if self.normalize_input:
            x = 2 * x - 1  # Scale from [0, 1] to [-1, 1]
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
            
            if idx == self.last_needed_block:
                break
        
        return outputs


def calculate_activation_statistics(
    images: torch.Tensor,
    model: nn.Module,
    batch_size: int = 64,
    dims: int = 2048,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of activations."""
    model.eval()
    
    # Get activations
    activations = []
    
    n_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Extracting features"):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)
        
        with torch.no_grad():
            features = model(batch)[0]
        
        # Pool features
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        features = features.squeeze(-1).squeeze(-1).cpu().numpy()
        activations.append(features)
    
    activations = np.concatenate(activations, axis=0)
    
    # Calculate statistics
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """Calculate Fréchet distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn("Product of covariance matrices is singular. Adding epsilon to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 64,
    device: str = 'cpu',
    dims: int = 2048
) -> float:
    """Calculate FID score between real and fake images."""
    # Initialize InceptionV3
    inception = InceptionV3([3], resize_input=True, normalize_input=True)
    inception = inception.to(device)
    inception.eval()
    
    # Calculate statistics for real images
    m1, s1 = calculate_activation_statistics(real_images, inception, batch_size, dims, device)
    
    # Calculate statistics for fake images
    m2, s2 = calculate_activation_statistics(fake_images, inception, batch_size, dims, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


def calculate_fid_from_paths(
    real_path: str,
    fake_path: str,
    batch_size: int = 64,
    device: str = 'cpu',
    num_workers: int = 4
) -> float:
    """Calculate FID using clean-fid library if available."""
    if HAS_CLEAN_FID:
        score = fid.compute_fid(real_path, fake_path, device=device, num_workers=num_workers)
        return score
    else:
        # Fall back to loading images and using built-in FID
        warnings.warn("Using built-in FID calculation. This may be slower than clean_fid.")
        
        # Load images from directories
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        real_dataset = ImageFolder(real_path, transform=transform)
        fake_dataset = ImageFolder(fake_path, transform=transform)
        
        real_loader = DataLoader(real_dataset, batch_size=batch_size, num_workers=num_workers)
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, num_workers=num_workers)
        
        # Collect all images
        real_images = []
        fake_images = []
        
        for imgs, _ in real_loader:
            real_images.append(imgs)
        for imgs, _ in fake_loader:
            fake_images.append(imgs)
            
        real_images = torch.cat(real_images, dim=0)
        fake_images = torch.cat(fake_images, dim=0)
        
        return calculate_fid(real_images, fake_images, batch_size, device)


def calculate_inception_score(
    images: torch.Tensor,
    batch_size: int = 32,
    splits: int = 10,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """Calculate Inception Score."""
    # Load InceptionV3
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception = inception.to(device)
    inception.eval()
    
    # Get predictions
    preds = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Computing IS"):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)
        
        # Resize to 299x299 for InceptionV3
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            pred = inception(batch)
            pred = F.softmax(pred, dim=1).cpu().numpy()
            preds.append(pred)
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute score
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)


class GenerativeMetrics:
    """Wrapper class for all generative model metrics."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._inception_model = None
    
    @property
    def inception_model(self):
        if self._inception_model is None:
            self._inception_model = InceptionV3([3]).to(self.device)
            self._inception_model.eval()
        return self._inception_model
    
    def compute_fid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 64
    ) -> float:
        """Compute FID score."""
        return calculate_fid(real_images, fake_images, batch_size, self.device)
    
    def compute_fid_from_paths(
        self,
        real_path: str,
        fake_path: str,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> float:
        """Compute FID from directory paths."""
        return calculate_fid_from_paths(real_path, fake_path, batch_size, self.device, num_workers)
    
    def compute_inception_score(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
        splits: int = 10
    ) -> Tuple[float, float]:
        """Compute Inception Score."""
        return calculate_inception_score(images, batch_size, splits, self.device)
    
    def compute_all_metrics(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 64
    ) -> dict:
        """Compute all metrics."""
        # FID
        fid_score = self.compute_fid(real_images, fake_images, batch_size)
        
        # IS for fake images
        is_mean, is_std = self.compute_inception_score(fake_images, batch_size)
        
        return {
            'fid': fid_score,
            'is_mean': is_mean,
            'is_std': is_std
        }


def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3
) -> Tuple[float, float]:
    """Compute precision and recall metrics."""
    from sklearn.neighbors import NearestNeighbors
    
    # Fit KNN on real features
    knn_real = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn_real.fit(real_features)
    
    # Fit KNN on fake features
    knn_fake = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn_fake.fit(fake_features)
    
    # Compute precision: How many fake samples have at least one real neighbor
    distances, _ = knn_real.kneighbors(fake_features)
    precision = np.mean(np.any(distances[:, 1:] < np.median(distances[:, 1:]), axis=1))
    
    # Compute recall: How many real samples have at least one fake neighbor
    distances, _ = knn_fake.kneighbors(real_features)
    recall = np.mean(np.any(distances[:, 1:] < np.median(distances[:, 1:]), axis=1))
    
    return precision, recall