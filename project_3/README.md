# Project 3: Cat Image Generation with Generative Models

This project implements two state-of-the-art generative models for cat image generation:
1. **Diffusion Models (DDPM)** - Denoising Diffusion Probabilistic Models
2. **GANs (StyleGAN2-ADA)** - StyleGAN2 with Adaptive Discriminator Augmentation

## Project Structure

```
project_3/
├── src/
│   ├── models/           # Model implementations
│   │   ├── diffusion_unet.py   # U-Net with attention for diffusion models
│   │   ├── ddpm.py             # DDPM core implementation with noise scheduling
│   │   └── stylegan2.py        # Complete StyleGAN2 with mapping network
│   ├── data/             # Data loading and preprocessing
│   │   ├── preprocessing.py    # Image preprocessing and dataset creation
│   │   └── dataset.py          # PyTorch datasets and data loaders
│   ├── trainers/         # Training scripts
│   │   ├── train_diffusion.py  # Complete diffusion model training loop
│   │   └── train_gan.py        # StyleGAN2 training with ADA augmentation
│   └── utils/            # Utilities and helpers
│       ├── config.py           # YAML configuration management
│       ├── helpers.py          # Device setup, checkpointing, visualization
│       ├── metrics.py          # FID, IS, precision/recall evaluation
│       └── ada.py              # Adaptive discriminator augmentation
├── configs/              # YAML configuration files
│   ├── diffusion_128.yaml      # Diffusion model config (128x128)
│   ├── diffusion_256.yaml      # Diffusion model config (256x256) 
│   ├── diffusion_fast.yaml     # Fast diffusion config for testing
│   ├── gan_128.yaml            # GAN config (128x128)
│   ├── gan_256.yaml            # GAN config (256x256)
│   └── gan_fast_test.yaml      # Fast GAN config for testing (64x64)
├── data/                 # Dataset storage
│   ├── cat-dataset/            # Original cat dataset (15k images)
│   ├── processed_cats_128/     # Preprocessed 128x128 images
│   └── processed_cats_256/     # Preprocessed 256x256 images
├── outputs/              # Training outputs and results
│   ├── checkpoints/            # Model checkpoints by experiment
│   ├── generated_samples/      # Generated image samples
│   ├── logs/                   # TensorBoard logs and training logs
│   └── evaluation/             # Evaluation results and metrics
├── preprocess_data.py          # Data preprocessing script
├── test_setup.py              # Setup verification script
└── requirements.txt           # Python dependencies
```

## Current Approach

### Diffusion Models (DDPM)
- **Architecture**: U-Net with multi-head attention, ResNet blocks, time embeddings
- **Noise Schedule**: Linear and cosine schedules with 1000 timesteps
- **Training**: MSE loss between predicted and actual noise
- **Sampling**: Iterative denoising process from random noise

### StyleGAN2 with ADA
- **Generator**: Style-based synthesis with modulated convolutions
- **Discriminator**: Progressive downsampling with minibatch std deviation
- **Training**: Adversarial loss with R1 regularization and path length regularization
- **Augmentation**: Adaptive discriminator augmentation to prevent overfitting

## File Descriptions

### Core Models
- **`src/models/diffusion_unet.py`**: U-Net architecture with attention layers and time conditioning
- **`src/models/ddpm.py`**: DDPM implementation with forward/reverse diffusion process
- **`src/models/stylegan2.py`**: Complete StyleGAN2 with generator, discriminator, and mapping network

### Training Infrastructure  
- **`src/trainers/train_diffusion.py`**: Full training loop with EMA, validation, checkpointing
- **`src/trainers/train_gan.py`**: GAN training with discriminator/generator alternation and regularization
- **`src/data/preprocessing.py`**: Image resizing, normalization, dataset splitting
- **`src/data/dataset.py`**: PyTorch datasets with augmentation and efficient loading

### Utilities
- **`src/utils/config.py`**: YAML configuration management and default configs
- **`src/utils/helpers.py`**: Device setup, checkpointing, image grid visualization
- **`src/utils/metrics.py`**: FID calculation with InceptionV3, IS computation
- **`src/utils/ada.py`**: Adaptive discriminator augmentation implementation

## Setup

1. **Activate virtual environment:**
```bash
source ../deep-learning/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Preprocess data:**
```bash
python preprocess_data.py
```

4. **Verify setup:**
```bash
python test_setup.py
```

## Training

### Diffusion Model
```bash
# Fast testing (smaller model, fewer epochs)
python src/trainers/train_diffusion.py --config configs/diffusion_fast.yaml

# Full 128x128 training 
python src/trainers/train_diffusion.py --config configs/diffusion_128.yaml

# Full 256x256 training
python src/trainers/train_diffusion.py --config configs/diffusion_256.yaml

# Resume training
python src/trainers/train_diffusion.py --config configs/diffusion_128.yaml --resume outputs/checkpoints/diffusion/checkpoint_step_X.pt
```

### GAN Model (StyleGAN2-ADA)
```bash
# Fast testing (64x64, 10 epochs)
python src/trainers/train_gan.py --config configs/gan_fast_test.yaml

# Full 128x128 training
python src/trainers/train_gan.py --config configs/gan_128.yaml

# Full 256x256 training  
python src/trainers/train_gan.py --config configs/gan_256.yaml

# Resume training
python src/trainers/train_gan.py --config configs/gan_128.yaml --resume outputs/checkpoints/gan/checkpoint_step_X.pt
```

## Monitoring

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## Current Status & Issues

### ✅ **Working Components**
- Complete model implementations (DDPM, StyleGAN2)
- Data preprocessing pipeline (15K cat images)
- Training infrastructure with checkpointing
- Evaluation metrics (FID, IS)
- Device detection (CUDA/MPS/CPU)
- Configuration management

### ❌ **Current Problems**

#### Fast Test Results (Analysis Completed)
**GAN Issues:**
- **Mode collapse**: Final samples are completely black
- **Training instability**: NaN losses, discriminator dominance  
- **No cat features**: Only abstract colored patches

**Diffusion Issues:**
- **Pure noise output**: No coherent structures generated
- **No learning**: No improvement across training steps
- **Misleading metrics**: Good loss reduction doesn't reflect quality

#### Root Causes
1. **Insufficient model capacity**: "Fast" configs too small for complex images
2. **Too short training**: 10 epochs insufficient for generative models
3. **Hyperparameter issues**: Learning rates, regularization weights
4. **Architecture limitations**: Reduced layers/channels compromise learning

### 🔧 **What Needs to Be Done**

#### Immediate Fixes
1. **Use full model configurations** (not fast/test versions)
2. **Increase training duration** to 100-1000+ epochs  
3. **Scale up model architecture** (more layers, channels)
4. **Hyperparameter tuning**:
   - Adjust learning rates
   - Balance discriminator/generator training
   - Tune regularization weights

#### Architecture Improvements
1. **Progressive training** for GANs
2. **Better noise scheduling** for diffusion
3. **Improved attention mechanisms**
4. **Transfer learning** from pretrained models

#### Recommended Training Strategy
1. Start with diffusion model (more stable)
2. Use 128x128 resolution initially
3. Train for 200+ epochs minimum
4. Monitor FID scores closely
5. Scale to 256x256 once stable results achieved

## Evaluation Metrics

- **FID (Fréchet Inception Distance)** - Lower is better (real images ~50-100)
- **IS (Inception Score)** - Higher is better (real images ~8-12)  
- **Precision & Recall** - Mode coverage analysis
- **Visual Inspection** - Most important for image quality

## Generated Samples

Generated samples are saved in `outputs/generated_samples/` during training.
Current fast test samples show poor quality - full training needed.

## Key Features

### ✅ **Implemented Features**
1. **Diffusion Model (DDPM):**
   - U-Net with multi-head attention and ResNet blocks
   - Linear and cosine noise schedules (1000 timesteps)
   - EMA (Exponential Moving Average) for stable generation
   - Time conditioning and sinusoidal embeddings
   - Mixed precision training support

2. **GAN Model (StyleGAN2-ADA):**
   - Style-based generator with modulated convolutions
   - Mapping network for latent space transformation
   - Adaptive discriminator augmentation (ADA)
   - R1 gradient penalty and path length regularization
   - Minibatch standard deviation for diversity

3. **Data Pipeline:**
   - Automatic preprocessing (resize, normalize, split)
   - Efficient PyTorch data loading with augmentation
   - Support for multiple resolutions (128x128, 256x256)
   - 15K cat images preprocessed and ready

4. **Training Infrastructure:**
   - Automatic checkpointing and resume capability
   - TensorBoard logging and monitoring
   - FID/IS evaluation during training
   - Early stopping based on validation metrics
   - Device-agnostic training (CUDA/MPS/CPU)

5. **Configuration System:**
   - YAML-based hyperparameter management
   - Separate configs for different resolutions
   - Fast test configs for debugging

## Configuration Files

| Config File | Purpose | Resolution | Epochs | Notes |
|-------------|---------|------------|--------|-------|
| `diffusion_fast.yaml` | Quick testing | 128x128 | 10 | Small model, fast iteration |
| `diffusion_128.yaml` | Full training | 128x128 | 200 | Production quality |
| `diffusion_256.yaml` | High res | 256x256 | 100 | Requires more GPU memory |
| `gan_fast_test.yaml` | Quick testing | 64x64 | 10 | **Currently broken** |
| `gan_128.yaml` | Full training | 128x128 | 200 | Not tested yet |
| `gan_256.yaml` | High res | 256x256 | 100 | Not tested yet |

### Key Configuration Parameters
- **`image_size`**: Resolution of generated images (64/128/256)
- **`batch_size`**: Training batch size (adjust based on GPU memory)
- **`num_epochs`**: Number of training epochs
- **`learning_rate`**: Learning rate for optimizers
- **`num_timesteps`**: Diffusion steps (1000 for DDPM)
- **`model_channels`**: Base channel count (affects model size)

## Hardware Requirements

- **Fast Testing:** 4-8GB GPU memory
- **128x128 Training:** 8-12GB GPU memory  
- **256x256 Training:** 16GB+ GPU memory
- **CPU Fallback:** Available but very slow

**Supported Devices:**
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS) 
- CPU (fallback)

## Known Issues & Solutions

### Current Major Issues
1. **Fast configs produce poor results** - Use full configs instead
2. **Short training insufficient** - Need 100+ epochs minimum
3. **GAN training instability** - Needs hyperparameter tuning

### Recommended Approach
1. **Start with diffusion models** (more stable than GANs)
2. **Use 128x128 resolution** initially
3. **Train for 200+ epochs** minimum
4. **Monitor FID scores** during training
5. **Switch to full configs** (not fast/test versions)

## Next Steps - Priority Order

### High Priority
1. **Fix GAN hyperparameters** in full configs
2. **Run full diffusion training** (128x128, 200 epochs)
3. **Compare model quality** using FID/IS metrics
4. **Optimize training speed** and memory usage

### Medium Priority  
1. **Implement progressive GAN training**
2. **Add more evaluation metrics**
3. **Create sample interpolation tools**
4. **Scale to 256x256 resolution**

### Low Priority
1. **Add Dogs vs Cats comparison task**
2. **Implement latent space analysis**
3. **Create web interface for generation**
4. **Experiment with different architectures**