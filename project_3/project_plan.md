# Project III - Image Generation with Generative Models

**Team:** Szymon Smagowski, Jerzy Kraszewski

## Dataset

### Primary Dataset: Cat Dataset
[Cat Dataset](https://www.kaggle.com/datasets/borhanitrash/cat-dataset)
- Contains various cat images with different resolutions
- Will be preprocessed to ensure consistent image dimensions

### Secondary Dataset: Dogs vs Cats
[Dogs vs Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/)
- Used for the additional task of comparing generation results
- Will be processed similarly to maintain consistency

## Chosen Approach: Diffusion Models

We have selected Diffusion Models as our primary approach for the following reasons:
1. Recent state-of-the-art results in image generation
2. More stable training compared to GANs
3. Better mode coverage and less prone to mode collapse
4. Good quality-to-computation trade-off

### Specific Implementation Details

#### 1. Data Preprocessing Pipeline
- Resize all images to 256x256 pixels (balance between quality and computational requirements)
- Normalize pixel values to [-1, 1] range
- Implement data augmentation:
  - Random horizontal flips
  - Small random rotations (±10 degrees)
  - Color jittering (brightness, contrast)
- Create train/validation/test splits (80/10/10)

#### 2. Model Architecture: DDPM (Denoising Diffusion Probabilistic Model)
- U-Net backbone with the following specifications:
  - 4 downsampling blocks
  - 4 upsampling blocks
  - Attention layers at 16x16, 8x8 resolutions
  - Time embedding using sinusoidal positional encoding
  - Residual connections throughout
- Noise schedule: Linear beta schedule with 1000 timesteps
- Training objective: Simplified objective (predicting noise)

#### 3. Model Inspection and Visualization
- Implement feature map visualization for each layer:
  - Downsampling path feature maps
  - Upsampling path feature maps
  - Attention maps
- Create visualization pipeline for:
  - Intermediate denoising steps
  - Layer-wise activation patterns
  - Attention heatmaps
- Analyze feature representations at different noise levels
- Visualize skip connection contributions
- Monitor gradient flow through the network

#### 4. Hyperparameter Investigation
We will systematically investigate the following parameters:
1. Number of diffusion steps (500, 1000, 2000)
2. Noise schedule type (linear, cosine)
3. Model capacity (number of channels in base layer: 64, 128, 256)
4. Attention mechanism variants (self-attention, cross-attention)
5. Training duration impact

#### 5. Evaluation Metrics
- Fréchet Inception Distance (FID)
- Inception Score
- Qualitative assessment through:
  - Visual inspection of generated samples
  - Latent space interpolation analysis
  - Mode coverage analysis
  - Layer-wise feature analysis

#### 6. Latent Space Interpolation
- Select two distinct cat images
- Generate their corresponding latent representations
- Create 8 intermediate points using linear interpolation
- Generate and analyze the resulting images
- Document the smoothness of transitions and feature preservation

#### 7. Additional Task: Dogs vs Cats
- Train the same model architecture on the combined dataset
- Compare results with cat-only model using:
  - FID scores
  - Visual quality assessment
  - Feature preservation analysis
  - Mode separation analysis

## Implementation Timeline

1. Data Preparation and Infrastructure
   - Set up data preprocessing pipeline
   - Implement data loading and augmentation
   - Create training infrastructure

2. Model Implementation
   - Implement DDPM architecture
   - Set up training loop
   - Implement evaluation metrics
   - Create visualization tools

3. Initial Training and Hyperparameter Tuning
   - Train baseline model
   - Conduct hyperparameter experiments
   - Document results
   - Analyze model internals

4. Analysis and Additional Task
   - Perform latent space interpolation
   - Train on combined dataset
   - Prepare final analysis
   - Generate comprehensive visualizations

## Notes on Computational Resources

- Implement gradient checkpointing to reduce memory usage
- Use mixed precision training
- Consider model parallelism if needed
- Implement checkpointing to resume training
- Use data prefetching for efficient training
- Monitor GPU memory usage and adjust batch size accordingly

