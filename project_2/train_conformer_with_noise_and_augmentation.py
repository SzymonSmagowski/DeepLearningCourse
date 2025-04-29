import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import librosa
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import from helper modules
from helpers.dataset_builder import DatasetBuilder
from helpers.utils import load_from_path, load_audio, normalize_audio_duration, extract_mfcc
import helpers.models as models


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Set up device (MPS for Mac, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def augment_audio(audio, sr, augmentation_methods=['pitch', 'speed', 'noise', 'stretch']):
    """
    Apply random audio augmentation techniques.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        augmentation_methods (list): List of augmentation methods to choose from
        
    Returns:
        np.ndarray: Augmented audio
    """
    # Choose a random augmentation method
    method = random.choice(augmentation_methods)
    
    if method == 'pitch':
        # Pitch shift by up to ±4 semitones
        n_steps = random.uniform(-4.0, 4.0)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    elif method == 'speed':
        # Speed up or slow down by up to ±15%
        speed_factor = random.uniform(0.85, 1.15)
        # Use librosa.effects.time_stretch for speed change
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    elif method == 'noise':
        # Add random Gaussian noise with SNR between 10-20dB
        noise_level = random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    elif method == 'stretch':
        # Time stretch without changing pitch by up to ±10%
        stretch_factor = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    return audio  # Return original if no method matches


def load_background_noise(noise_path):
    """Load all background noise files from the directory"""
    noise_files = []
    noise_path = Path(noise_path)
    
    if not noise_path.exists() or not noise_path.is_dir():
        raise FileNotFoundError(f"Background noise directory not found at {noise_path}")
    
    for file in noise_path.glob("*.wav"):
        try:
            audio, sr = load_audio(file)
            noise_files.append((audio, sr, file.name))
            print(f"Loaded noise file: {file.name}, duration: {len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not noise_files:
        raise ValueError("No valid noise files found in the directory")
    
    return noise_files


def add_background_noise(audio, sr, noise_audio, noise_sr, volume_factor):
    """Add background noise to audio at specified volume"""
    # Ensure noise is at same sample rate
    if noise_sr != sr:
        noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=sr)
    
    # If noise is shorter than audio, repeat it
    if len(noise_audio) < len(audio):
        repetitions = math.ceil(len(audio) / len(noise_audio))
        noise_audio = np.tile(noise_audio, repetitions)
    
    # If noise is longer than audio, take a random segment
    if len(noise_audio) > len(audio):
        start = random.randint(0, len(noise_audio) - len(audio))
        noise_audio = noise_audio[start:start + len(audio)]
    
    # Normalize both audio and noise
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
    noise_norm = noise_audio / (np.max(np.abs(noise_audio)) + 1e-8)
    
    # Mix audio with noise at specified volume
    mixed_audio = audio_norm + volume_factor * noise_norm
    
    # Normalize mixed audio to prevent clipping
    mixed_audio = mixed_audio / (np.max(np.abs(mixed_audio)) + 1e-8)
    
    return mixed_audio


def preprocess_dataset_with_augmentation_and_noise(dataset, noise_files, volume_factor, apply_augmentation, name=""):
    """
    Reads every WAV → centers/normalises length → optionally adds augmentation → 
    adds noise → MFCC → returns (X, y) tensors ready for PyTorch.
    """
    features, labels = [], []
    
    for sample in tqdm(dataset, desc=f"Pre-processing {name} with noise {volume_factor:.2f}" + (" and augmentation" if apply_augmentation else "")):
        # Load original audio
        audio, sr = load_audio(sample["file_path"])
        audio = normalize_audio_duration(audio, sr)
        
        # Apply augmentation if requested
        if apply_augmentation:
            audio = augment_audio(audio, sr)
        
        # Add noise if requested (volume_factor > 0)
        if volume_factor > 0 and noise_files:
            # Select a random noise file
            noise_audio, noise_sr, noise_name = random.choice(noise_files)
            
            # Add noise
            audio = add_background_noise(audio, sr, noise_audio, noise_sr, volume_factor)
        
        # Extract MFCC
        mfcc = extract_mfcc(audio, sr)
        
        features.append(mfcc)
        labels.append(sample["label_id"])
    
    X = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def create_conformer_model(config, num_classes, device):
    """Create Conformer model based on configuration"""
    model = models.ConformerClassifier(
        n_classes=num_classes,
        n_mfcc=config.get('n_mfcc', 40),
        d_model=config.get('d_model', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        kernel_size=config.get('kernel_size', 15),
        expansion_factor=config.get('expansion_factor', 4),
        dropout=config.get('dropout', 0.1),
        pool=config.get('pool', 'mean')
    ).to(device)
    
    return model


def train_epoch(model, device, train_loader, criterion, optimizer):
    """Train model for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return total_loss / total, correct / total


def validate(model, device, val_loader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return total_loss / total, correct / total


def test_model(model, device, test_loader, id2label):
    """Evaluate model on test set and generate confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    model.eval()
    all_preds, all_targets = [], []
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Statistics
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Collect all predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(id2label))))
    
    return accuracy, cm


def save_results(config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir, noise_level, apply_augmentation):
    """Save training results to JSON file"""
    results = {
        'config': config,
        'history': history,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'confusion_matrix': conf_matrix.tolist(),
        'id2label': id2label,
        'noise_level': float(noise_level),
        'apply_augmentation': apply_augmentation
    }
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_str = "_aug" if apply_augmentation else ""
    
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{timestamp}_conformer_noise_{noise_level:.2f}{aug_str}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def get_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Creates a cosine learning rate scheduler with optional warmup"""
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_with_augmentation_and_noise(config, train_dataset, val_dataset, test_dataset, noise_files, volume_factor, apply_augmentation, batch_size, device, id2label, results_dir, early_stopping_patience=5):
    """Train and evaluate model with a specific noise level and optional augmentation"""
    aug_str = "augmentation and " if apply_augmentation else ""
    print(f"\n===== Training with {aug_str}noise level: {volume_factor:.2f} =====")
    
    # Preprocess datasets with added noise and optional augmentation
    X_train, y_train = preprocess_dataset_with_augmentation_and_noise(
        train_dataset, noise_files, volume_factor, apply_augmentation, "train"
    )
    X_val, y_val = preprocess_dataset_with_augmentation_and_noise(
        val_dataset, noise_files, volume_factor, False, "valid"  # No augmentation for validation
    )
    X_test, y_test = preprocess_dataset_with_augmentation_and_noise(
        test_dataset, noise_files, volume_factor, False, "test"  # No augmentation for test
    )
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Create data loaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Create Conformer model
    model = create_conformer_model(config, len(id2label), device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if config.get('optimizer', 'adamw') == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 0.0005), 
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif config.get('optimizer', 'adamw') == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 0.0005), 
            weight_decay=config.get('weight_decay', 0.01)
        )
    else:  # default to adamw if not specified
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 0.0005), 
            weight_decay=config.get('weight_decay', 0.01)
        )
    
    # Define learning rate scheduler
    if config.get('scheduler', 'cosine') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('scheduler_step_size', 10), 
            gamma=config.get('scheduler_gamma', 0.5)
        )
    elif config.get('scheduler', 'cosine') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.get('scheduler_gamma', 0.5), 
            patience=2, 
            verbose=True
        )
    elif config.get('scheduler', 'cosine') == 'cosine':
        # Calculate number of training steps
        epochs = config.get('epochs', 50)
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        scheduler = get_cosine_scheduler(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(1, config.get('epochs', 50) + 1):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        
        # Update learning rate
        if scheduler is not None:
            if config.get('scheduler', 'cosine') == 'step' or config.get('scheduler', 'cosine') == 'cosine':
                scheduler.step()
            elif config.get('scheduler', 'cosine') == 'plateau':
                scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"[{epoch:02d}/{config.get('epochs', 50)}] "
              f"train {train_loss:.4f}/{train_acc:.3f} – "
              f"val {val_loss:.4f}/{val_acc:.3f}")
        
        # Check if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"⏹️  Early stopping after {epoch} epochs (no val-improve for {early_stopping_patience} epochs).")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_acc, conf_matrix = test_model(model, device, test_loader, id2label)
    print(f"Test accuracy: {test_acc:.3%}")
    
    # Save results
    results_file = save_results(
        config, history, best_val_acc, test_acc, 
        conf_matrix, id2label, results_dir, 
        volume_factor, apply_augmentation
    )
    
    return {
        'volume_factor': volume_factor,
        'apply_augmentation': apply_augmentation,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'results_file': results_file
    }


def train_conformer_with_noise_and_augmentation():
    parser = argparse.ArgumentParser(description='Train Conformer with background noise and augmentation')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/speech_commands_v0.01',
                      help='Path to the Speech Commands dataset')
    parser.add_argument('--subset', action='store_true',
                      help='Use only yes/no subset instead of full dataset')
    parser.add_argument('--unknown_ratio', type=float, default=0.2,
                      help='Ratio of unknown samples to keep')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--results_dir', type=str, default='results/conformer_noise_aug',
                      help='Directory to save results')
    
    # Noise and augmentation parameters
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.8],
                      help='Volume factors for noise (can specify multiple, e.g., 0.0 0.1 0.3 0.5)')
    parser.add_argument('--run_augmentation', action='store_true',
                      help='Run experiments with data augmentation')
    
    args = parser.parse_args()
    
    # Set up device and seed
    device = get_device()
    set_seed(args.seed)
    
    # Define Conformer model configuration
    conformer_config = {
        "n_mfcc": 40,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "kernel_size": 15,
        "expansion_factor": 4,
        "dropout": 0.1,
        "pool": "mean",
        "optimizer": "adamw",
        "learning_rate": 0.0005,
        "weight_decay": 0.01,
        "epochs": 50,
        "scheduler": "cosine",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.5
    }
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    train_, valid_, test_ = load_from_path(Path(args.data_path))
    
    # Prepare dataset
    builder = DatasetBuilder(train_, valid_, test_, args.seed)
    
    if args.subset:
        # Use only yes/no subset
        train, valid, test = builder.keep_labels(['yes', 'no'], make_rest_unknown=False).build()
    else:
        # Use full dataset with standard command labels
        standard_commands = ['yes', 'no', 'up', 'down', 'left', 'right', 
                            'on', 'off', 'stop', 'go']
        train, valid, test = builder \
            .keep_labels(standard_commands, make_rest_unknown=True) \
            .unknown_ratio(args.unknown_ratio) \
            .build()
    
    # Load background noise files
    noise_path = Path(args.data_path) / "_background_noise_"
    try:
        noise_files = load_background_noise(noise_path)
        print(f"Loaded {len(noise_files)} background noise files")
    except Exception as e:
        print(f"Warning: Could not load background noise files: {e}")
        print("Proceeding without background noise")
        noise_files = []
    
    # Define test combinations: with/without augmentation for each noise level
    test_configs = []
    
    # Add configurations without augmentation
    for noise_level in args.noise_levels:
        test_configs.append({
            'noise_level': noise_level,
            'apply_augmentation': False
        })
    
    # Add configurations with augmentation if requested
    if args.run_augmentation:
        for noise_level in args.noise_levels:
            test_configs.append({
                'noise_level': noise_level,
                'apply_augmentation': True
            })
    
    # Train models for each configuration
    all_results = []
    for config in test_configs:
        result = train_with_augmentation_and_noise(
            conformer_config, train, valid, test, 
            noise_files, config['noise_level'], config['apply_augmentation'], 
            args.batch_size, device, builder.id2label, args.results_dir
        )
        all_results.append(result)
    
    # Save summary of all results
    summary = {
        'model_config': conformer_config,
        'results': all_results
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{args.results_dir}/summary_{timestamp}_conformer_results.json"
    
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n===== RESULTS SUMMARY =====")
    print(f"{'Noise Level':<12} {'Augmentation':<15} {'Val Accuracy':<15} {'Test Accuracy':<15}")
    print("-" * 60)
    
    for result in all_results:
        aug_str = "Yes" if result['apply_augmentation'] else "No"
        print(f"{result['volume_factor']:<12.2f} {aug_str:<15} {result['best_val_acc']:<15.3%} {result['test_acc']:<15.3%}")
    
    # Find best configuration
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]['test_acc'])
    best_result = all_results[best_idx]
    
    print("\nBest configuration:")
    print(f"  Noise level: {best_result['volume_factor']:.2f}")
    print(f"  Augmentation: {'Yes' if best_result['apply_augmentation'] else 'No'}")
    print(f"  Validation accuracy: {best_result['best_val_acc']:.3%}")
    print(f"  Test accuracy: {best_result['test_acc']:.3%}")


if __name__ == "__main__":
    train_conformer_with_noise_and_augmentation()