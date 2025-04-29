#!/usr/bin/env python3
"""
Compare different models for binary classification (speech vs. silence/noise).
Tests LSTM, GRU, and Conformer models with specified configurations.
"""

import argparse
import os
import json
import time
import datetime
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from pathlib import Path

# Import custom modules
from helpers.binary_noise_balanced_dataset import create_balanced_dataset_with_augmentation
import helpers.models as models


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
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


def create_model(model_type, config, input_size=None):
    """Create model based on type and configuration"""
    if model_type == 'lstm':
        if input_size is None:
            input_size = config.get('input_size', 40)
        
        model = models.LSTMModel(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 3),
            num_classes=2,  # Binary classification
            bidirectional=config.get('bidirectional', False),
            dropout=config.get('dropout', 0.3)
        )
    
    elif model_type == 'gru':
        if input_size is None:
            input_size = config.get('input_size', 40)
            
        model = models.GRUModel(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 3),
            num_classes=2,  # Binary classification
            bidirectional=config.get('bidirectional', False),
            dropout=config.get('dropout', 0.3)
        )
    
    elif model_type == 'conformer':
        if input_size is None:
            input_size = config.get('n_mfcc', 40)
            
        model = models.ConformerClassifier(
            n_classes=2,  # Binary classification
            n_mfcc=input_size,
            d_model=config.get('d_model', 128),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 4),
            kernel_size=config.get('kernel_size', 15),
            dropout=config.get('dropout', 0.1),
            pool=config.get('pool', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    scheduler_type = config.get('scheduler', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = config.get('scheduler_step_size', 10)
        gamma = config.get('scheduler_gamma', 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        epochs = config.get('epochs', 50)
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'none' or not scheduler_type:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


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
    """Evaluate model on test set and generate metrics"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Collect predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = np.mean(all_preds == all_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary'
    )
    
    # Class-specific metrics
    class_report = classification_report(
        all_targets, all_preds, 
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return accuracy, precision, recall, f1, cm, class_report


def train_and_evaluate(model_type, config, data_loaders, device, results_dir):
    """Train and evaluate a model with the given configuration"""
    train_loader, val_loader, test_loader, id2label = data_loaders
    
    # Get input size from data
    input_size = train_loader.dataset.tensors[0].shape[1]
    
    # Update config with actual input size
    if model_type in ['lstm', 'gru']:
        config['input_size'] = input_size
    else:
        config['n_mfcc'] = input_size
    
    # Create model
    model = create_model(model_type, config, input_size)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    epochs = config.get('epochs', 50)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    start_time = time.time()
    
    print(f"\n{'='*20} Training {model_type.upper()} Model {'='*20}")
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"[{epoch:02d}/{epochs}] "
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
            print(f"⏹️  Early stopping after {epoch} epochs "
                  f"(no val-improve for {early_stopping_patience} epochs).")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {datetime.timedelta(seconds=int(training_time))}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    accuracy, precision, recall, f1, cm, class_report = test_model(model, device, test_loader, id2label)
    
    print(f"\nTest Results for {model_type.upper()}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Save results
    results = {
        'model_type': model_type,
        'config': config,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'confusion_matrix': cm.tolist(),
        'class_report': class_report,
        'training_time': training_time
    }
    
    # Create unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{timestamp}_{model_type}_binary.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_file = f"{results_dir}/{timestamp}_{model_type}_binary.pt"
    torch.save({
        'model_type': model_type,
        'config': config,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'id2label': id2label
    }, model_file)
    
    print(f"Results saved to {results_file}")
    print(f"Model saved to {model_file}")
    
    # Plot training history
    plot_history(history, model_type, results_dir, timestamp)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, id2label, model_type, results_dir, timestamp)
    
    return results


def plot_history(history, model_type, results_dir, timestamp):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(f'Training History - {model_type.upper()}')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{results_dir}/{timestamp}_{model_type}_history.png")
    plt.close()


def plot_confusion_matrix(cm, id2label, model_type, results_dir, timestamp):
    """Plot confusion matrix"""
    labels = [id2label[i] for i in sorted(id2label.keys())]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot
    plt.savefig(f"{results_dir}/{timestamp}_{model_type}_cm.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare models for binary speech vs noise classification')
    parser.add_argument('--data_path', type=str, default='data/speech_commands_v0.01',
                      help='Path to the Speech Commands dataset')
    parser.add_argument('--results_dir', type=str, default='results/binary_comparison',
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--oversampling_factor', type=float, default=5.0,
                      help='Factor for oversampling noise examples')
    parser.add_argument('--undersampling_factor', type=float, default=0.3,
                      help='Factor for undersampling speech examples')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create dataset
    print("Creating balanced binary dataset...")
    data_loaders = create_balanced_dataset_with_augmentation(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed,
        samples_per_word=100,     # Take this many from each word class
        chunks_per_noise=60,      # Maximum chunks per noise file
        enforce_balance=True      # Ensure exact class balance
    )
    
    # Define model configurations
    lstm_config = {
        "hidden_size": 256,
        "num_layers": 3,
        "bidirectional": False,
        "dropout": 0.3,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-05,
        "epochs": 50,
        "scheduler": "step",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.5,
        "early_stopping_patience": 10
    }
    
    gru_config = {
        "hidden_size": 256,
        "num_layers": 3,
        "bidirectional": False,
        "dropout": 0.3,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-05,
        "epochs": 50,
        "scheduler": "step",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.5,
        "early_stopping_patience": 10
    }
    
    conformer_config = {
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
        "scheduler_gamma": 0.5,
        "early_stopping_patience": 10
    }
    
    # Create results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Save run configuration
    run_config = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'oversampling_factor': args.oversampling_factor,
        'undersampling_factor': args.undersampling_factor,
        'device': str(device),
        'models': {
            'lstm': lstm_config,
            'gru': gru_config,
            'conformer': conformer_config
        }
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{results_dir}/{timestamp}_run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Train and evaluate models
    results = {}
    
    # 1. LSTM
    lstm_results = train_and_evaluate('lstm', lstm_config, data_loaders, device, results_dir)
    results['lstm'] = lstm_results
    
    # 2. GRU
    gru_results = train_and_evaluate('gru', gru_config, data_loaders, device, results_dir)
    results['gru'] = gru_results
    
    # 3. Conformer
    conformer_results = train_and_evaluate('conformer', conformer_config, data_loaders, device, results_dir)
    results['conformer'] = conformer_results
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON OF MODEL PERFORMANCE")
    print("="*50)
    print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Time (s)':<10}")
    print("-"*50)
    
    for model_type, result in results.items():
        metrics = result['metrics']
        time_s = result['training_time']
        print(f"{model_type:<10} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{time_s:<10.1f}")
    
    print("="*50)
    
    # Plot comparison
    plot_comparison(results, results_dir, timestamp)
    
    # Save overall comparison
    with open(f"{results_dir}/{timestamp}_comparison.json", 'w') as f:
        json.dump({
            'run_config': run_config,
            'results': {k: {
                'metrics': v['metrics'],
                'training_time': v['training_time']
            } for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")


def plot_comparison(results, results_dir, timestamp):
    """Plot comparison of model performance"""
    model_types = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data
    data = {metric: [results[model]['metrics'][metric] for model in model_types] for metric in metrics}
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Bar width
    bar_width = 0.2
    r = np.arange(len(model_types))
    
    # Create bars
    for i, metric in enumerate(metrics):
        plt.bar(r + i * bar_width, data[metric], width=bar_width, label=metric.capitalize())
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Comparison of Model Performance')
    plt.xticks(r + bar_width * (len(metrics) - 1) / 2, [m.upper() for m in model_types])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Ensure y-axis starts at 0 and ends at 1
    plt.ylim(0, 1.0)
    
    # Add values on top of bars
    for i, metric in enumerate(metrics):
        for j, value in enumerate(data[metric]):
            plt.text(r[j] + i * bar_width, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{results_dir}/{timestamp}_model_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()