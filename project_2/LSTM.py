import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Import from helper modules
from helpers.dataset_builder import DatasetBuilder
from helpers.utils import load_from_path, preprocess_dataset
import helpers.models as models  # Make sure LSTMModel is added to this file


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


def save_results(config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir):
    """Save training results to JSON file"""
    results = {
        'config': config,
        'history': history,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'confusion_matrix': conf_matrix.tolist(),
        'id2label': id2label
    }
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_h{config['hidden_size']}_l{config['num_layers']}"
    if config['bidirectional']:
        model_name += "_bidir"
    
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{timestamp}_{model_name}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def train_and_evaluate(config, train_loader, val_loader, test_loader, device, id2label, results_dir, early_stopping_patience=5):
    """Train and evaluate model with given configuration"""
    # Create LSTM model
    model = models.LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'], 
        num_layers=config['num_layers'],
        num_classes=len(id2label),
        bidirectional=config['bidirectional'],
        dropout=config['dropout']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=0.9, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Define learning rate scheduler
    if config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], 
                                     patience=2, verbose=True)
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
    
    for epoch in range(1, config['epochs'] + 1):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        
        # Update learning rate
        if scheduler is not None:
            if config['scheduler'] == 'step':
                scheduler.step()
            elif config['scheduler'] == 'plateau':
                scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"[{epoch:02d}/{config['epochs']}] "
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
        conf_matrix, id2label, results_dir
    )
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'results_file': results_file
    }


def main():
    parser = argparse.ArgumentParser(description='Speech Commands Classification with LSTM')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/speech_commands_v0.01',
                      help='Path to the Speech Commands dataset')
    parser.add_argument('--subset', action='store_true',
                      help='Use only yes/no subset instead of full dataset')
    parser.add_argument('--unknown_ratio', type=float, default=0.2,
                      help='Ratio of unknown samples to keep')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--results_dir', type=str, default='results/lstm',
                      help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size for LSTM layers')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'adamw', 'sgd'],
                      help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Maximum number of epochs')
    parser.add_argument('--early_stopping', type=int, default=5,
                      help='Early stopping patience (epochs)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='step',
                      choices=['step', 'plateau', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=10,
                      help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                      help='Gamma for schedulers (multiplicative factor)')
    
    # Hyperparameter search
    parser.add_argument('--search', action='store_true',
                      help='Perform hyperparameter search')
    
    args = parser.parse_args()
    
    # Set up device and seed
    device = get_device()
    set_seed(args.seed)
    
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
            
    # Extract features
    print(f"Extracting features...")
    X_train, y_train = preprocess_dataset(train, "train")
    X_val, y_val = preprocess_dataset(valid, "valid")
    X_test, y_test = preprocess_dataset(test, "test")
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Prepare data loaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Get input size from data
    input_size = X_train.shape[1]  # Number of MFCC features
    
    # Create model configuration
    config = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'bidirectional': args.bidirectional,
        'dropout': args.dropout,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'scheduler': args.scheduler,
        'scheduler_step_size': args.scheduler_step_size,
        'scheduler_gamma': args.scheduler_gamma,
    }
    
    if args.search:
        # Hyperparameter search
        print("Starting hyperparameter search...")
        
        # Define specific configurations to try (20 total)
        configs = [
            # Small networks
            {**config, "hidden_size": 32, "num_layers": 1, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 32, "num_layers": 1, "bidirectional": False, "dropout": 0.3},
            {**config, "hidden_size": 64, "num_layers": 1, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 64, "num_layers": 1, "bidirectional": False, "dropout": 0.3},
            
            # Medium networks
            {**config, "hidden_size": 64, "num_layers": 2, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 64, "num_layers": 2, "bidirectional": False, "dropout": 0.3},
            {**config, "hidden_size": 128, "num_layers": 1, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 128, "num_layers": 1, "bidirectional": False, "dropout": 0.3},
            {**config, "hidden_size": 128, "num_layers": 2, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 128, "num_layers": 2, "bidirectional": False, "dropout": 0.3},
            
            # Large networks
            {**config, "hidden_size": 256, "num_layers": 2, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 256, "num_layers": 2, "bidirectional": False, "dropout": 0.3},
            {**config, "hidden_size": 256, "num_layers": 3, "bidirectional": False, "dropout": 0.1},
            {**config, "hidden_size": 256, "num_layers": 3, "bidirectional": False, "dropout": 0.3},
            
            # Bidirectional variants
            {**config, "hidden_size": 64, "num_layers": 1, "bidirectional": True, "dropout": 0.1},
            {**config, "hidden_size": 64, "num_layers": 1, "bidirectional": True, "dropout": 0.3},
            {**config, "hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.1},
            {**config, "hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.3},
            {**config, "hidden_size": 256, "num_layers": 2, "bidirectional": True, "dropout": 0.1},
            {**config, "hidden_size": 256, "num_layers": 2, "bidirectional": True, "dropout": 0.3},
        ]
        
        print(f"Total configurations to try: {len(configs)}")
        
        # Train and evaluate each configuration
        results = []
        for i, cfg in enumerate(configs):
            print(f"\nConfiguration {i+1}/{len(configs)}:")
            print(f"  hidden_size: {cfg['hidden_size']}, num_layers: {cfg['num_layers']}, " 
                  f"bidirectional: {cfg['bidirectional']}, dropout: {cfg['dropout']}")
            
            result = train_and_evaluate(
                cfg, train_loader, val_loader, test_loader, 
                device, builder.id2label, args.results_dir, 
                early_stopping_patience=args.early_stopping
            )
            results.append(result)
        
        # Find best configuration
        best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
        best_config = configs[best_idx]
        best_result = results[best_idx]
        
        print("\nBest configuration:")
        print(f"  {best_config}")
        print(f"  Validation accuracy: {best_result['best_val_acc']:.3%}")
        print(f"  Test accuracy: {best_result['test_acc']:.3%}")
        
        # Save best configuration
        with open(f"{args.results_dir}/best_config.json", 'w') as f:
            json.dump({
                'config': best_config,
                'result': {
                    'val_acc': best_result['best_val_acc'],
                    'test_acc': best_result['test_acc']
                }
            }, f, indent=2)
    
    else:
        # Train and evaluate with single configuration
        print("Training with configuration:")
        print(f"  {config}")
        
        train_and_evaluate(
            config, train_loader, val_loader, test_loader, 
            device, builder.id2label, args.results_dir, 
            early_stopping_patience=args.early_stopping
        )


if __name__ == "__main__":
    main()