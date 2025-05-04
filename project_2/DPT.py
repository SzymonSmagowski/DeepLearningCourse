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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Import from helper modules
from helpers.dataset_builder import DatasetBuilder
from helpers.utils import load_from_path, preprocess_dataset
import helpers.models as models  # Using the added DualPathTransformer model

from helpers.model_runner_utils import set_seed, get_device, train_epoch, validate, test_model, _save_results

def save_results(config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir):
    model_name = f"dpt_d{config['d_model']}_h{config['n_heads']}_l{config['num_layers']}_dr{config['dropout']}"
    if config['version'] == 2:
        model_name += "_v2"
    model_name += f"_{config['fusion_method']}"
    return _save_results(model_name, config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir)


def train_and_evaluate(config, train_loader, val_loader, test_loader, device, id2label, results_dir, early_stopping_patience=5):
    """Train and evaluate model with given configuration"""
    # Create DualPathTransformer model
    if config['version'] == 1:
        model = models.DualPathTransformer(
            n_classes=len(id2label),
            n_mfcc=config['n_mfcc'],
            d_model=config['d_model'], 
            n_heads=config['n_heads'],
            num_layers=config['num_layers'], 
            dim_ff=config['dim_ff'],
            dropout=config['dropout'],
            fusion_method=config['fusion_method']
        ).to(device)
    else:  # version 2
        model = models.DualPathTransformerV2(
            n_classes=len(id2label),
            n_mfcc=config['n_mfcc'],
            d_model=config['d_model'], 
            n_heads=config['n_heads'],
            num_layers=config['num_layers'], 
            dim_ff=config['dim_ff'],
            dropout=config['dropout'],
            fusion_method=config['fusion_method']
        ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Define learning rate scheduler
    if config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], 
                                     patience=2, verbose=True)
    elif config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
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
            if config['scheduler'] == 'step' or config['scheduler'] == 'cosine':
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
    parser = argparse.ArgumentParser(description='Speech Commands Classification with Dual-Path Transformer')
    
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
    parser.add_argument('--results_dir', type=str, default='results/dpt',
                      help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--version', type=int, default=2, choices=[1, 2],
                      help='DPT version to use (1 or 2)')
    parser.add_argument('--d_model', type=int, default=128,
                      help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--dim_ff', type=int, default=256,
                      help='Dimension of feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--fusion_method', type=str, default='concat',
                      choices=['add', 'multiply', 'concat', 'gated'],
                      help='Method to fuse frequency and temporal paths')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                      choices=['adam', 'adamw'],
                      help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Maximum number of epochs')
    parser.add_argument('--early_stopping', type=int, default=5,
                      help='Early stopping patience (epochs)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['step', 'plateau', 'cosine', 'none'],
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
    n_mfcc = X_train.shape[1]  # Number of MFCC features
    
    # Create model configuration
    config = {
        'version': args.version,
        'n_mfcc': n_mfcc,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'num_layers': args.num_layers,
        'dim_ff': args.dim_ff,
        'dropout': args.dropout,
        'fusion_method': args.fusion_method,
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
        
        # Define base configurations to try (without dropout)
        base_configs = [
            # Test both versions with different fusion methods
            {'version': 1, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'add'},
            {'version': 1, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'concat'},
            {'version': 2, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'add'},
            {'version': 2, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'multiply'},
            {'version': 2, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'concat'},
            {'version': 2, 'd_model': 128, 'n_heads': 4, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'gated'},
            
            # Small models
            {'version': 2, 'd_model': 64, 'n_heads': 2, 'num_layers': 2, 'dim_ff': 128, 'fusion_method': 'concat'},
            {'version': 2, 'd_model': 96, 'n_heads': 3, 'num_layers': 3, 'dim_ff': 192, 'fusion_method': 'concat'},
            
            # Varying dimensions and heads
            {'version': 2, 'd_model': 128, 'n_heads': 2, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'concat'},
            {'version': 2, 'd_model': 128, 'n_heads': 8, 'num_layers': 4, 'dim_ff': 256, 'fusion_method': 'concat'},
            
            # Varying layers
            {'version': 2, 'd_model': 128, 'n_heads': 4, 'num_layers': 6, 'dim_ff': 256, 'fusion_method': 'concat'},
            
            # Large models
            {'version': 2, 'd_model': 192, 'n_heads': 6, 'num_layers': 4, 'dim_ff': 384, 'fusion_method': 'concat'},
            {'version': 2, 'd_model': 256, 'n_heads': 8, 'num_layers': 4, 'dim_ff': 512, 'fusion_method': 'concat'},
        ]
        
        # Add different dropout values for each base configuration
        dropout_values = [0.1, 0.2, 0.3]
        configs = []
        
        # Number of base configs we can fit to keep total near 20
        max_base_configs = min(len(base_configs), 7)  # 7 base * 3 dropout = 21 total configs
        
        for i, base_cfg in enumerate(base_configs[:max_base_configs]):
            for dropout in dropout_values:
                configs.append({
                    **config,  # Base settings from args
                    **base_cfg,  # Architecture specific settings
                    'dropout': dropout  # Varying dropout
                })
        
        print(f"Total configurations to try: {len(configs)}")
        
        # Train and evaluate each configuration
        results = []
        for i, cfg in enumerate(configs):
            print(f"\nConfiguration {i+1}/{len(configs)}:")
            print(f"  Version: {cfg['version']}, d_model: {cfg['d_model']}, n_heads: {cfg['n_heads']}, "
                  f"num_layers: {cfg['num_layers']}, fusion_method: {cfg['fusion_method']}, "
                  f"dropout: {cfg['dropout']}")
            
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
        
        print("\n===========================================================")
        print("BEST CONFIGURATION FOUND:")
        print(f"  Version: {best_config['version']}")
        print(f"  d_model: {best_config['d_model']}")
        print(f"  n_heads: {best_config['n_heads']}")
        print(f"  num_layers: {best_config['num_layers']}")
        print(f"  dim_ff: {best_config['dim_ff']}")
        print(f"  dropout: {best_config['dropout']}")
        print(f"  fusion_method: {best_config['fusion_method']}")
        print(f"  Validation accuracy: {best_result['best_val_acc']:.3%}")
        print(f"  Test accuracy: {best_result['test_acc']:.3%}")
        print("===========================================================")
        
        # Extra: Try a larger version of the best configuration
        print("\nTraining larger version of the best configuration...")
        
        # Scale up the best configuration
        larger_config = {**best_config}
        larger_config['d_model'] = min(2 * best_config['d_model'], 512)  # Double model dimension, max 512
        larger_config['n_heads'] = min(2 * best_config['n_heads'], 16)  # Double heads, max 16
        larger_config['num_layers'] = min(best_config['num_layers'] + 2, 8)  # Add 2 layers, max 8
        larger_config['dim_ff'] = min(2 * best_config['dim_ff'], 1024)  # Double FF dim, max 1024
        
        print(f"  d_model: {larger_config['d_model']}, n_heads: {larger_config['n_heads']}, "
              f"num_layers: {larger_config['num_layers']}, dim_ff: {larger_config['dim_ff']}, "
              f"dropout: {larger_config['dropout']}")
        
        larger_result = train_and_evaluate(
            larger_config, train_loader, val_loader, test_loader, 
            device, builder.id2label, args.results_dir, 
            early_stopping_patience=args.early_stopping
        )
        
        print("\n===========================================================")
        print("LARGER MODEL RESULTS:")
        print(f"  Validation accuracy: {larger_result['best_val_acc']:.3%}")
        print(f"  Test accuracy: {larger_result['test_acc']:.3%}")
        print(f"  Improvement over best: {larger_result['test_acc'] - best_result['test_acc']:.3%}")
        print("===========================================================")
        
        # Save best configurations
        with open(f"{args.results_dir}/best_config.json", 'w') as f:
            json.dump({
                'best_config': best_config,
                'best_result': {
                    'val_acc': best_result['best_val_acc'],
                    'test_acc': best_result['test_acc']
                },
                'larger_config': larger_config,
                'larger_result': {
                    'val_acc': larger_result['best_val_acc'],
                    'test_acc': larger_result['test_acc']
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