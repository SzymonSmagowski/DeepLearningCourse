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

from helpers.utils import load_from_path, preprocess_dataset
import helpers.models as models  # Using the added ConformerClassifier model

from helpers.model_runner_utils import set_seed, get_device, train_epoch, validate, test_model, _save_results

from helpers.speech_datasets import (
    get_task1_dataloaders, get_task2_dataloaders,
    get_task3_dataloaders, get_task4_dataloaders,
)



def save_results(config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"conformer_d{config['d_model']}_h{config['num_heads']}_l{config['num_layers']}"
    return _save_results(model_name, config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir)


def train_and_evaluate(config, train_loader, val_loader, test_loader, device, id2label, results_dir, early_stopping_patience=5):
    """Train and evaluate model with given configuration"""
    # Create Conformer model
    model = models.ConformerClassifier(
        n_classes=len(id2label),
        n_mfcc=config['n_mfcc'],
        d_model=config['d_model'], 
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        kernel_size=config['kernel_size'],
        expansion_factor=config['expansion_factor'],
        dropout=config['dropout'],
        pool=config['pool']
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
    parser = argparse.ArgumentParser(description='Speech Commands Classification with Conformer')
    
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
    parser.add_argument('--results_dir', type=str, default='results/conformer',
                      help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=144,
                      help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of conformer layers')
    parser.add_argument('--kernel_size', type=int, default=31,
                      help='Kernel size for depthwise convolution')
    parser.add_argument('--expansion_factor', type=int, default=4,
                      help='Expansion factor for feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--pool', type=str, default='mean',
                      choices=['cls', 'mean'],
                      help='Pooling method for classification')
    
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
    
    parser.add_argument(
        "--task", type=int, default=1, choices=[1, 2, 3, 4],
        help="Which dataset task definition to use"
    )

    args = parser.parse_args()

    # Set up device and seed
    device = get_device()
    set_seed(args.seed)

    known_10 = ['yes','no','up','down','left','right','on','off','stop','go']

    task2loader = {
        1: lambda: get_task1_dataloaders(
                data_dir=args.data_path, known_commands=known_10,
                batch_size=args.batch_size, seed=args.seed),
        2: lambda: get_task2_dataloaders(
                data_dir=args.data_path,
                batch_size=args.batch_size, seed=args.seed),
        3: lambda: get_task3_dataloaders(
                data_dir=args.data_path, known_commands=known_10,
                batch_size=args.batch_size, seed=args.seed),
        4: lambda: get_task4_dataloaders(
                data_dir=args.data_path, known_commands=known_10,
                batch_size=args.batch_size, seed=args.seed),
    }

    train_loader, val_loader, test_loader, id2label = task2loader[args.task]()
    input_size = 40

    # Create model configuration
    config = {
        'n_mfcc': input_size,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'kernel_size': args.kernel_size,
        'expansion_factor': args.expansion_factor,
        'dropout': args.dropout,
        'pool': args.pool,
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
            # Small models - varying kernel sizes and layers
            {**config, 'd_model': 64, 'num_heads': 2, 'num_layers': 2, 'kernel_size': 15, 'expansion_factor': 2, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 64, 'num_heads': 2, 'num_layers': 2, 'kernel_size': 31, 'expansion_factor': 2, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 96, 'num_heads': 3, 'num_layers': 2, 'kernel_size': 15, 'expansion_factor': 2, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 96, 'num_heads': 3, 'num_layers': 3, 'kernel_size': 15, 'expansion_factor': 2, 'dropout': 0.1, 'pool': 'mean'},
            
            # Medium models - varying dimensions, heads and pooling
            {**config, 'd_model': 128, 'num_heads': 4, 'num_layers': 3, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 128, 'num_heads': 4, 'num_layers': 3, 'kernel_size': 31, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 128, 'num_heads': 4, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 128, 'num_heads': 8, 'num_layers': 3, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 128, 'num_heads': 4, 'num_layers': 3, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'cls'},
            
            # Medium models - dropout variants
            {**config, 'd_model': 144, 'num_heads': 4, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.0, 'pool': 'mean'},
            {**config, 'd_model': 144, 'num_heads': 4, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 144, 'num_heads': 4, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.2, 'pool': 'mean'},
            {**config, 'd_model': 144, 'num_heads': 4, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.3, 'pool': 'mean'},
            
            # Larger models - varying dimensions and expansion factors
            {**config, 'd_model': 192, 'num_heads': 6, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 192, 'num_heads': 6, 'num_layers': 6, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 192, 'num_heads': 6, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 8, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 256, 'num_heads': 8, 'num_layers': 4, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            {**config, 'd_model': 256, 'num_heads': 8, 'num_layers': 6, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
            
            # Extra large model (for best configuration scaled up)
            {**config, 'd_model': 320, 'num_heads': 10, 'num_layers': 6, 'kernel_size': 15, 'expansion_factor': 4, 'dropout': 0.1, 'pool': 'mean'},
        ]
        
        print(f"Total configurations to try: {len(configs)}")
        
        # Train and evaluate each configuration
        results = []
        for i, cfg in enumerate(configs):
            print(f"\nConfiguration {i+1}/{len(configs)}:")
            print(f"  d_model: {cfg['d_model']}, num_heads: {cfg['num_heads']}, "
                  f"num_layers: {cfg['num_layers']}, kernel_size: {cfg['kernel_size']}, "
                  f"expansion_factor: {cfg['expansion_factor']}, dropout: {cfg['dropout']}, "
                  f"pool: {cfg['pool']}")
            
            result = train_and_evaluate(
                cfg, train_loader, val_loader, test_loader, 
                device, id2label, args.results_dir, 
                early_stopping_patience=args.early_stopping
            )
            results.append(result)
        
        # Find best configuration
        best_idx = max(range(len(results)), key=lambda i: results[i]['test_acc'])
        best_config = configs[best_idx]
        best_result = results[best_idx]
        
        print("\n===========================================================")
        print("BEST CONFIGURATION FOUND:")
        print(f"  d_model: {best_config['d_model']}")
        print(f"  num_heads: {best_config['num_heads']}")
        print(f"  num_layers: {best_config['num_layers']}")
        print(f"  kernel_size: {best_config['kernel_size']}")
        print(f"  expansion_factor: {best_config['expansion_factor']}")
        print(f"  dropout: {best_config['dropout']}")
        print(f"  pool: {best_config['pool']}")
        print(f"  Validation accuracy: {best_result['best_val_acc']:.3%}")
        print(f"  Test accuracy: {best_result['test_acc']:.3%}")
        print("===========================================================")
        
        # Extra: Try a larger version of the best configuration
        print("\nTraining larger version of the best configuration...")
        
        # Scale up the best configuration
        larger_config = {**best_config}
        larger_config['d_model'] = min(2 * best_config['d_model'], 512)  # Double model dimension, max 512
        larger_config['num_heads'] = min(2 * best_config['num_heads'], 16)  # Double heads, max 16
        larger_config['num_layers'] = min(best_config['num_layers'] + 2, 8)  # Add 2 layers, max 8
        
        print(f"  d_model: {larger_config['d_model']}, num_heads: {larger_config['num_heads']}, "
              f"num_layers: {larger_config['num_layers']}")
        
        larger_result = train_and_evaluate(
            larger_config, train_loader, val_loader, test_loader, 
            device, id2label, args.results_dir, 
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
            device, id2label, args.results_dir, 
            early_stopping_patience=args.early_stopping
        )


if __name__ == "__main__":
    main()