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

import helpers.models as models  # Make sure LSTMModel is added to this file

from helpers.model_runner_utils import set_seed, get_device, train_epoch, validate, test_model, _save_results

from helpers.speech_datasets import (
    get_task1_dataloaders, get_task2_dataloaders,
    get_task3_dataloaders, get_task4_dataloaders,
)


def save_results(config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir):
    model_name = f"lstm_h{config['hidden_size']}_l{config['num_layers']}"
    if config['bidirectional']:
        model_name += "_bidir"
    return _save_results(model_name, config, history, best_val_acc, test_acc, conf_matrix, id2label, results_dir)


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
                device, id2label, args.results_dir, 
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
            device, id2label, args.results_dir, 
            early_stopping_patience=args.early_stopping
        )


if __name__ == "__main__":
    main()