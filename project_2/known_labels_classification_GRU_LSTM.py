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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm

# Import from helper modules
from helpers.dataset_builder import DatasetBuilder
from helpers.utils import load_from_path, preprocess_dataset, plot_training_history
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


def train_epoch(model, device, train_loader, criterion, optimizer):
    """Train model for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    
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
        
        # Save predictions for metrics
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    return metrics


def validate(model, device, val_loader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    
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
            
            # Save predictions for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    return metrics


def test_model(model, device, test_loader, id2label):
    """Evaluate model on test set and generate complete metrics"""
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    
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
            
            # Collect predictions for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(all_targets, all_preds, 
                                  target_names=target_names,
                                  output_dict=True)
    
    # Calculate per-class metrics for macro averaging
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro'
    )
    
    return {
        'accuracy': correct / total,
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_results(config, history, test_metrics, id2label, results_dir):
    """Save training results to JSON file"""
    # Convert numpy arrays and other non-serializable objects to lists/standard Python types
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, list)):
            serializable_history[key] = [float(v) if isinstance(v, np.number) else v for v in value]
        else:
            serializable_history[key] = value
    
    serializable_metrics = {}
    for key, value in test_metrics.items():
        if isinstance(value, np.ndarray):
            # Handle confusion matrix
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle classification report (dictionary of dictionaries)
            serializable_metrics[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    serializable_metrics[key][k] = {kk: float(vv) if isinstance(vv, np.number) else vv 
                                                   for kk, vv in v.items()}
                else:
                    serializable_metrics[key][k] = float(v) if isinstance(v, np.number) else v
        elif isinstance(value, np.number):
            # Handle any scalar numpy values
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    # Convert id2label to serializable format (keys need to be strings in JSON)
    serializable_id2label = {str(k): v for k, v in id2label.items()}
    
    results = {
        'config': config,
        'history': serializable_history,
        'test_metrics': serializable_metrics,
        'id2label': serializable_id2label
    }
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model_type']}_h{config['hidden_size']}_l{config['num_layers']}"
    if 'bidirectional' in config and config['bidirectional']:
        model_name += "_bidir"
    
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{timestamp}_{model_name}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def create_model(config, num_classes, device):
    """Create model based on configuration"""
    model_type = config['model_type'].lower()
    
    if model_type == 'lstm':
        model = models.LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'], 
            num_layers=config['num_layers'],
            num_classes=num_classes,
            bidirectional=config.get('bidirectional', False),
            dropout=config.get('dropout', 0.0)
        )
    elif model_type == 'gru':
        model = models.GRUModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'], 
            num_layers=config['num_layers'],
            num_classes=num_classes,
            bidirectional=config.get('bidirectional', False),
            dropout=config.get('dropout', 0.0)
        )
    elif model_type == 'conformer':
        model = models.ConformerClassifier(
            n_classes=num_classes,
            n_mfcc=config.get('n_mfcc', config['input_size']),
            d_model=config.get('d_model', config['hidden_size']),
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', 4),
            kernel_size=config.get('kernel_size', 31),
            expansion_factor=config.get('expansion_factor', 4),
            dropout=config.get('dropout', 0.1),
            pool=config.get('pool', 'mean')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_and_evaluate(config, train_loader, val_loader, test_loader, id2label, device, results_dir, early_stopping_patience=5):
    """Train and evaluate model with given configuration"""
    # Create model
    num_classes = len(id2label)
    model = create_model(config, num_classes, device)
    
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
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] * 0.1
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
    
    for epoch in range(1, config['epochs'] + 1):
        # Train one epoch
        train_metrics = train_epoch(model, device, train_loader, criterion, optimizer)
        
        # Evaluate on validation set
        val_metrics = validate(model, device, val_loader, criterion)
        
        # Update learning rate
        if scheduler is not None:
            if config['scheduler'] == 'step':
                scheduler.step()
            elif config['scheduler'] == 'plateau':
                scheduler.step(val_metrics['loss'])
            elif config['scheduler'] == 'cosine':
                scheduler.step()
        
        # Save metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print progress
        print(f"[{epoch:02d}/{config['epochs']}] "
              f"train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.3f} – "
              f"val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.3f}")
        
        # Check if best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
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
    test_metrics = test_model(model, device, test_loader, id2label)
    print(f"Test metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  Macro Precision: {test_metrics['precision']:.3f}")
    print(f"  Macro Recall: {test_metrics['recall']:.3f}")
    print(f"  Macro F1 score: {test_metrics['f1']:.3f}")
    
    # Plot confusion matrix
    cm = test_metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {config["model_type"]}')
    plt.colorbar()
    classes = [id2label[i] for i in range(len(id2label))]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # Add text annotations for larger matrices
    if len(classes) < 15:  # Only add text for matrices that aren't too large
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save confusion matrix
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    confusion_matrix_path = f"{results_dir}/plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['model_type']}_cm.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Plot training history
    plot_history(history, config, results_dir)
    
    # Save results
    results_file = save_results(
        config, history, test_metrics, id2label, results_dir
    )
    
    return {
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'results_file': results_file
    }


def plot_history(history, config, results_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(f'{config["model_type"].upper()} Model (h={config["hidden_size"]}, l={config["num_layers"]})')
    plt.tight_layout()
    
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    history_path = f"{results_dir}/plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['model_type']}_history.png"
    plt.savefig(history_path)
    plt.close()


def create_model_comparison_figure(configs, results, results_dir):
    """Create a figure comparing the performance of different models"""
    model_names = [cfg['model_type'].upper() for cfg in configs]
    metrics = {
        'Accuracy': [r['test_metrics']['accuracy'] for r in results],
        'Precision': [r['test_metrics']['precision'] for r in results],
        'Recall': [r['test_metrics']['recall'] for r in results],
        'F1 Score': [r['test_metrics']['f1'] for r in results]
    }
    
    plt.figure(figsize=(12, 10))
    
    # Plot bar charts for each metric
    for i, (metric_name, metric_values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        bars = plt.bar(model_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.title(f'{metric_name} Comparison')
        plt.ylabel(metric_name)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.suptitle('Model Performance Comparison (Multiclass Classification)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    comparison_path = f"{results_dir}/plots/model_comparison.png"
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"Model comparison figure saved to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='Multiclass Speech Commands Classification')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/speech_commands_v0.01',
                      help='Path to the Speech Commands dataset')
    parser.add_argument('--classes', type=str, nargs='+', 
                      default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
                      help='List of classes to use for classification')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--results_dir', type=str, default='results/multiclass',
                      help='Directory to save results')
    
    # Model parameters - these will be overridden in the compare mode with specific configurations
    parser.add_argument('--model_type', type=str, default='lstm',
                      choices=['lstm', 'gru', 'conformer'],
                      help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size for model layers')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of layers in the model')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional RNN (for RNN-based models)')
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
                      choices=['step', 'plateau', 'cosine', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=10,
                      help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                      help='Gamma for schedulers (multiplicative factor)')
    
    # Model comparison mode
    parser.add_argument('--compare', action='store_true',
                      help='Compare specified model configurations')
    
    args = parser.parse_args()
    
    # Set up device and seed
    device = get_device()
    set_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    train_, valid_, test_ = load_from_path(Path(args.data_path))
    
    # Prepare dataset - only keep specified classes
    builder = DatasetBuilder(train_, valid_, test_, args.seed)
    train, valid, test = builder.keep_labels(args.classes, make_rest_unknown=False).build()
    
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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Get input size from data
    input_size = X_train.shape[1]  # Number of MFCC features
    
    # Create model configuration
    config = {
        'model_type': args.model_type,
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
    
    if args.compare:
        # Model comparison with specified configurations
        print("Starting model comparison with specified configurations...")
        
        # Define the exact configurations as specified
        configs = [
            # LSTM configuration
            {
                'model_type': 'lstm',
                'input_size': 40,
                'hidden_size': 256,
                'num_layers': 3,
                'bidirectional': False,
                'dropout': 0.3,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-05,
                'epochs': 50,
                'scheduler': 'step',
                'scheduler_step_size': 10,
                'scheduler_gamma': 0.5
            },
            
            # GRU configuration
            {
                'model_type': 'gru',
                'input_size': 40,
                'hidden_size': 256,
                'num_layers': 3,
                'bidirectional': False,
                'dropout': 0.3,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-05,
                'epochs': 50,
                'scheduler': 'step',
                'scheduler_step_size': 10,
                'scheduler_gamma': 0.5
            },
            
            # Conformer configuration
            {
                'model_type': 'conformer',
                'input_size': 40,
                'n_mfcc': 40,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 4,
                'kernel_size': 15,
                'expansion_factor': 4,
                'dropout': 0.1,
                'pool': 'mean',
                'optimizer': 'adamw',
                'learning_rate': 0.0005,
                'weight_decay': 0.01,
                'epochs': 50,
                'scheduler': 'cosine',
                'scheduler_step_size': 10,
                'scheduler_gamma': 0.5
            }
        ]
        
        # Train and evaluate each configuration
        results = []
        for i, cfg in enumerate(configs):
            print(f"\nConfiguration {i+1}/{len(configs)}:")
            print(f"  Model: {cfg['model_type']}, hidden_size: {cfg['hidden_size']}, "
                  f"num_layers: {cfg['num_layers']}")
            
            result = train_and_evaluate(
                cfg, train_loader, val_loader, test_loader, 
                builder.id2label, device, args.results_dir, 
                early_stopping_patience=args.early_stopping
            )
            results.append(result)
        
        # Summary of results
        print("\n======= MODEL COMPARISON RESULTS =======")
        print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        print("-" * 50)
        
        for i, (cfg, result) in enumerate(zip(configs, results)):
            model_name = cfg['model_type'].upper()
            acc = result['test_metrics']['accuracy']
            prec = result['test_metrics']['precision']
            rec = result['test_metrics']['recall']
            f1 = result['test_metrics']['f1']
            
            print(f"{model_name:<10} {acc:.4f}      {prec:.4f}      {rec:.4f}      {f1:.4f}")
        
        # Find best configuration based on accuracy
        best_idx = max(range(len(results)), key=lambda i: results[i]['test_metrics']['accuracy'])
        best_config = configs[best_idx]
        best_result = results[best_idx]
        
        print("\nBest model based on accuracy:")
        print(f"  Model: {best_config['model_type']}, hidden_size: {best_config['hidden_size']}, "
              f"num_layers: {best_config['num_layers']}")
        print(f"  Test accuracy: {best_result['test_metrics']['accuracy']:.4f}")
        print(f"  Macro precision: {best_result['test_metrics']['precision']:.4f}")
        print(f"  Macro recall: {best_result['test_metrics']['recall']:.4f}")
        print(f"  Macro F1 score: {best_result['test_metrics']['f1']:.4f}")
        
        # Save best configuration
        with open(f"{args.results_dir}/best_config.json", 'w') as f:
            json.dump({
                'config': best_config,
                'result': {
                    'accuracy': float(best_result['test_metrics']['accuracy']),
                    'precision': float(best_result['test_metrics']['precision']),
                    'recall': float(best_result['test_metrics']['recall']),
                    'f1': float(best_result['test_metrics']['f1'])
                }
            }, f, indent=2)
        
        # Create comparison figure
        create_model_comparison_figure(configs, results, args.results_dir)
        
    else:
        # Train and evaluate with single configuration
        print("Training with configuration:")
        print(f"  Model: {config['model_type']}, hidden_size: {config['hidden_size']}, "
              f"num_layers: {config['num_layers']}")
        
        result = train_and_evaluate(
            config, train_loader, val_loader, test_loader, 
            builder.id2label, device, args.results_dir, 
            early_stopping_patience=args.early_stopping
        )
        
        print(f"\nTraining completed:")
        print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Macro F1 score: {result['test_metrics']['f1']:.4f}")
        print(f"  Results saved to: {result['results_file']}")


if __name__ == "__main__":
    main()