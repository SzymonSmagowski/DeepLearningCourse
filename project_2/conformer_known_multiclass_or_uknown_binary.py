#!/usr/bin/env python3

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

class BinaryDatasetBuilder:
    """
    Dataset builder specifically for binary classification (known vs unknown)
    """
    def __init__(self, train, val, test, known_classes, seed=0):
        self._orig = dict(train=copy.deepcopy(train),
                          val=copy.deepcopy(val),
                          test=copy.deepcopy(test))
        self._train = self._orig["train"]
        self._val = self._orig["val"]
        self._test = self._orig["test"]
        self.seed = seed
        self.known_classes = set(known_classes)
        
        # Class mapping for binary classification
        self.label2id = {"known": 0, "unknown": 1}
        self.id2label = {0: "known", 1: "unknown"}
        
        # Config
        self._trim_ratio = None
        self._balance_ratio = None  # Known:Unknown ratio
        
    def trim(self, ratio: float):
        """Keep specified percentage of the dataset"""
        self._trim_ratio = ratio
        return self
        
    def balance(self, ratio: float):
        """Set ratio between known and unknown classes (known:unknown)"""
        self._balance_ratio = ratio
        return self
    
    def _apply_binary_labels(self, split):
        """Convert multi-class labels to binary (known vs unknown)"""
        for entry in split:
            if entry["label"] in self.known_classes:
                entry["original_label"] = entry["label"]
                entry["label"] = "known"
                entry["label_id"] = self.label2id["known"]
            else:
                entry["original_label"] = entry["label"]
                entry["label"] = "unknown"
                entry["label_id"] = self.label2id["unknown"]
        return split
    
    def _apply_trim(self, split):
        """Trim dataset while maintaining class balance"""
        if self._trim_ratio is None:
            return split
            
        known = [e for e in split if e["label"] == "known"]
        unknown = [e for e in split if e["label"] == "unknown"]
        
        # Calculate number of samples to keep for each class
        n_known = int(len(known) * self._trim_ratio)
        n_unknown = int(len(unknown) * self._trim_ratio)
        
        # Sample randomly
        random.seed(self.seed)
        known_sample = random.sample(known, n_known) if n_known < len(known) else known
        unknown_sample = random.sample(unknown, n_unknown) if n_unknown < len(unknown) else unknown
        
        return known_sample + unknown_sample
    
    def _apply_balance(self, split):
        """Balance dataset according to the specified known:unknown ratio"""
        if self._balance_ratio is None:
            return split
            
        known = [e for e in split if e["label"] == "known"]
        unknown = [e for e in split if e["label"] == "unknown"]
        
        # Calculate target counts
        known_ratio = self._balance_ratio / (1 + self._balance_ratio)
        unknown_ratio = 1 - known_ratio
        
        total_size = min(len(known) / known_ratio, len(unknown) / unknown_ratio)
        target_known = int(total_size * known_ratio)
        target_unknown = int(total_size * unknown_ratio)
        
        # Sample to achieve target ratio
        random.seed(self.seed)
        known_sample = random.sample(known, target_known) if target_known < len(known) else known
        unknown_sample = random.sample(unknown, target_unknown) if target_unknown < len(unknown) else unknown
        
        return known_sample + unknown_sample
    
    def build(self, verbose=True):
        """Build the binary classification datasets"""
        splits = dict(train=self._train, val=self._val, test=self._test)
        
        for name, split in splits.items():
            # Apply transformations
            split[:] = self._apply_binary_labels(split)
            split[:] = self._apply_trim(split)
            split[:] = self._apply_balance(split)
            
            if verbose:
                known_count = sum(1 for e in split if e["label"] == "known")
                unknown_count = sum(1 for e in split if e["label"] == "unknown")
                total = known_count + unknown_count
                print(f"{name:>5}: {total:4d} samples | "
                      f"Known: {known_count} ({known_count/total:.1%}), "
                      f"Unknown: {unknown_count} ({unknown_count/total:.1%})")
        
        return splits["train"], splits["val"], splits["test"]


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
    model_name = f"{config['model_type']}"
    if 'd_model' in config:
        model_name += f"_d{config['d_model']}"
    
    model_name += f"_l{config['num_layers']}"
    
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{timestamp}_{model_name}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def create_conformer_model(config, num_classes, device):
    """Create Conformer model based on configuration"""
    model = models.ConformerClassifier(
        n_classes=num_classes,
        n_mfcc=config.get('n_mfcc', config['input_size']),
        d_model=config.get('d_model', 128),
        num_layers=config['num_layers'],
        num_heads=config.get('num_heads', 4),
        kernel_size=config.get('kernel_size', 31),
        expansion_factor=config.get('expansion_factor', 4),
        dropout=config.get('dropout', 0.1),
        pool=config.get('pool', 'mean')
    )
    
    return model.to(device)


def train_and_evaluate_conformer(config, train_loader, val_loader, test_loader, id2label, device, results_dir, early_stopping_patience=5):
    """Train and evaluate Conformer model with given configuration"""
    # Create model
    num_classes = len(id2label)
    model = create_conformer_model(config, num_classes, device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Define learning rate scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['learning_rate'] * 0.1
        )
    elif config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
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
    plt.title(f'Confusion Matrix - Conformer')
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
    confusion_matrix_path = f"{results_dir}/plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_conformer_cm.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Plot training history
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
    
    plt.suptitle(f'Conformer Model (d_model={config["d_model"]}, layers={config["num_layers"]})')
    plt.tight_layout()
    
    history_path = f"{results_dir}/plots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_conformer_history.png"
    plt.savefig(history_path)
    plt.close()
    
    # Save results
    results_file = save_results(
        config, history, test_metrics, id2label, results_dir
    )
    
    return {
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'results_file': results_file
    }


def main():
    parser = argparse.ArgumentParser(description='Run Conformer Model for Classification Task')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/speech_commands_v0.01',
                      help='Path to the Speech Commands dataset')
    parser.add_argument('--classes', type=str, nargs='+', 
                      default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
                      help='List of classes to use for classification')
    parser.add_argument('--binary', action='store_true',
                      help='Whether to use binary classification (known vs unknown) instead of multiclass')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--results_dir', type=str, default='results/conformer',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set up device and seed
    device = get_device()
    set_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    train_, valid_, test_ = load_from_path(Path(args.data_path))
    
    if args.binary:
        # Binary classification setup (known vs unknown)
        
        builder = BinaryDatasetBuilder(train_, valid_, test_, 
                                     known_classes=args.classes,
                                     seed=args.seed)
        
        train, valid, test = builder.balance(1.0).build()  # 1:1 ratio of known:unknown
    else:
        # Multiclass classification among known classes
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
    
    # Define Conformer model configuration
    conformer_config = {
        'model_type': 'conformer',
        'input_size': input_size,
        'n_mfcc': input_size,
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
    
    # Train and evaluate
    print("Training Conformer model:")
    print(f"  d_model: {conformer_config['d_model']}, num_layers: {conformer_config['num_layers']}")
    
    result = train_and_evaluate_conformer(
        conformer_config, train_loader, val_loader, test_loader, 
        builder.id2label, device, args.results_dir, 
        early_stopping_patience=5
    )
    
    print(f"\nTraining completed:")
    print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"  Macro F1 score: {result['test_metrics']['f1']:.4f}")
    print(f"  Results saved to: {result['results_file']}")


if __name__ == "__main__":
    main()