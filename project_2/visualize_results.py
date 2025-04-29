#!/usr/bin/env python3
"""
Visualization tool for speech command classification results.
This script generates plots from saved results files.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_history(history, title="Training History", figsize=(12, 5), save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", figsize=(10, 8), save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (array): Confusion matrix array
        class_names (list): List of class names
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_single_result(results_file, output_dir=None):
    """Visualize a single results file"""
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(results_file).parent / 'visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    history = results['history']
    id2label = results['id2label']
    cm = np.array(results['confusion_matrix'])
    config = results['config']
    
    # Create base filename
    base_filename = Path(results_file).stem
    
    # Create model name for title
    model_name = f"{config['model_type'].upper()} (h={config['hidden_size']}, l={config['num_layers']})"
    if config['bidirectional']:
        model_name += " Bidirectional"
    
    # Plot training history
    plot_training_history(
        history, 
        title=f"Training History - {model_name}",
        save_path=f"{output_dir}/{base_filename}_history.png"
    )
    
    # Plot confusion matrix
    if isinstance(id2label, dict):
        if all(k.isdigit() for k in id2label.keys()):
            # Convert string keys to integers for proper ordering
            id2label = {int(k): v for k, v in id2label.items()}
        class_names = [id2label[i] for i in sorted(id2label.keys())]
    else:
        class_names = id2label
    
    plot_confusion_matrix(
        cm, 
        class_names=class_names,
        title=f"Confusion Matrix - {model_name}",
        save_path=f"{output_dir}/{base_filename}_cm.png"
    )
    
    print(f"Visualizations for {base_filename} saved to {output_dir}")


def visualize_multiple_results(results_dir, output_dir=None, top_n=None):
    """
    Compare multiple results files.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save visualizations
        top_n: Only show top N models by test accuracy
    """
    # Set output directory
    if output_dir is None:
        output_dir = Path(results_dir) / 'comparisons'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result files
    result_files = glob.glob(f"{results_dir}/*.json")
    result_files = [f for f in result_files if not f.endswith('best_config.json')]
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Load all results
    all_results = []
    for file in result_files:
        try:
            with open(file, 'r') as f:
                result = json.load(f)
                
                # Extract key information
                model_info = {
                    'file': file,
                    'model_type': result['config']['model_type'],
                    'hidden_size': result['config']['hidden_size'],
                    'num_layers': result['config']['num_layers'],
                    'bidirectional': result['config']['bidirectional'],
                    'val_acc': result['best_val_acc'],
                    'test_acc': result['test_acc'],
                    'epochs': len(result['history']['train_loss']),
                    'final_train_loss': result['history']['train_loss'][-1],
                    'final_val_loss': result['history']['val_loss'][-1]
                }
                
                all_results.append(model_info)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Convert to dataframe
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No valid results to compare")
        return
    
    # Sort by test accuracy
    df = df.sort_values('test_acc', ascending=False)
    
    # Limit to top N if specified
    if top_n is not None:
        df = df.head(top_n)
    
    # Create model names
    df['model_name'] = df.apply(
        lambda row: f"{row['model_type']} (h={row['hidden_size']}, l={row['num_layers']}{', bidir' if row['bidirectional'] else ''})",
        axis=1
    )
    
    # Create comparison plots
    plt.figure(figsize=(12, 6))
    
    # Plot accuracies
    ax = plt.subplot(111)
    bar_width = 0.35
    index = np.arange(len(df))
    
    # Plot validation and test accuracy
    ax.bar(index, df['val_acc'], bar_width, label='Validation Accuracy')
    ax.bar(index + bar_width, df['test_acc'], bar_width, label='Test Accuracy')
    
    # Add labels
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison - Accuracy')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    plt.close()
    
    # Plot training curves for all models
    plt.figure(figsize=(14, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for file in df['file']:
        with open(file, 'r') as f:
            result = json.load(f)
            
        model_name = next(info['model_name'] for info in all_results if info['file'] == file)
        plt.plot(result['history']['train_loss'], label=f"{model_name} - Train")
        plt.plot(result['history']['val_loss'], label=f"{model_name} - Val", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 1, 2)
    for file in df['file']:
        with open(file, 'r') as f:
            result = json.load(f)
            
        model_name = next(info['model_name'] for info in all_results if info['file'] == file)
        plt.plot(result['history']['train_acc'], label=f"{model_name} - Train")
        plt.plot(result['history']['val_acc'], label=f"{model_name} - Val", linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves_comparison.png")
    plt.close()
    
    # Create a summary table and save as CSV
    summary = df[['model_name', 'hidden_size', 'num_layers', 'bidirectional', 
                 'val_acc', 'test_acc', 'epochs', 'final_train_loss', 'final_val_loss']]
    summary.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    print(f"Comparison visualizations saved to {output_dir}")
    
    # Return the best model
    best_model = df.iloc[0]
    print(f"\nBest model: {best_model['model_name']}")
    print(f"Test accuracy: {best_model['test_acc']:.4f}")
    print(f"Validation accuracy: {best_model['val_acc']:.4f}")
    
    return best_model


def main():
    parser = argparse.ArgumentParser(description='Visualize speech command classification results')
    
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations')
    parser.add_argument('--file', type=str, default=None,
                       help='Visualize a specific results file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple results')
    parser.add_argument('--top_n', type=int, default=None,
                       help='Only show top N models by test accuracy')
    
    args = parser.parse_args()
    
    if args.file:
        # Visualize a single file
        visualize_single_result(args.file, args.output_dir)
    elif args.compare:
        # Compare multiple files
        visualize_multiple_results(args.results_dir, args.output_dir, args.top_n)
    else:
        # Visualize all files individually
        result_files = glob.glob(f"{args.results_dir}/*.json")
        result_files = [f for f in result_files if not f.endswith('best_config.json')]
        
        if not result_files:
            print(f"No result files found in {args.results_dir}")
            return
        
        for file in result_files:
            visualize_single_result(file, args.output_dir)


if __name__ == "__main__":
    main()