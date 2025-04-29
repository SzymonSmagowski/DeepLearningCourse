#!/usr/bin/env python3
"""
Find the best performing model in each *_search directory based on test accuracy.
Create plots for model performance and confusion matrices.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def extract_model_name(filename):
    """Extract model name from filename between underscores."""
    parts = filename.split('_')
    for i in range(len(parts)):
        if parts[i].isdigit() and len(parts[i]) == 6:  # Timestamp format
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"

def plot_confusion_matrix(confusion_matrix, id2label, model_name, config, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    labels = [id2label[str(i)] for i in range(len(confusion_matrix))]
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_training_metrics(history, model_name, config, output_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_metrics_{model_name}.png'))
    plt.close()

def find_best_models():
    # Base directories
    results_dir = "results"
    result_diagram_dir = "result_diagram_dir"
    os.makedirs(result_diagram_dir, exist_ok=True)
    
    # Find all *_search directories
    search_dirs = []
    for item in os.listdir(results_dir):
        if item.endswith("_search") and os.path.isdir(os.path.join(results_dir, item)):
            search_dirs.append(item)
    
    # Store test accuracies for joint plot and best configs
    test_accuracies = []
    model_names = []
    best_configs = {}
    
    # Process each architecture directory
    for dir_name in sorted(search_dirs):
        print(f"\nProcessing {dir_name}:")
        dir_path = os.path.join(results_dir, dir_name)
        
        # Get all JSON files in this directory
        json_files = []
        for file in os.listdir(dir_path):
            if file.endswith(".json") and file != "best_config.json":
                json_files.append(os.path.join(dir_path, file))
        
        if not json_files:
            print("  No JSON files found")
            continue
        
        # Find the best model (highest test accuracy)
        best_model = None
        best_file = None
        best_accuracy = -1
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                test_acc = data.get("test_acc", 0)
                
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_model = data
                    best_file = os.path.basename(file_path)
            except:
                continue
        
        # Process the best model for this architecture
        if best_model:
            model_name = extract_model_name(best_file)
            config = best_model.get("config", {})
            
            # Store best config
            best_configs[model_name] = {
                "config": config,
                "test_accuracy": best_accuracy,
                "file": best_file
            }
            
            # Create plots for the best model
            plot_confusion_matrix(
                best_model.get("confusion_matrix", []),
                best_model.get("id2label", {}),
                model_name,
                config,
                result_diagram_dir
            )
            
            plot_training_metrics(
                best_model.get("history", {}),
                model_name,
                config,
                result_diagram_dir
            )
            
            # Store for joint plot
            test_accuracies.append(best_accuracy)
            model_names.append(model_name)
            
            # Print model info
            print(f"  Best model: {model_name}")
            print(f"  Test accuracy: {best_accuracy:.4f}")
            print(f"  File: {best_file}")
        else:
            print("  No valid models found")
    
    # Save best configurations to JSON file
    with open(os.path.join(result_diagram_dir, 'best_configs.json'), 'w') as f:
        json.dump(best_configs, f, indent=4)
    
    # Create joint test accuracy plot
    if test_accuracies:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, test_accuracies)
        
        # Add accuracy values to the plot
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title('Test Accuracy Comparison Across Best Models')
        plt.xlabel('Model Architecture')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(result_diagram_dir, 'test_accuracy_comparison.png'))
        plt.close()

if __name__ == "__main__":
    find_best_models()