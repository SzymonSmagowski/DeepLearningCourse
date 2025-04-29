import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def extract_noise_level(filename):
    # Extract noise level from filename like "20250429_072401_conformer_noise_0.10.json"
    return float(filename.split('_')[-1].replace('.json', ''))

def plot_confusion_matrix(confusion_matrix, labels, noise_level, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Noise Level: {noise_level})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, noise_level, save_path):
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Training History (Noise Level: {noise_level})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory for plots
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    # Dictionary to store test accuracies for comparison
    test_accuracies = {}
    
    # Process each JSON file
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            if filename == 'best_conformer.json':
                # Load best conformer data for later comparison
                with open(filename, 'r') as f:
                    best_data = json.load(f)
                test_accuracies['best_conformer'] = best_data['test_acc']
                continue
            
            # Extract noise level from filename
            noise_level = extract_noise_level(filename)
            
            # Load JSON data
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Store test accuracy for comparison
            test_accuracies[noise_level] = data['test_acc']
            
            # Create confusion matrix plot
            cm_path = output_dir / f'confusion_matrix_noise_{noise_level}.png'
            plot_confusion_matrix(
                data['confusion_matrix'],
                list(data['id2label'].values()),
                noise_level,
                cm_path
            )
            
            # Create training history plot
            history_path = output_dir / f'training_history_noise_{noise_level}.png'
            plot_training_history(data['history'], noise_level, history_path)
    
    # Plot test accuracy comparison
    plt.figure(figsize=(10, 6))
    noise_levels = sorted([k for k in test_accuracies.keys() if isinstance(k, float)])
    accuracies = [test_accuracies[noise] for noise in noise_levels]
    
    plt.plot(noise_levels, accuracies, 'o-', label='Noise Augmented Models')
    plt.axhline(y=test_accuracies['best_conformer'], color='r', linestyle='--', 
                label='Best Conformer')
    
    plt.title('Test Accuracy vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_accuracy_comparison.png')
    plt.close()

if __name__ == '__main__':
    main()
