#!/usr/bin/env python3
"""
Analyze how different parameter configurations affect model performance.
Create plots showing parameter sensitivity and their impact on test accuracy.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def extract_model_name(filename):
    """Extract model name from filename between underscores."""
    parts = filename.split('_')
    for i in range(len(parts)):
        if parts[i].isdigit() and len(parts[i]) == 6:  # Timestamp format
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown"

def plot_parameter_sensitivity(data, parameter_name, model_name, output_dir):
    """Plot how a specific parameter affects test accuracy."""
    plt.figure(figsize=(10, 6))
    
    # Sort data by parameter value
    sorted_data = sorted(data, key=lambda x: x['value'])
    values = [d['value'] for d in sorted_data]
    accuracies = [d['accuracy'] for d in sorted_data]
    
    plt.plot(values, accuracies, 'o-', markersize=8)
    plt.title(f'Parameter Sensitivity: {parameter_name} - {model_name}')
    plt.xlabel(parameter_name)
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    
    # Save the plot
    safe_param_name = parameter_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(output_dir, f'parameter_sensitivity_{model_name}_{safe_param_name}.png'))
    plt.close()

def analyze_parameter_sensitivity():
    # Base directories
    results_dir = "results"
    output_dir = "performance_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all *_search directories
    search_dirs = []
    for item in os.listdir(results_dir):
        if item.endswith("_search") and os.path.isdir(os.path.join(results_dir, item)):
            search_dirs.append(item)
    
    # Process each architecture directory
    for dir_name in sorted(search_dirs):
        print(f"\nAnalyzing {dir_name}:")
        dir_path = os.path.join(results_dir, dir_name)
        
        # Get all JSON files in this directory
        json_files = []
        for file in os.listdir(dir_path):
            if file.endswith(".json") and file != "best_config.json":
                json_files.append(os.path.join(dir_path, file))
        
        if not json_files:
            print("  No JSON files found")
            continue
        
        # Store all configurations and their accuracies
        all_configs = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_configs.append({
                        'config': data.get('config', {}),
                        'test_acc': data.get('test_acc', 0),
                        'file': os.path.basename(file_path)
                    })
            except:
                continue
        
        if not all_configs:
            print("  No valid configurations found")
            continue
        
        # Extract model name from the first file
        model_name = extract_model_name(all_configs[0]['file'])
        
        # Analyze each parameter
        config_keys = set()
        for config in all_configs:
            config_keys.update(config['config'].keys())
        
        # For each parameter, create a sensitivity plot
        for param in config_keys:
            param_data = []
            for config in all_configs:
                if param in config['config']:
                    param_data.append({
                        'value': config['config'][param],
                        'accuracy': config['test_acc']
                    })
            
            if len(set(d['value'] for d in param_data)) > 1:  # Only plot if parameter varies
                plot_parameter_sensitivity(param_data, param, model_name, output_dir)
                print(f"  Created sensitivity plot for parameter: {param}")

if __name__ == "__main__":
    analyze_parameter_sensitivity()