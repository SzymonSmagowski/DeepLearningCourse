# Enhanced Visualizations for Deep Learning Experiments

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os
from sklearn.metrics import confusion_matrix

# Create diagrams directory if it doesn't exist
diagrams_dir = 'diagrams'
if not os.path.exists(diagrams_dir):
    os.makedirs(diagrams_dir)
    print(f"Created directory: {diagrams_dir}")

# Function to save figures to the diagrams directory
def save_figure(fig, filename):
    filepath = os.path.join(diagrams_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")
    plt.close(fig)
    return filepath

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)

# Load the experiment results
try:
    with open('experiment_results_6_epochs.json', 'r') as f:
        results = json.load(f)
    print("Successfully loaded experiment results!")
except FileNotFoundError:
    print("Could not find experiment_results.json. Please run the experiments first.")
    # Create dummy results for notebook demonstration
    results = {
        "dummy_data": True,
        # Simplified dummy data for demonstration
    }

# Check if we're using dummy data
using_dummy_data = results.get("dummy_data", False)
if using_dummy_data:
    print("Using dummy data for demonstration purposes.")
    
# Function to create a more readable label from experiment name
def format_label(name):
    if "Architecture" in name:
        return name.split("_")[1].capitalize()
    elif "Training_Hyperparams" in name:
        params = results[name]["hyperparams"]
        return f"{params['optimizer'].upper()}, LR={params['learning_rate']}"
    elif "Regularization_Hyperparams" in name:
        wd = results[name]["hyperparams"]["weight_decay"]
        return f"Weight Decay={wd}"
    elif "Augmentation_" in name:
        return name.split("_")[1].replace("_", " ").title()
    elif "Reduced_Train_Size" in name:
        size = name.split("_")[-1]
        return f"Train Size: {size}"
    else:
        return name.replace("_", " ").title()

#-------------------------------------------------------------------------
# 1. Architecture Comparison - Enhanced Visualization
#-------------------------------------------------------------------------
print("\n1. Architecture Comparison")
arch_results = {k: v for k, v in results.items() if "Architecture_" in k}
if arch_results:
    arch_names = [format_label(name) for name in arch_results.keys()]
    arch_accs = [results[name]["test_accuracy"] for name in arch_results.keys()]
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(arch_names, arch_accs, color=sns.color_palette("viridis", len(arch_names)))
    ax.set_title("Architecture Comparison - Test Accuracy", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(0, 100)  # Set y-axis from 0 to 100 for percentage
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "1_architecture_comparison.png")
    plt.show()
    
    # Create detailed comparison table as a heatmap
    if len(arch_results) >= 2:
        # Extract accuracy and parameters for comparison
        data = []
        for name in arch_results.keys():
            arch_data = {"Architecture": format_label(name), 
                         "Test Accuracy": results[name]["test_accuracy"]}
            # Add more metrics if available
            if "train_accuracies" in results[name]:
                arch_data["Final Train Acc"] = results[name]["train_accuracies"][-1]
            if "valid_accuracies" in results[name]:
                arch_data["Final Valid Acc"] = results[name]["valid_accuracies"][-1]
            
            data.append(arch_data)
        
        # Convert to DataFrame for easy visualization
        df = pd.DataFrame(data)
        df = df.set_index("Architecture")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(arch_results) * 1.2))
        sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title("Architecture Performance Metrics")
        plt.tight_layout()
        save_figure(fig, "1_architecture_metrics_heatmap.png")
        plt.show()
    
    # Print the best architecture
    best_arch = max(arch_results.keys(), key=lambda k: results[k]["test_accuracy"])
    print(f"Best architecture: {format_label(best_arch)} with {results[best_arch]['test_accuracy']:.2f}% accuracy")
else:
    print("No architecture comparison results found.")

# Modified code for visualization - separating different architectures
# Add this to your visualization script

#-------------------------------------------------------------------------
# 2. Training Hyperparameters Analysis - Separated by Architecture
#-------------------------------------------------------------------------
print("\n2. Training Hyperparameters Analysis - Separated by Architecture")
training_results = {k: v for k, v in results.items() if "Training_Hyperparams_" in k}
if training_results:
    # Group results by architecture
    architectures = set()
    for name in training_results.keys():
        # Extract architecture name from experiment name
        if "efficientnet" in name.lower():
            architectures.add("efficientnet")
        elif "custom_cnn" in name.lower():
            architectures.add("custom_cnn")
        # Add other architecture checks if needed
    
    # Create a separate visualization for each architecture
    for arch in architectures:
        # Filter results for this architecture
        arch_results = {k: v for k, v in training_results.items() if arch in k.lower()}
        
        # Create a dataframe for easy visualization and analysis
        data = []
        for name in arch_results.keys():
            hyperparams = results[name]["hyperparams"]
            data.append({
                "Experiment": name,
                "Optimizer": hyperparams.get("optimizer", "N/A"),
                "Learning Rate": hyperparams.get("learning_rate", 0),
                "Scheduler": hyperparams.get("scheduler", "N/A"),
                "Weight Decay": hyperparams.get("weight_decay", 0),
                "Test Accuracy": results[name]["test_accuracy"]
            })
        
        df = pd.DataFrame(data)
        
        # 1. Bar chart comparing hyperparameter configurations for this architecture
        fig, ax = plt.subplots(figsize=(14, 8))
        experiments = [f"{exp.split('_')[-1]}" for exp in df["Experiment"]]  # Extract experiment number/id
        accuracies = df["Test Accuracy"]
        
        # Create a color map based on optimizer type
        optimizer_types = df["Optimizer"].unique()
        color_map = {opt: color for opt, color in zip(optimizer_types, 
                                                 sns.color_palette("husl", len(optimizer_types)))}
        colors = [color_map[opt] for opt in df["Optimizer"]]
        
        # Sort by accuracy for better visualization
        sorted_indices = np.argsort(accuracies)[::-1]  # Descending order
        experiments = [experiments[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Prepare more readable x-axis labels
        x_labels = []
        for i, idx in enumerate(sorted_indices):
            row = df.iloc[idx]
            x_labels.append(f"{row['Optimizer']}, LR={row['Learning Rate']}")
        
        bars = ax.bar(range(len(accuracies)), accuracies, color=colors)
        ax.set_xticks(range(len(accuracies)))
        ax.set_xticklabels(x_labels)
        
        ax.set_title(f"Training Hyperparameters Comparison - {arch.capitalize()}", fontsize=16)
        ax.set_ylabel("Test Accuracy (%)", fontsize=14)
        ax.set_ylim(max(0, min(accuracies) - 5), 100)  # Set reasonable y-axis limits
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(i, height + 0.5, f"{height:.1f}%", 
                   ha='center', va='bottom', fontweight='bold')
        
        # Add legend for optimizer types
        legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[opt]) for opt in optimizer_types]
        ax.legend(legend_handles, optimizer_types, title="Optimizer")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_figure(fig, f"2_hyperparameters_{arch}.png")
        plt.show()
        
        # 2. Learning rate analysis for this architecture
        if len(df["Learning Rate"].unique()) > 1:
            # Group by learning rate and calculate mean accuracy
            lr_analysis = df.groupby("Learning Rate")["Test Accuracy"].agg(['mean', 'min', 'max', 'count'])
            lr_analysis = lr_analysis.reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(lr_analysis["Learning Rate"], lr_analysis["mean"], 'o-', markersize=10, linewidth=2)
            
            # Add error bars if we have multiple data points per learning rate
            has_error_data = (lr_analysis["count"] > 1).any()
            if has_error_data:
                ax.errorbar(lr_analysis["Learning Rate"], lr_analysis["mean"], 
                           yerr=[lr_analysis["mean"] - lr_analysis["min"], 
                                 lr_analysis["max"] - lr_analysis["mean"]], 
                           fmt='o', capsize=5, alpha=0.5)
            
            ax.set_title(f"Effect of Learning Rate on Test Accuracy - {arch.capitalize()}", fontsize=16)
            ax.set_xlabel("Learning Rate", fontsize=14)
            ax.set_ylabel("Test Accuracy (%)", fontsize=14)
            ax.set_xscale('log')  # Log scale for learning rate
            
            # Add value labels
            for x, y in zip(lr_analysis["Learning Rate"], lr_analysis["mean"]):
                ax.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_figure(fig, f"2_learning_rate_{arch}.png")
            plt.show()
    
    # Print the best training hyperparameters
    best_train = max(training_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_params = results[best_train]["hyperparams"]
    best_arch = "EfficientNet" if "efficientnet" in best_train.lower() else "Custom CNN"
    print(f"Best training hyperparameters: {best_arch} with {best_params} - {results[best_train]['test_accuracy']:.2f}% accuracy")
else:
    print("No training hyperparameter results found.")


#-------------------------------------------------------------------------
# 3. Regularization Hyperparameters Analysis - Separated by Architecture
#-------------------------------------------------------------------------
print("\n3. Regularization Hyperparameters Analysis - Separated by Architecture")
reg_results = {k: v for k, v in results.items() if "Regularization_Hyperparams_" in k}
if reg_results:
    # Group results by architecture
    architectures = set()
    for name in reg_results.keys():
        # Extract architecture name from experiment name
        if "efficientnet" in name.lower():
            architectures.add("efficientnet")
        elif "custom_cnn" in name.lower():
            architectures.add("custom_cnn")
    
    # Create a separate visualization for each architecture
    for arch in architectures:
        # Filter results for this architecture
        arch_results = {k: v for k, v in reg_results.items() if arch in k.lower()}
        
        # Create a dataframe for the regularization experiments
        data = []
        for name in arch_results.keys():
            hyperparams = results[name]["hyperparams"]
            record = {
                "Experiment": name,
                "Weight Decay": hyperparams.get("weight_decay", 0),
                "Dropout Rate": hyperparams.get("dropout_rate", "N/A"),
                "Test Accuracy": results[name]["test_accuracy"]
            }
            
            # Add training/validation accuracy if available
            if "train_accuracies" in results[name]:
                record["Train Accuracy"] = results[name]["train_accuracies"][-1]
            if "valid_accuracies" in results[name]:
                record["Valid Accuracy"] = results[name]["valid_accuracies"][-1]
                
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # 1. Weight decay impact visualization
        if len(df["Weight Decay"].unique()) > 1:
            # Sort by weight decay for visualization
            df_wd = df.sort_values("Weight Decay")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_wd["Weight Decay"], df_wd["Test Accuracy"], 'o-', linewidth=2, markersize=10, color="purple")
            ax.set_title(f"Effect of Weight Decay on Test Accuracy - {arch.capitalize()}", fontsize=16)
            ax.set_xlabel("Weight Decay", fontsize=14)
            ax.set_ylabel("Test Accuracy (%)", fontsize=14)
            
            # Handle zero values for log scale
            if 0 in df_wd["Weight Decay"].values:
                ax.set_xscale('symlog', linthresh=1e-5)  # Use symmetric log scale
            else:
                ax.set_xscale('log')  # Use standard log scale
            
            # Add value labels
            for x, y in zip(df_wd["Weight Decay"], df_wd["Test Accuracy"]):
                ax.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_figure(fig, f"3_weight_decay_{arch}.png")
            plt.show()

        # 2. Dropout rate impact visualization
        if "Dropout Rate" in df.columns and df["Dropout Rate"].nunique() > 1:
            # Convert dropout rate to numeric if it's not already
            df["Dropout Rate"] = pd.to_numeric(df["Dropout Rate"], errors='coerce')
            
            # Remove N/A values
            df_dropout = df.dropna(subset=["Dropout Rate"]).sort_values("Dropout Rate")
            
            if len(df_dropout) > 1:  # At least 2 different dropout rates
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_dropout["Dropout Rate"], df_dropout["Test Accuracy"], 
                       'o-', linewidth=2, markersize=10, color="teal")
                ax.set_title(f"Effect of Dropout Rate on Test Accuracy - {arch.capitalize()}", fontsize=16)
                ax.set_xlabel("Dropout Rate", fontsize=14)
                ax.set_ylabel("Test Accuracy (%)", fontsize=14)
                
                # Add value labels
                for x, y in zip(df_dropout["Dropout Rate"], df_dropout["Test Accuracy"]):
                    ax.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_figure(fig, f"3_dropout_rate_{arch}.png")
    

    # Print the best regularization hyperparameters
    best_reg = max(reg_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_wd = results[best_reg]["hyperparams"]["weight_decay"]
    best_arch = "EfficientNet" if "efficientnet" in best_reg.lower() else "Custom CNN"
    print(f"Best weight decay: {best_arch} with {best_wd} - {results[best_reg]['test_accuracy']:.2f}% accuracy")
else:
    print("No regularization hyperparameter results found.")


#-------------------------------------------------------------------------
# 4. Data Augmentation Analysis - Separated by Architecture  
#-------------------------------------------------------------------------
print("\n4. Data Augmentation Analysis - Separated by Architecture")
aug_results = {k: v for k, v in results.items() if "Augmentation_" in k}
if aug_results:
    # Organize results by model architecture
    architectures = set()
    for k in aug_results.keys():
        parts = k.split('_')
        if len(parts) > 2:  # If format is Augmentation_architecture_technique
            architectures.add(parts[1])
    
    # Create a separate visualization for each architecture
    for arch in architectures:
        # Filter results for this architecture
        arch_results = {k: v for k, v in aug_results.items() if f"_{arch}_" in k}
        
        # Create a dataframe for this architecture
        data = []
        for k, v in arch_results.items():
            parts = k.split('_')
            if len(parts) > 2:
                technique = '_'.join(parts[2:])  # Join remaining parts for technique name
            else:
                technique = parts[1]
                
            data.append({
                "Technique": technique.replace('_', ' ').title(),
                "Test Accuracy": v["test_accuracy"]
            })
        
        df = pd.DataFrame(data)
        
        # Sort by accuracy
        df = df.sort_values("Test Accuracy", ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define a color palette that distinguishes standard vs. advanced techniques
        is_advanced = ["cutout" in t.lower() or "mixup" in t.lower() for t in df["Technique"]]
        colors = ["#3498db" if adv else "#e74c3c" for adv in is_advanced]
        
        bars = ax.bar(df["Technique"], df["Test Accuracy"], color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
        
        # Add legend for advanced vs standard techniques
        handles = [plt.Rectangle((0,0),1,1, color="#e74c3c"), 
                   plt.Rectangle((0,0),1,1, color="#3498db")]
        ax.legend(handles, ["Standard Augmentation", "Advanced Augmentation"])
        
        ax.set_title(f"Data Augmentation Techniques - {arch.capitalize()}", fontsize=16)
        ax.set_ylabel("Test Accuracy (%)", fontsize=14)
        ax.set_ylim(max(0, df["Test Accuracy"].min() - 5), 100)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_figure(fig, f"4_augmentation_{arch}.png")
        plt.show()
        
        # 2. Performance gain visualization
        # Find the baseline (no augmentation) accuracy
        baseline_key = next((k for k in arch_results.keys() if "baseline" in k.lower()), None)
        
        if baseline_key:
            baseline_acc = results[baseline_key]["test_accuracy"]
            
            # Calculate improvements
            improvements = []
            for k, v in arch_results.items():
                if k != baseline_key:  # Skip the baseline itself
                    # Extract technique name
                    parts = k.split('_')
                    if len(parts) > 2:
                        technique = '_'.join(parts[2:])
                    else:
                        technique = parts[1]
                    
                    technique = technique.replace('_', ' ').title()
                    improvements.append({
                        "Technique": technique,
                        "Improvement": v["test_accuracy"] - baseline_acc
                    })
            
            # Sort by improvement
            improvements = sorted(improvements, key=lambda x: x["Improvement"], reverse=True)
            
            # Create bar chart of improvements
            fig, ax = plt.subplots(figsize=(12, 6))
            techniques = [imp["Technique"] for imp in improvements]
            gains = [imp["Improvement"] for imp in improvements]
            
            # Use color based on improvement (green for positive, red for negative)
            colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gains]
            
            bars = ax.bar(techniques, gains, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                color = "black"
                va = "bottom"
                offset = 0.1
                if height < 0:
                    va = "top"
                    offset = -0.5
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f"{height:+.1f}%", ha='center', va=va, fontweight='bold', color=color)
            
            ax.set_title(f"Performance Gain from Data Augmentation - {arch.capitalize()}", fontsize=16)
            ax.set_ylabel("Accuracy Improvement (%)", fontsize=14)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            save_figure(fig, f"4_augmentation_gains_{arch}.png")
            plt.show()
    
    # Print the best augmentation technique with architecture
    best_aug = max(aug_results.keys(), key=lambda k: results[k]["test_accuracy"])
    parts = best_aug.split('_')
    if len(parts) > 2:
        best_arch = parts[1]
        best_tech = '_'.join(parts[2:]).replace('_', ' ').title()
    else:
        best_arch = "unknown"
        best_tech = parts[1].replace('_', ' ').title()
    
    print(f"Best augmentation technique: {best_tech} on {best_arch.capitalize()} with {results[best_aug]['test_accuracy']:.2f}% accuracy")
else:
    print("No data augmentation results found.")

#-------------------------------------------------------------------------
# 5. Few-shot Learning Analysis
#-------------------------------------------------------------------------
print("\n5. Few-shot Learning Analysis")
few_shot_results = {k: v for k, v in results.items() if "few_shot" in k.lower()}
if few_shot_results:
    # Group results by architecture
    architectures = []
    accuracies = []
    std_deviations = []
    
    for name, result in few_shot_results.items():
        arch = "efficientnet" if "efficientnet" in name.lower() else "custom_cnn"
        architectures.append(arch.capitalize())
        accuracies.append(result["accuracy"])
        std_deviations.append(result.get("std", 0))  # Get std if available
    
    # Create bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color palette
    colors = sns.color_palette("viridis", len(architectures))
    
    # Create bar chart
    bars = ax.bar(architectures, accuracies, yerr=std_deviations, 
                 capsize=10, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, acc, std in zip(bars, accuracies, std_deviations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{acc:.1f}%\n±{std:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Few-shot Learning Performance by Architecture (5-way, 5-shot)", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(0, max(accuracies) + max(std_deviations) + 5)
    
    # Add a horizontal line for random chance (20% for 5-way classification)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1)
    ax.text(ax.get_xlim()[0], 21, "Random Chance (20%)", color='red', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "5_few_shot_learning.png")
    
    # Create detailed visualization showing few-shot vs regular learning
    # Find regular accuracy results for these architectures
    regular_accuracies = {}
    for arch in set(architectures):
        arch_lower = arch.lower()
        for key, value in results.items():
            if f"Architecture_{arch_lower}" in key and "test_accuracy" in value:
                regular_accuracies[arch_lower] = value["test_accuracy"]
                break
    
    # Create comparison data
    if regular_accuracies:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.35
        index = np.arange(len(set(architectures)))
        
        # Organize data by unique architectures
        unique_archs = list(set(architectures))
        few_shot_accs = []
        few_shot_stds = []
        regular_accs = []
        
        for arch in unique_archs:
            arch_lower = arch.lower()
            fs_idx = architectures.index(arch)
            few_shot_accs.append(accuracies[fs_idx])
            few_shot_stds.append(std_deviations[fs_idx])
            regular_accs.append(regular_accuracies.get(arch_lower, 0))
        
        # Create grouped bar chart
        bar1 = ax.bar(index, regular_accs, bar_width, label='Regular Learning', 
                     color='#3498db', alpha=0.8)
        bar2 = ax.bar(index + bar_width, few_shot_accs, bar_width, yerr=few_shot_stds,
                     capsize=10, label='Few-shot Learning (5-way, 5-shot)', 
                     color='#e74c3c', alpha=0.8)
        
        # Add value labels
        for bar, acc in zip(bar1, regular_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold', color='#3498db')
        
        for bar, acc, std in zip(bar2, few_shot_accs, few_shot_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{acc:.1f}%\n±{std:.1f}%", ha='center', va='bottom', fontweight='bold', color='#e74c3c')
        
        # Add labels and legend
        ax.set_xlabel('Architecture', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('Regular Learning vs. Few-shot Learning Comparison', fontsize=16)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(unique_archs)
        ax.legend(fontsize=12)
        
        # Add a note explaining few-shot learning
        ax.text(0.5, -0.15, 
                "Note: Few-shot learning tests the model's ability to classify new examples\n"
                "using only 5 examples per class (5-shot) across 5 different classes (5-way).",
                transform=ax.transAxes, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        save_figure(fig, "5_few_shot_vs_regular.png")
    
    # Create adaptation curve if we have multiple shot values
    shot_levels = set()
    shot_results = {}
    
    for name, result in few_shot_results.items():
        # Try to extract shot information if available
        if "shot" in name:
            parts = name.split("_")
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0 and parts[i-1].lower() == "shot":
                    shots = int(part)
                    shot_levels.add(shots)
                    
                    arch = "efficientnet" if "efficientnet" in name.lower() else "custom_cnn"
                    if arch not in shot_results:
                        shot_results[arch] = {}
                    
                    shot_results[arch][shots] = {
                        "accuracy": result["accuracy"],
                        "std": result.get("std", 0)
                    }
    
    # If we have multiple shot levels, create adaptation curve
    if len(shot_levels) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for arch, shots_data in shot_results.items():
            shot_nums = sorted(shots_data.keys())
            shot_accs = [shots_data[s]["accuracy"] for s in shot_nums]
            shot_stds = [shots_data[s]["std"] for s in shot_nums]
            
            ax.errorbar(shot_nums, shot_accs, yerr=shot_stds, marker='o', 
                       markersize=8, capsize=6, linewidth=2, label=arch.capitalize())
            
            # Add value labels
            for x, y, std in zip(shot_nums, shot_accs, shot_stds):
                ax.text(x, y + std + 1, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
        
        ax.set_title("Few-shot Learning Adaptation Curve (5-way classification)", fontsize=16)
        ax.set_xlabel("Number of Shots (Examples per Class)", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.legend(fontsize=12)
        
        # Use integer x-ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a horizontal line for random chance (20% for 5-way classification)
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1)
        ax.text(ax.get_xlim()[0], 21, "Random Chance (20%)", color='red', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, "5_few_shot_adaptation_curve.png")
    
    # Print few-shot learning analysis
    print(f"Few-shot learning analysis by architecture:")
    for i, (arch, acc, std) in enumerate(zip(architectures, accuracies, std_deviations)):
        print(f"  {arch}: {acc:.2f}% ± {std:.2f}%")
    
    best_arch_idx = np.argmax(accuracies)
    print(f"\nBest few-shot learning architecture: {architectures[best_arch_idx]} with {accuracies[best_arch_idx]:.2f}% ± {std_deviations[best_arch_idx]:.2f}% accuracy")
    
    print(f"\nFew-shot learning allows the model to recognize new categories with limited examples.")
    print(f"This is especially valuable in real-world scenarios where collecting large datasets is impractical.")
else:
    print("No few-shot learning results found.")


#-------------------------------------------------------------------------
# 6. Reduced Training Set Size Analysis - Separated by Architecture
#-------------------------------------------------------------------------
print("\n6. Reduced Training Set Size Analysis - Separated by Architecture")
reduced_results = {k: v for k, v in results.items() if "Reduced_Train_Size_" in k}
if reduced_results:
    # Group by architecture
    architectures = set()
    for name in reduced_results.keys():
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part in ["efficientnet", "custom_cnn"]:
                architectures.add(part)
                break
    
    # Create a visualization comparing both architectures on the same plot
    if len(architectures) > 1:
        # Prepare data for all architectures
        arch_data = {}
        for arch in architectures:
            # Filter results for this architecture
            arch_results = {k: v for k, v in reduced_results.items() if arch in k.lower()}
            
            # Extract size and accuracy data
            sizes = []
            accs = []
            for name, result in arch_results.items():
                # Extract size (usually the last part of the name)
                size = int(name.split('_')[-1])
                sizes.append(size)
                accs.append(result["test_accuracy"])
            
            # Get the baseline accuracy from full dataset for this architecture
            baseline_acc = None
            for key, value in results.items():
                if f"Architecture_{arch}" in key and "test_accuracy" in value:
                    baseline_acc = value["test_accuracy"]
                    break
            
            # Add the full dataset size if we have the baseline accuracy
            full_train_size = None
            for key in results.keys():
                if "TRAIN_SUBSET_SIZE" in str(key):
                    full_train_size = results[key]
                    break
            
            if not full_train_size:
                # Estimate full size as 2x the largest reduced size
                full_train_size = max(sizes) * 2
            
            # Add baseline results if available
            if baseline_acc is not None:
                sizes.append(full_train_size)
                accs.append(baseline_acc)
            
            # Store data for this architecture
            arch_data[arch] = {
                "sizes": sizes,
                "accuracies": accs
            }
        
        # Create comparison plot for all architectures
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define color palette for architectures
        colors = {
            "efficientnet": "#2980b9",  # Blue
            "custom_cnn": "#e74c3c"     # Red
        }
        
        # Plot data for each architecture
        for arch, data in arch_data.items():
            # Sort by size
            sorted_indices = np.argsort(data["sizes"])
            sizes = [data["sizes"][i] for i in sorted_indices]
            accs = [data["accuracies"][i] for i in sorted_indices]
            
            # Plot
            ax.plot(sizes, accs, 'o-', 
                   linewidth=2, markersize=10, 
                   label=arch.capitalize(),
                   color=colors.get(arch, "green"))
            
            # Add value labels
            for x, y in zip(sizes, accs):
                ax.text(x, y + 0.5, f"{y:.1f}%", 
                       ha='center', va='bottom', 
                       fontweight='bold',
                       color=colors.get(arch, "green"))
        
        ax.set_title(f"Effect of Training Dataset Size on Accuracy - Architecture Comparison", fontsize=16)
        ax.set_xlabel("Training Set Size", fontsize=14)
        ax.set_ylabel("Test Accuracy (%)", fontsize=14)
        ax.legend(fontsize=12)
        
        # Use integer ticks for x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, "6_training_size_comparison.png")
    
    # Create separate visualizations for each architecture (as in original code)
    for arch in architectures:
        # Filter results for this architecture
        arch_results = {k: v for k, v in reduced_results.items() if arch in k.lower()}
        
        # Extract data
        data = []
        for name, result in arch_results.items():
            # Extract size (usually the last part of the name)
            size = int(name.split('_')[-1])
                
            data.append({
                "Size": size,
                "Test Accuracy": result["test_accuracy"]
            })
        
        df = pd.DataFrame(data)
        
        # Get the baseline accuracy from full dataset for this architecture
        baseline_acc = None
        for key, value in results.items():
            if f"Architecture_{arch}" in key and "test_accuracy" in value:
                baseline_acc = value["test_accuracy"]
                break
        
        # Add the full dataset size if we have the baseline accuracy
        full_train_size = None
        for key in results.keys():
            if "TRAIN_SUBSET_SIZE" in str(key):
                full_train_size = results[key]
                break
        
        if not full_train_size:
            # Estimate full size as 2x the largest reduced size
            full_train_size = max(df["Size"]) * 2

        # Add baseline results to dataframe if available
        if baseline_acc is not None:
            df = pd.concat([df, pd.DataFrame([{
                "Size": full_train_size,
                "Test Accuracy": baseline_acc
            }])], ignore_index=True)
        
        # Sort by size
        df = df.sort_values("Size")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(df["Size"], df["Test Accuracy"], 'o-', 
               linewidth=2, markersize=10, label=arch.capitalize())
        
        # Add value labels
        for x, y in zip(df["Size"], df["Test Accuracy"]):
            ax.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Effect of Training Dataset Size on Accuracy - {arch.capitalize()}", fontsize=16)
        ax.set_xlabel("Training Set Size", fontsize=14)
        ax.set_ylabel("Test Accuracy (%)", fontsize=14)
        
        # Use integer ticks for x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, f"6_training_size_impact_{arch}.png")
        
        # Calculate data efficiency for this architecture
        if len(df) > 1:
            # Get the maximum accuracy for this architecture
            max_acc = df["Test Accuracy"].max()
            
            # Calculate relative accuracy (percentage of max accuracy)
            df["Relative Accuracy"] = df["Test Accuracy"] / max_acc * 100
            
            # Create efficiency plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot relative accuracy vs dataset size
            ax.plot(df["Size"], df["Relative Accuracy"], 'o-', 
                   linewidth=2, markersize=10)
            
            # Add value labels
            for x, y in zip(df["Size"], df["Relative Accuracy"]):
                ax.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f"Data Efficiency: Relative Performance vs Dataset Size - {arch.capitalize()}", fontsize=16)
            ax.set_xlabel("Training Set Size", fontsize=14)
            ax.set_ylabel("Relative Accuracy (% of Max)", fontsize=14)
            
            # Add reference lines
            ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0], 91, "90% of max accuracy", va='bottom', alpha=0.6)
            
            ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0], 96, "95% of max accuracy", va='bottom', alpha=0.6)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_figure(fig, f"6_data_efficiency_{arch}.png")
    
    # Print training set size analysis by architecture
    print(f"Training dataset size analysis by architecture:")
    for arch in architectures:
        arch_results = {k: v for k, v in reduced_results.items() if arch in k.lower()}
        print(f"\n  Architecture: {arch.capitalize()}")
        
        # Extract sizes and accuracies
        sizes = []
        accs = []
        for name, result in arch_results.items():
            size = int(name.split('_')[-1])
            sizes.append(size)
            accs.append(result["test_accuracy"])
        
        # Sort by size
        sorted_indices = np.argsort(sizes)
        sizes = [sizes[i] for i in sorted_indices]
        accs = [accs[i] for i in sorted_indices]
        
        for size, acc in zip(sizes, accs):
            print(f"    Size {size}: {acc:.2f}% accuracy")
else:
    print("No reduced training set size results found.")


#-------------------------------------------------------------------------
# 7. Ensemble Methods Analysis - Separated by Architecture
#-------------------------------------------------------------------------
print("\n7. Ensemble Methods Analysis - Separated by Architecture")
# Find all ensemble results
ensemble_results = {}
for k, v in results.items():
    if "ensemble" in k.lower():
        ensemble_results[k] = v
    elif isinstance(v, dict) and "hard_voting" in v and "soft_voting" in v:
        ensemble_results[k] = v

if ensemble_results:
    # Group by architecture
    architectures = set()
    for key in ensemble_results.keys():
        if "ensemble_" in key.lower():
            # Extract architecture from key
            arch = key.split('_')[-1]
            if arch in ["efficientnet", "custom_cnn"]:
                architectures.add(arch)
    
    # If we couldn't extract architectures from the keys, try to find them in the data
    if not architectures:
        for key, value in results.items():
            if "Architecture_" in key:
                arch = key.split('_')[1]
                if arch in ["efficientnet", "custom_cnn"]:
                    architectures.add(arch)
    
    # Create a separate visualization for each architecture
    for arch in architectures:
        # Find the ensemble results for this architecture
        arch_ensemble = None
        for key, value in ensemble_results.items():
            if arch in key.lower():
                arch_ensemble = value
                break
        
        if arch_ensemble is None:
            continue
        
        # Get the best single model accuracy for this architecture
        best_single_acc = 0
        for key, value in results.items():
            if isinstance(value, dict) and "test_accuracy" in value:
                if arch in key and value["test_accuracy"] > best_single_acc:
                    best_single_acc = value["test_accuracy"]
        
        # Create comparison data
        methods = ["Hard Voting", "Soft Voting", "Best Single Model"]
        accuracies = [
            arch_ensemble["hard_voting"]["accuracy"],
            arch_ensemble["soft_voting"]["accuracy"],
            best_single_acc
        ]
        
        # Create the comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(methods, accuracies, color=["#3498db", "#2ecc71", "#e74c3c"])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Ensemble Methods vs. Single Model - {arch.capitalize()}", fontsize=16)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.set_ylim(max(0, min(accuracies) - 5), 100)
        
        plt.tight_layout()
        save_figure(fig, f"7_ensemble_methods_{arch}.png")
        plt.show()
        
        # Calculate improvement over single model
        hard_improvement = arch_ensemble["hard_voting"]["accuracy"] - best_single_acc
        soft_improvement = arch_ensemble["soft_voting"]["accuracy"] - best_single_acc
        
        # Create improvement plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        improvement_methods = ["Hard Voting", "Soft Voting"]
        improvements = [hard_improvement, soft_improvement]
        
        # Use color based on improvement
        colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]
        
        bars = ax.bar(improvement_methods, improvements, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            va = "bottom"
            offset = 0.1
            if height < 0:
                va = "top"
                offset = -0.5
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f"{height:+.1f}%", ha='center', va=va, fontweight='bold')
        
        ax.set_title(f"Improvement from Ensemble Methods - {arch.capitalize()}", fontsize=16)
        ax.set_ylabel("Accuracy Improvement (%)", fontsize=14)
        
        # Add a reference line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, f"7_ensemble_improvement_{arch}.png")
        plt.show()
        
        # Print ensemble results
        print(f"Ensemble methods for {arch.capitalize()}:")
        print(f"  Hard voting: {arch_ensemble['hard_voting']['accuracy']:.2f}%")
        print(f"  Soft voting: {arch_ensemble['soft_voting']['accuracy']:.2f}%")
        print(f"  Best single model: {best_single_acc:.2f}%")
        
        # Calculate improvement
        best_ensemble = max(arch_ensemble['hard_voting']['accuracy'], arch_ensemble['soft_voting']['accuracy'])
        improvement = best_ensemble - best_single_acc
        print(f"  Ensemble improvement over best single model: {improvement:+.2f}%")
else:
    print("No ensemble method results found.")

#-------------------------------------------------------------------------
# 8. Confusion Matrix Analysis - Enhanced Visualization
#-------------------------------------------------------------------------
print("\n8. Confusion Matrix Analysis")
# Find results with confusion matrices
cm_results = {}
for k, v in results.items():
    if isinstance(v, dict) and "confusion_matrix" in v:
        cm_results[k] = v

if cm_results:
    # Find the best model(s) to visualize confusion matrices
    best_models = sorted(cm_results.keys(), 
                        key=lambda k: results[k]["test_accuracy"], 
                        reverse=True)[:min(3, len(cm_results))]
    
    # Extract class names if available
    class_names = None
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and all(isinstance(item, str) for item in v):
            # This might be the class names list
            if 5 <= len(v) <= 20:  # Reasonable number of classes
                class_names = v
                break
    
    # If we didn't find class names, use numbers
    if class_names is None:
        # Get the number of classes from the first confusion matrix
        first_cm = results[list(cm_results.keys())[0]]["confusion_matrix"]
        n_classes = len(first_cm)
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Visualize confusion matrices for best models
    for model_name in best_models:
        model_results = results[model_name]
        cm = np.array(model_results["confusion_matrix"])  # Convert list to numpy array
        acc = model_results["test_accuracy"]
        
        # Create a normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts matrix
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title(f"{format_label(model_name)} - Confusion Matrix (Counts)\nTest Accuracy: {acc:.2f}%", fontsize=16)
        
        # Add colorbar
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Count', rotation=270, labelpad=15)
        
        # Add labels to counts matrix
        thresh1 = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh1 else "black")
        
        # Normalized matrix
        im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax2.set_title(f"{format_label(model_name)} - Normalized Confusion Matrix\nTest Accuracy: {acc:.2f}%", fontsize=16)
        
        # Add colorbar
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Probability', rotation=270, labelpad=15)
        
        # Add labels to normalized matrix
        thresh2 = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax2.text(j, i, format(cm_normalized[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh2 else "black")
        
        # Add axis labels for both plots
        for ax in [ax1, ax2]:
            ax.set_xlabel('Predicted label', fontsize=14)
            ax.set_ylabel('True label', fontsize=14)
            
            # Add ticks
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            
            # Add tick labels
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            
            # Rotate tick labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()
        save_figure(fig, f"8_confusion_matrix_{model_name}.png")
        plt.show()
        
        # Calculate and visualize per-class metrics
        if "all_preds" in model_results and "all_labels" in model_results:
            # Extract predictions and true labels
            preds = model_results["all_preds"]
            labels = model_results["all_labels"]
            
            # Calculate per-class accuracy, precision, recall
            class_metrics = []
            for i, class_name in enumerate(class_names):
                # For each class, calculate:
                # - Accuracy: (TP + TN) / (TP + TN + FP + FN)
                # - Precision: TP / (TP + FP)
                # - Recall: TP / (TP + FN)
                
                # For binary calculations
                true_positive = np.sum((np.array(labels) == i) & (np.array(preds) == i))
                false_positive = np.sum((np.array(labels) != i) & (np.array(preds) == i))
                false_negative = np.sum((np.array(labels) == i) & (np.array(preds) != i))
                true_negative = np.sum((np.array(labels) != i) & (np.array(preds) != i))
                
                accuracy = (true_positive + true_negative) / len(preds)
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics.append({
                    "Class": class_name,
                    "Accuracy": accuracy * 100,
                    "Precision": precision * 100,
                    "Recall": recall * 100,
                    "F1 Score": f1 * 100
                })
            
            # Create a dataframe for easy visualization
            metrics_df = pd.DataFrame(class_metrics)
            
            # Plot per-class metrics
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Reshape data for grouped bar chart
            plot_data = []
            for _, row in metrics_df.iterrows():
                for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                    plot_data.append({
                        "Class": row["Class"],
                        "Metric": metric,
                        "Value": row[metric]
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create grouped bar chart
            sns.barplot(x="Class", y="Value", hue="Metric", data=plot_df, ax=ax)
            
            ax.set_title(f"{format_label(model_name)} - Per-Class Performance Metrics", fontsize=16)
            ax.set_ylabel("Percentage (%)", fontsize=14)
            ax.set_ylim(0, 100)
            
            # Rotate x-labels if many classes
            if len(class_names) > 5:
                plt.xticks(rotation=45, ha="right")
            
            plt.tight_layout()
            save_figure(fig, f"8_class_metrics_{model_name}.png")
            plt.show()
            
            # Identify most confused classes
            most_confused = []
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    if i != j:  # Skip diagonal (correctly classified)
                        most_confused.append({
                            "True": class_names[i],
                            "Predicted": class_names[j],
                            "Count": cm[i, j],
                            "Rate": cm[i, j] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0
                        })
            
            # Sort by count and get top confusions
            most_confused.sort(key=lambda x: x["Count"], reverse=True)
            top_confused = most_confused[:min(10, len(most_confused))]
            
            if top_confused:
                # Visualize top confused classes
                fig, ax = plt.subplots(figsize=(14, 8))
                
                confusions = [f"{c['True']} → {c['Predicted']}" for c in top_confused]
                counts = [c["Count"] for c in top_confused]
                rates = [c["Rate"] * 100 for c in top_confused]
                
                # Create a twin axis for showing both count and percentage
                ax2 = ax.twinx()
                
                # Plot counts as bars
                bars = ax.bar(confusions, counts, color=plt.cm.Reds(np.array(rates)/100))
                ax.set_ylabel("Count", color='blue', fontsize=14)
                ax.tick_params(axis='y', colors='blue')
                
                # Plot rates as a line
                line = ax2.plot(confusions, rates, 'o-', color='darkred', linewidth=2, markersize=8)
                ax2.set_ylabel("Confusion Rate (%)", color='darkred', fontsize=14)
                ax2.tick_params(axis='y', colors='darkred')
                
                # Add value labels
                for i, (bar, rate) in enumerate(zip(bars, rates)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f"{height}", ha='center', va='bottom', fontweight='bold', color='blue')
                    ax2.text(i, rate + 1, f"{rate:.1f}%", ha='center', va='bottom', 
                            fontweight='bold', color='darkred')
                
                ax.set_title(f"{format_label(model_name)} - Top Class Confusions", fontsize=16)
                
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                save_figure(fig, f"8_top_confusions_{model_name}.png")
                plt.show()
    
    # Print confusion matrix analysis
    print(f"Confusion matrix analysis completed for top {len(best_models)} models.")
    print(f"Detailed visualizations saved to the '{diagrams_dir}' directory.")
else:
    print("No confusion matrix data found in the results.")

#-------------------------------------------------------------------------
# 9. Comprehensive Summary Visualization
#-------------------------------------------------------------------------
print("\n9. Comprehensive Comparison of All Methods")

# Collect the best result from each category
best_results = {}

# Best architecture
arch_results = {k: v for k, v in results.items() if "Architecture_" in k}
if arch_results:
    best_arch = max(arch_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_results["Architecture"] = {"name": format_label(best_arch), "accuracy": results[best_arch]["test_accuracy"]}

# Best training hyperparameters
training_results = {k: v for k, v in results.items() if "Training_Hyperparams_" in k}
if training_results:
    best_train = max(training_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_results["Training HP"] = {"name": format_label(best_train), "accuracy": results[best_train]["test_accuracy"]}

# Best regularization
reg_results = {k: v for k, v in results.items() if "Regularization_Hyperparams_" in k}
if reg_results:
    best_reg = max(reg_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_results["Regularization"] = {"name": format_label(best_reg), "accuracy": results[best_reg]["test_accuracy"]}

# Best augmentation
aug_results = {k: v for k, v in results.items() if "Augmentation_" in k}
if aug_results:
    best_aug = max(aug_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_results["Augmentation"] = {"name": format_label(best_aug), "accuracy": results[best_aug]["test_accuracy"]}

# Few-shot learning
few_shot_results = {k: v for k, v in results.items() if "few_shot" in k.lower()}
if few_shot_results:
    best_fs = max(few_shot_results.keys(), key=lambda k: results[k]["accuracy"])
    best_results["Few-shot"] = {"name": format_label(best_fs), "accuracy": results[best_fs]["accuracy"]}

# Best ensemble
ensemble_results = {}
for k, v in results.items():
    if isinstance(v, dict) and "hard_voting" in v and "soft_voting" in v:
        ensemble_results[k] = v

if ensemble_results:
    best_ensemble = max(ensemble_results.keys(), 
                        key=lambda k: max(results[k]["hard_voting"]["accuracy"], 
                                         results[k]["soft_voting"]["accuracy"]))
    
    # Determine which voting method was better
    if results[best_ensemble]["soft_voting"]["accuracy"] > results[best_ensemble]["hard_voting"]["accuracy"]:
        method = "Soft Voting"
        acc = results[best_ensemble]["soft_voting"]["accuracy"]
    else:
        method = "Hard Voting"
        acc = results[best_ensemble]["hard_voting"]["accuracy"]
        
    best_results["Ensemble"] = {"name": method, "accuracy": acc}

# Create a comprehensive comparison chart
if best_results:
    categories = list(best_results.keys())
    methods = [best_results[cat]["name"] for cat in categories]
    accuracies = [best_results[cat]["accuracy"] for cat in categories]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]  # Descending order
    categories = [categories[i] for i in sorted_indices]
    methods = [methods[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Create a colorful bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.bar(categories, accuracies, color=sns.color_palette("viridis", len(categories)))
    
    # Add method labels above the bars
    for i, (bar, method) in enumerate(zip(bars, methods)):
        ax.text(bar.get_x() + bar.get_width()/2., 30, 
                method, ha='center', va='bottom', 
                rotation=90, color='white', fontweight='bold')
    
    # Add value labels at the top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Comparison of Best Results Across All Experiment Types", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(0, max(accuracies) + 5)

    plt.tight_layout()
    save_figure(fig, "9_comprehensive_comparison.png")
    plt.show()
    
    # Print overall best method
    best_category = categories[0]
    best_method = methods[0]
    best_acc = accuracies[0]
    print(f"Overall best performance: {best_category} ({best_method}) with {best_acc:.2f}% accuracy")

#-------------------------------------------------------------------------
# 10. Learning Curves and Training Progress Analysis
#-------------------------------------------------------------------------
print("\n10. Training Progress Analysis")

# Find results with training curves
training_curves_results = {}
for k, v in results.items():
    if isinstance(v, dict) and all(curve in v for curve in ["train_losses", "valid_losses", "train_accuracies", "valid_accuracies"]):
        training_curves_results[k] = v

if training_curves_results:
    # Find the best model(s) to visualize learning curves
    best_models = sorted(training_curves_results.keys(), 
                        key=lambda k: results[k].get("test_accuracy", 0), 
                        reverse=True)[:min(3, len(training_curves_results))]
    
    for model_name in best_models:
        model_results = results[model_name]
        
        # Extract learning curves
        train_losses = model_results["train_losses"]
        valid_losses = model_results["valid_losses"]
        train_accs = model_results["train_accuracies"]
        valid_accs = model_results["valid_accuracies"]
        test_acc = model_results.get("test_accuracy", None)
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Create learning curves plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Loss subplot
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.plot(epochs, valid_losses, 'r-', linewidth=2, label='Validation Loss')
        ax1.set_title(f'Learning Curves for {format_label(model_name)}', fontsize=16)
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add final loss values to the plot
        ax1.text(epochs[-1] + 0.1, train_losses[-1], f"{train_losses[-1]:.4f}", 
                va='center', ha='left', color='blue', fontweight='bold')
        ax1.text(epochs[-1] + 0.1, valid_losses[-1], f"{valid_losses[-1]:.4f}", 
                va='center', ha='left', color='red', fontweight='bold')
        
        # Accuracy subplot
        ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
        ax2.plot(epochs, valid_accs, 'r-', linewidth=2, label='Validation Accuracy')
        
        # Add test accuracy as a horizontal line if available
        if test_acc is not None:
            ax2.axhline(y=test_acc, color='g', linestyle='--', linewidth=2, label=f'Test Accuracy: {test_acc:.2f}%')
        
        ax2.set_title(f'Accuracy Curves for {format_label(model_name)}', fontsize=16)
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('Accuracy (%)', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add final accuracy values to the plot
        ax2.text(epochs[-1] + 0.1, train_accs[-1], f"{train_accs[-1]:.2f}%", 
                va='center', ha='left', color='blue', fontweight='bold')
        ax2.text(epochs[-1] + 0.1, valid_accs[-1], f"{valid_accs[-1]:.2f}%", 
                va='center', ha='left', color='red', fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig, f"10_learning_curves_{model_name}.png")
        plt.show()
        
        # Create a convergence analysis plot - showing how quickly the model converges
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate relative improvement for each epoch
        # Final validation accuracy as the target
        target_acc = valid_accs[-1]
        # Calculate what percentage of the final accuracy is achieved at each epoch
        relative_progress = [acc / target_acc * 100 for acc in valid_accs]
        
        # Plot relative progress
        ax.plot(epochs, relative_progress, 'o-', linewidth=2, markersize=8, color='purple')
        
        # Add markers for specific convergence points (80%, 90%, 95% of final accuracy)
        convergence_points = [80, 90, 95]
        for point in convergence_points:
            # Find first epoch where relative progress exceeds this point
            for i, progress in enumerate(relative_progress):
                if progress >= point:
                    ax.scatter(i + 1, progress, s=150, color='red', zorder=10)
                    ax.annotate(f"{point}% at epoch {i+1}", 
                               xy=(i + 1, progress), xytext=(-20, 20),
                               textcoords='offset points', 
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))
                    break
        
        ax.set_title(f'Convergence Analysis for {format_label(model_name)}', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Percentage of Final Accuracy (%)', fontsize=14)
        ax.set_xlim(0.5, len(epochs) + 0.5)
        ax.set_ylim(min(relative_progress) - 5, 105)
        
        # Add a horizontal line at 100%
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, f"10_convergence_analysis_{model_name}.png")
        plt.show()
        
        # Analyze overfitting
        # Calculate the gap between training and validation accuracy
        acc_gaps = [train - valid for train, valid in zip(train_accs, valid_accs)]
        
        # Create the overfitting analysis plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the accuracy gap
        ax.bar(epochs, acc_gaps, color=plt.cm.RdYlGn_r(np.array(acc_gaps)/max(acc_gaps) if max(acc_gaps) > 0 else 0))
        
        ax.set_title(f'Overfitting Analysis for {format_label(model_name)}', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Train-Validation Accuracy Gap (%)', fontsize=14)
        
        # Add a reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add threshold lines for overfitting
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Severe Overfitting')
        
        ax.legend(fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, f"10_overfitting_analysis_{model_name}.png")
        plt.show()
        
        # Print convergence analysis
        final_train_acc = train_accs[-1]
        final_valid_acc = valid_accs[-1]
        print(f"Training convergence for {format_label(model_name)}:")
        print(f"  Final training accuracy: {final_train_acc:.2f}%")
        print(f"  Final validation accuracy: {final_valid_acc:.2f}%")
        if test_acc is not None:
            print(f"  Test accuracy: {test_acc:.2f}%")
        
        # Analyze overfitting
        gap = final_train_acc - final_valid_acc
        if gap > 10:
            print("  Analysis: The model shows signs of severe overfitting (training accuracy significantly higher than validation)")
        elif gap > 5:
            print("  Analysis: The model shows signs of mild overfitting")
        elif gap < -5:
            print("  Analysis: Unusual pattern - validation accuracy higher than training (possible data leakage or sampling issues)")
        else:
            print("  Analysis: The model shows good generalization (training and validation accuracies are close)")
else:
    print("No training curves available in the results.")

#-------------------------------------------------------------------------
# 11. Summary and Recommendations
#-------------------------------------------------------------------------
print("\n11. Summary and Recommendations")

# Collect the top performing settings
top_performers = {}

# Best architecture
arch_results = {k: v for k, v in results.items() if "Architecture_" in k}
if arch_results:
    best_arch = max(arch_results.keys(), key=lambda k: results[k]["test_accuracy"])
    top_performers["Architecture"] = format_label(best_arch)

# Best training hyperparameters
training_results = {k: v for k, v in results.items() if "Training_Hyperparams_" in k}
if training_results:
    best_train = max(training_results.keys(), key=lambda k: results[k]["test_accuracy"])
    optimizer = results[best_train]["hyperparams"].get("optimizer", "unknown")
    lr = results[best_train]["hyperparams"].get("learning_rate", "unknown")
    scheduler = results[best_train]["hyperparams"].get("scheduler", "plateau")
    top_performers["Optimizer"] = optimizer.upper()
    top_performers["Learning Rate"] = lr
    top_performers["Scheduler"] = scheduler

# Best regularization
reg_results = {k: v for k, v in results.items() if "Regularization_Hyperparams_" in k}
if reg_results:
    best_reg = max(reg_results.keys(), key=lambda k: results[k]["test_accuracy"])
    wd = results[best_reg]["hyperparams"].get("weight_decay", "unknown")
    dropout = results[best_reg]["hyperparams"].get("dropout_rate", "unknown")
    top_performers["Weight Decay"] = wd
    if dropout != "unknown":
        top_performers["Dropout Rate"] = dropout

# Best augmentation
aug_results = {k: v for k, v in results.items() if "Augmentation_" in k}
if aug_results:
    best_aug = max(aug_results.keys(), key=lambda k: results[k]["test_accuracy"])
    top_performers["Data Augmentation"] = format_label(best_aug)

# Best ensemble method
ensemble_results = {}
for k, v in results.items():
    if isinstance(v, dict) and "hard_voting" in v and "soft_voting" in v:
        ensemble_results[k] = v

if ensemble_results:
    best_ensemble = max(ensemble_results.keys(), 
                       key=lambda k: max(results[k]["hard_voting"]["accuracy"], 
                                        results[k]["soft_voting"]["accuracy"]))
    
    if results[best_ensemble]["soft_voting"]["accuracy"] > results[best_ensemble]["hard_voting"]["accuracy"]:
        top_performers["Ensemble Method"] = "Soft Voting"
    else:
        top_performers["Ensemble Method"] = "Hard Voting"

# Create a summary dashboard
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
axs = axs.flatten()

# 1. Architecture Performance (top left)
if arch_results:
    arch_names = [format_label(name) for name in arch_results.keys()]
    arch_accs = [results[name]["test_accuracy"] for name in arch_results.keys()]
    
    # Sort by accuracy
    sorted_indices = np.argsort(arch_accs)[::-1]
    arch_names = [arch_names[i] for i in sorted_indices]
    arch_accs = [arch_accs[i] for i in sorted_indices]
    
    axs[0].bar(arch_names, arch_accs, color=sns.color_palette("viridis", len(arch_names)))
    axs[0].set_title("Architecture Performance", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_ylim(0, 100)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right")

# 2. Best Hyperparameters (top middle)
if training_results:
    best_params = []
    best_accs = []
    
    # Get best for each key parameter
    param_types = ["optimizer", "learning_rate", "scheduler"]
    for param in param_types:
        # Get unique values
        param_values = set()
        for name in training_results:
            if param in results[name]["hyperparams"]:
                param_values.add(results[name]["hyperparams"][param])
        
        # For each value, find the best accuracy
        for value in param_values:
            max_acc = 0
            for name in training_results:
                if param in results[name]["hyperparams"] and results[name]["hyperparams"][param] == value:
                    max_acc = max(max_acc, results[name]["test_accuracy"])
            
            best_params.append(f"{param}={value}")
            best_accs.append(max_acc)
    
    # Sort by accuracy
    sorted_indices = np.argsort(best_accs)[::-1]
    best_params = [best_params[i] for i in sorted_indices]
    best_accs = [best_accs[i] for i in sorted_indices]
    
    # Take top 5 for clarity
    best_params = best_params[:5]
    best_accs = best_accs[:5]
    
    axs[1].bar(best_params, best_accs, color=sns.color_palette("magma", len(best_params)))
    axs[1].set_title("Best Hyperparameters", fontsize=14)
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(0, 100)
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")

# 3. Regularization Impact (top right)
if reg_results:
    # Extract weight decay values and accuracies
    weight_decays = [results[name]["hyperparams"]["weight_decay"] for name in reg_results.keys()]
    accuracies = [results[name]["test_accuracy"] for name in reg_results.keys()]
    
    # Sort by weight decay
    sorted_indices = np.argsort(weight_decays)
    weight_decays = [weight_decays[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    axs[2].plot(weight_decays, accuracies, 'o-', linewidth=2, markersize=8, color="purple")
    axs[2].set_title("Weight Decay Impact", fontsize=14)
    axs[2].set_xlabel("Weight Decay")
    axs[2].set_ylabel("Accuracy (%)")
    
    # Handle zero values for log scale
    if 0 in weight_decays:
        axs[2].set_xscale('symlog', linthresh=1e-5)
    else:
        axs[2].set_xscale('log')

# 4. Data Augmentation Comparison (bottom left)
if aug_results:
    aug_names = [format_label(name) for name in aug_results.keys()]
    aug_accs = [results[name]["test_accuracy"] for name in aug_results.keys()]
    
    # Sort by accuracy
    sorted_indices = np.argsort(aug_accs)[::-1]
    aug_names = [aug_names[i] for i in sorted_indices]
    aug_accs = [aug_accs[i] for i in sorted_indices]
    
    # Define colors based on technique type
    colors = []
    for name in aug_names:
        if "cutout" in name.lower() or "mixup" in name.lower():
            colors.append("#3498db")  # Blue for advanced
        else:
            colors.append("#e74c3c")  # Red for standard
    
    axs[3].bar(aug_names, aug_accs, color=colors)
    axs[3].set_title("Data Augmentation Techniques", fontsize=14)
    axs[3].set_ylabel("Accuracy (%)")
    axs[3].set_ylim(0, 100)
    plt.setp(axs[3].get_xticklabels(), rotation=45, ha="right")

# 5. Ensemble Methods (bottom middle)
if ensemble_results:
    methods = []
    accs = []
    
    # Get best ensemble performance
    for name, ensemble in ensemble_results.items():
        methods.append("Hard-" + format_label(name))
        accs.append(ensemble["hard_voting"]["accuracy"])
        
        methods.append("Soft-" + format_label(name))
        accs.append(ensemble["soft_voting"]["accuracy"])
    
    # Add best single model for comparison
    best_single_acc = 0
    for key, value in results.items():
        if isinstance(value, dict) and "test_accuracy" in value:
            best_single_acc = max(best_single_acc, value["test_accuracy"])
    
    methods.append("Best Single")
    accs.append(best_single_acc)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accs)[::-1]
    methods = [methods[i] for i in sorted_indices]
    accs = [accs[i] for i in sorted_indices]
    
    # Define colors based on method type
    colors = []
    for method in methods:
        if "Hard" in method:
            colors.append("#3498db")  # Blue for hard voting
        elif "Soft" in method:
            colors.append("#2ecc71")  # Green for soft voting
        else:
            colors.append("#e74c3c")  # Red for single model
    
    axs[4].bar(methods, accs, color=colors)
    axs[4].set_title("Ensemble Methods", fontsize=14)
    axs[4].set_ylabel("Accuracy (%)")
    axs[4].set_ylim(0, 100)
    plt.setp(axs[4].get_xticklabels(), rotation=45, ha="right")

# 6. Final Recommendations (bottom right) - text based
axs[5].axis('off')  # Turn off axes for text box
recommendation_text = "RECOMMENDATIONS:\n\n"

if "Architecture" in top_performers:
    recommendation_text += f"1. Architecture: {top_performers['Architecture']}\n\n"
if "Optimizer" in top_performers and "Learning Rate" in top_performers:
    recommendation_text += f"2. Training: {top_performers['Optimizer']} (lr={top_performers['Learning Rate']})\n"
if "Scheduler" in top_performers:
    recommendation_text += f"   Scheduler: {top_performers['Scheduler']}\n\n"
if "Weight Decay" in top_performers:
    recommendation_text += f"3. Regularization: Weight Decay={top_performers['Weight Decay']}\n"
if "Dropout Rate" in top_performers:
    recommendation_text += f"   Dropout Rate={top_performers['Dropout Rate']}\n\n"
if "Data Augmentation" in top_performers:
    recommendation_text += f"4. Augmentation: {top_performers['Data Augmentation']}\n\n"
if "Ensemble Method" in top_performers:
    recommendation_text += f"5. Use {top_performers['Ensemble Method']} for final predictions"

axs[5].text(0, 0.95, recommendation_text, fontsize=12, va='top', ha='left', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
axs[5].set_title("Recommendations", fontsize=14)

plt.tight_layout()
save_figure(fig, "11_summary_dashboard.png")
plt.show()

# Print recommendations
print("Based on the experiment results, here are the recommended settings:")
for category, value in top_performers.items():
    print(f"  {category}: {value}")

print("\nOverall recommendations:")
if "Architecture" in top_performers:
    print(f"  1. Use the best performing architecture: {top_performers['Architecture']}")
if "Data Augmentation" in top_performers:
    print(f"  2. Apply data augmentation: {top_performers['Data Augmentation']}")
print("  3. Use optimal hyperparameters:")
if "Optimizer" in top_performers:
    print(f"     - Optimizer: {top_performers['Optimizer']}")
if "Learning Rate" in top_performers:
    print(f"     - Learning rate: {top_performers['Learning Rate']}")
if "Weight Decay" in top_performers:
    print(f"     - Weight decay: {top_performers['Weight Decay']}")
if "Dropout Rate" in top_performers:
    print(f"     - Dropout rate: {top_performers['Dropout Rate']}")
if "Ensemble Method" in top_performers:
    print(f"  4. Consider ensemble methods: {top_performers['Ensemble Method']} for final predictions")

print(f"\nAll visualizations have been saved to the '{diagrams_dir}' directory for inclusion in your report.")

# Return success
print("\nVisualization script completed successfully!")