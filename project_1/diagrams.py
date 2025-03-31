# Deep Learning Experiments Analysis Notebook

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)

# Load the experiment results
try:
    with open('experiment_results.json', 'r') as f:
        results = json.load(f)
    print("Successfully loaded experiment results!")
except FileNotFoundError:
    print("Could not find experiment_results.json. Please run the experiments first.")
    # Create dummy results for notebook demonstration
    results = {
        "dummy_data": True,
        "Architecture_efficientnet": {"test_accuracy": 85.5},
        "Architecture_custom_cnn": {"test_accuracy": 78.2},
        "Training_Hyperparams_1": {"hyperparams": {"optimizer": "adam", "learning_rate": 0.001}, "test_accuracy": 83.1},
        "Training_Hyperparams_2": {"hyperparams": {"optimizer": "sgd", "learning_rate": 0.01}, "test_accuracy": 81.5},
        "Training_Hyperparams_3": {"hyperparams": {"optimizer": "adam", "learning_rate": 0.001, "scheduler": "cosine"}, "test_accuracy": 84.2},
        "Regularization_Hyperparams_1": {"hyperparams": {"weight_decay": 0.0001}, "test_accuracy": 83.5},
        "Regularization_Hyperparams_2": {"hyperparams": {"weight_decay": 0.001}, "test_accuracy": 82.8},
        "Regularization_Hyperparams_3": {"hyperparams": {"weight_decay": 0.0}, "test_accuracy": 81.2},
        "Augmentation_baseline": {"test_accuracy": 82.5},
        "Augmentation_horizontal_flip": {"test_accuracy": 84.1},
        "Augmentation_rotation": {"test_accuracy": 83.7},
        "Augmentation_color_jitter": {"test_accuracy": 83.9},
        "Augmentation_cutout": {"test_accuracy": 85.0},
        "few_shot": {"accuracy": 65.3, "std": 3.2},
        "Reduced_Train_Size_4000": {"test_accuracy": 79.8},
        "Reduced_Train_Size_2000": {"test_accuracy": 75.3},
        "ensemble": {"hard_voting": {"accuracy": 86.2}, "soft_voting": {"accuracy": 87.1}}
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

# 1. Architecture Comparison
print("\n1. Architecture Comparison")
arch_results = {k: v for k, v in results.items() if "Architecture_" in k}
if arch_results:
    arch_names = [format_label(name) for name in arch_results.keys()]
    arch_accs = [results[name]["test_accuracy"] for name in arch_results.keys()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(arch_names, arch_accs, color=sns.color_palette("viridis", len(arch_names)))
    plt.title("Architecture Comparison - Test Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(arch_accs) - 5, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("architecture_comparison.png")
    plt.show()
    
    # Print the best architecture
    best_arch = max(arch_results.keys(), key=lambda k: results[k]["test_accuracy"])
    print(f"Best architecture: {format_label(best_arch)} with {results[best_arch]['test_accuracy']:.2f}% accuracy")
else:
    print("No architecture comparison results found.")

# 2. Training Hyperparameters Analysis
print("\n2. Training Hyperparameters Analysis")
training_results = {k: v for k, v in results.items() if "Training_Hyperparams_" in k}
if training_results:
    # Create readable labels
    labels = [format_label(name) for name in training_results.keys()]
    accuracies = [results[name]["test_accuracy"] for name in training_results.keys()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=sns.color_palette("magma", len(labels)))
    plt.title("Training Hyperparameters - Test Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(accuracies) - 5, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("training_hyperparams_comparison.png")
    plt.show()
    
    # Print the best training hyperparameters
    best_train = max(training_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_params = results[best_train]["hyperparams"]
    print(f"Best training hyperparameters: {best_params} with {results[best_train]['test_accuracy']:.2f}% accuracy")
else:
    print("No training hyperparameter results found.")

# 3. Regularization Hyperparameters Analysis
print("\n3. Regularization Hyperparameters Analysis")
reg_results = {k: v for k, v in results.items() if "Regularization_Hyperparams_" in k}
if reg_results:
    # Extract weight decay values and accuracies
    weight_decays = [results[name]["hyperparams"]["weight_decay"] for name in reg_results.keys()]
    accuracies = [results[name]["test_accuracy"] for name in reg_results.keys()]
    
    # Sort by weight decay for better visualization
    sorted_indices = np.argsort(weight_decays)
    weight_decays = [weight_decays[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(weight_decays, accuracies, 'o-', linewidth=2, markersize=10)
    plt.title("Effect of Weight Decay on Test Accuracy")
    plt.xlabel("Weight Decay")
    plt.ylabel("Accuracy (%)")
    plt.xscale('symlog', linthresh=1e-5)  # Handles zero weight decay nicely
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(weight_decays, accuracies)):
        plt.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("regularization_comparison.png")
    plt.show()
    
    # Print the best regularization hyperparameters
    best_reg = max(reg_results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_wd = results[best_reg]["hyperparams"]["weight_decay"]
    print(f"Best weight decay: {best_wd} with {results[best_reg]['test_accuracy']:.2f}% accuracy")
else:
    print("No regularization hyperparameter results found.")

# 4. Data Augmentation Analysis
print("\n4. Data Augmentation Analysis")
aug_results = {k: v for k, v in results.items() if "Augmentation_" in k}
if aug_results:
    aug_names = [format_label(name) for name in aug_results.keys()]
    aug_accs = [results[name]["test_accuracy"] for name in aug_results.keys()]
    
    # Create a color palette that distinguishes standard vs. advanced techniques
    colors = sns.color_palette("YlOrRd", len(aug_names))
    if "cutout" in " ".join(aug_names).lower():
        # Make cutout a distinct color
        for i, name in enumerate(aug_names):
            if "cutout" in name.lower():
                colors[i] = (0.2, 0.4, 0.8)  # Different blue color
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(aug_names, aug_accs, color=colors)
    plt.title("Data Augmentation Techniques - Test Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(aug_accs) - 2, max(aug_accs) + 3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("augmentation_comparison.png")
    plt.show()
    
    # Print the best augmentation technique
    best_aug = max(aug_results.keys(), key=lambda k: results[k]["test_accuracy"])
    print(f"Best augmentation technique: {format_label(best_aug)} with {results[best_aug]['test_accuracy']:.2f}% accuracy")
else:
    print("No data augmentation results found.")

# 5. Few-shot Learning Results
print("\n5. Few-shot Learning Results")
if "few_shot" in results:
    few_shot = results["few_shot"]
    acc = few_shot["accuracy"]
    std = few_shot["std"]
    
    plt.figure(figsize=(8, 6))
    plt.bar(["5-way, 5-shot"], [acc], yerr=[std], capsize=10, color="green", alpha=0.7)
    plt.title("Few-shot Learning Performance")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    
    # Add value label
    plt.text(0, acc + std + 1, f"{acc:.1f}% ± {std:.1f}%", 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("few_shot_learning.png")
    plt.show()
    
    print(f"Few-shot learning (5-way, 5-shot): {acc:.2f}% ± {std:.2f}%")
else:
    print("No few-shot learning results found.")

# 6. Reduced Training Set Size Analysis
print("\n6. Reduced Training Set Size Analysis")
reduced_results = {k: v for k, v in results.items() if "Reduced_Train_Size_" in k}
if reduced_results:
    # Get the baseline accuracy from full dataset
    baseline_acc = None
    for key in results.keys():
        if "Architecture_" in key and "efficientnet" in key.lower():
            baseline_acc = results[key]["test_accuracy"]
            break
    
    # Extract sizes and accuracies
    sizes = [int(name.split("_")[-1]) for name in reduced_results.keys()]
    accuracies = [results[name]["test_accuracy"] for name in reduced_results.keys()]
    
    # Add the full dataset size if we have the baseline accuracy
    if baseline_acc is not None:
        # Assuming the full size is 2x the largest reduced size
        full_size = max(sizes) * 2
        sizes.append(full_size)
        accuracies.append(baseline_acc)
    
    # Sort by size
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, accuracies, 'o-', linewidth=2, markersize=10, color="purple")
    plt.title("Effect of Training Dataset Size on Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(sizes, accuracies):
        plt.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    # Use integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig("reduced_training_size.png")
    plt.show()
    
    # Print the relationship between dataset size and accuracy
    print(f"Training dataset size analysis:")
    for size, acc in zip(sizes, accuracies):
        print(f"  Size {size}: {acc:.2f}% accuracy")
else:
    print("No reduced training set size results found.")

# 7. Ensemble Methods Analysis
print("\n7. Ensemble Methods Analysis")
if "ensemble" in results:
    ensemble = results["ensemble"]
    
    # Get the best single model accuracy
    best_single_acc = 0
    for key, value in results.items():
        if isinstance(value, dict) and "test_accuracy" in value:
            if value["test_accuracy"] > best_single_acc:
                best_single_acc = value["test_accuracy"]
    
    # Extract ensemble accuracies
    methods = ["Hard Voting", "Soft Voting", "Best Single Model"]
    accuracies = [
        ensemble["hard_voting"]["accuracy"],
        ensemble["soft_voting"]["accuracy"],
        best_single_acc
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=["#6d88cf", "#5c4d93", "#3c3c3c"])
    plt.title("Ensemble Methods vs. Single Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(accuracies) - 2, max(accuracies) + 2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("ensemble_methods.png")
    plt.show()
    
    # Print ensemble results
    print(f"Ensemble methods comparison:")
    print(f"  Hard voting: {ensemble['hard_voting']['accuracy']:.2f}%")
    print(f"  Soft voting: {ensemble['soft_voting']['accuracy']:.2f}%")
    print(f"  Best single model: {best_single_acc:.2f}%")
    
    # Calculate improvement
    best_ensemble = max(ensemble["hard_voting"]["accuracy"], ensemble["soft_voting"]["accuracy"])
    improvement = best_ensemble - best_single_acc
    print(f"  Ensemble improvement over best single model: {improvement:.2f}%")
else:
    print("No ensemble method results found.")

# 8. Comprehensive Comparison of All Methods
print("\n8. Comprehensive Comparison of All Methods")

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
if "few_shot" in results:
    best_results["Few-shot"] = {"name": "5-way, 5-shot", "accuracy": results["few_shot"]["accuracy"]}

# Best ensemble
if "ensemble" in results:
    ensemble = results["ensemble"]
    best_ensemble_acc = max(ensemble["hard_voting"]["accuracy"], ensemble["soft_voting"]["accuracy"])
    best_ensemble_name = "Soft Voting" if ensemble["soft_voting"]["accuracy"] > ensemble["hard_voting"]["accuracy"] else "Hard Voting"
    best_results["Ensemble"] = {"name": best_ensemble_name, "accuracy": best_ensemble_acc}

# Create a comprehensive comparison chart
if best_results:
    categories = list(best_results.keys())
    methods = [best_results[cat]["name"] for cat in categories]
    accuracies = [best_results[cat]["accuracy"] for cat in categories]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]  # Descending
    categories = [categories[i] for i in sorted_indices]
    methods = [methods[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Create a colorful bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(categories, accuracies, color=sns.color_palette("viridis", len(categories)))
    
    # Add method labels above the bars
    for i, (bar, method) in enumerate(zip(bars, methods)):
        plt.text(bar.get_x() + bar.get_width()/2., 30, 
                method, ha='center', va='bottom', 
                rotation=90, color='white', fontweight='bold')
    
    # Add value labels at the top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.title("Comparison of Best Results Across All Experiment Types")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(accuracies) + 5)
    plt.tight_layout()
    plt.savefig("comprehensive_comparison.png")
    plt.show()
    
    # Print overall best method
    best_category = categories[0]
    best_method = methods[0]
    best_acc = accuracies[0]
    print(f"Overall best performance: {best_category} ({best_method}) with {best_acc:.2f}% accuracy")

# 9. Training vs. Validation Curves for Best Architecture
print("\n9. Training Progress Analysis")

# Check if training curves are available
has_training_curves = False
for key, value in results.items():
    if isinstance(value, dict) and "train_losses" in value and "valid_losses" in value:
        has_training_curves = True
        break

if has_training_curves:
    # Find the best architecture or model to plot curves for
    best_model_key = None
    best_acc = 0
    
    for key, value in results.items():
        if isinstance(value, dict) and "test_accuracy" in value:
            if value["test_accuracy"] > best_acc:
                best_acc = value["test_accuracy"]
                best_model_key = key
    
    if best_model_key and "train_losses" in results[best_model_key]:
        model_results = results[best_model_key]
        
        epochs = list(range(1, len(model_results["train_losses"]) + 1))
        
        plt.figure(figsize=(16, 6))
        
        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, model_results["train_losses"], 'b-', label='Training Loss')
        plt.plot(epochs, model_results["valid_losses"], 'r-', label='Validation Loss')
        plt.title(f'Training Curves for {format_label(best_model_key)}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, model_results["train_accuracies"], 'b-', label='Training Accuracy')
        plt.plot(epochs, model_results["valid_accuracies"], 'r-', label='Validation Accuracy')
        plt.title(f'Accuracy Curves for {format_label(best_model_key)}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_curves.png")
        plt.show()
        
        # Print convergence analysis
        final_train_acc = model_results["train_accuracies"][-1]
        final_valid_acc = model_results["valid_accuracies"][-1]
        print(f"Training convergence for {format_label(best_model_key)}:")
        print(f"  Final training accuracy: {final_train_acc:.2f}%")
        print(f"  Final validation accuracy: {final_valid_acc:.2f}%")
        print(f"  Test accuracy: {model_results['test_accuracy']:.2f}%")
        
        if final_train_acc - final_valid_acc > 5:
            print("  Analysis: The model shows signs of overfitting (training accuracy significantly higher than validation)")
        elif final_valid_acc - final_train_acc > 5:
            print("  Analysis: Unusual pattern - validation accuracy higher than training (possible data leakage or sampling issues)")
        else:
            print("  Analysis: The model shows good generalization (training and validation accuracies are close)")
else:
    print("No training curves available in the results.")

# 10. Summary and Recommendations
print("\n10. Summary and Recommendations")

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
    top_performers["Weight Decay"] = wd

# Best augmentation
aug_results = {k: v for k, v in results.items() if "Augmentation_" in k}
if aug_results:
    best_aug = max(aug_results.keys(), key=lambda k: results[k]["test_accuracy"])
    top_performers["Data Augmentation"] = format_label(best_aug)

# Best ensemble method
if "ensemble" in results:
    ensemble = results["ensemble"]
    if ensemble["soft_voting"]["accuracy"] > ensemble["hard_voting"]["accuracy"]:
        top_performers["Ensemble Method"] = "Soft Voting"
    else:
        top_performers["Ensemble Method"] = "Hard Voting"

# Print recommendations
print("Based on the experiment results, here are the recommended settings:")
for category, value in top_performers.items():
    print(f"  {category}: {value}")

print("\nOverall recommendations:")
print("  1. Use the best performing architecture: " + top_performers.get("Architecture", "Unknown"))
print("  2. Apply data augmentation: " + top_performers.get("Data Augmentation", "Unknown"))
print("  3. Use optimal hyperparameters:")
print(f"     - Optimizer: {top_performers.get('Optimizer', 'Unknown')}")
print(f"     - Learning rate: {top_performers.get('Learning Rate', 'Unknown')}")
print(f"     - Weight decay: {top_performers.get('Weight Decay', 'Unknown')}")
print(f"  4. Consider ensemble methods: {top_performers.get('Ensemble Method', 'Unknown')} for final predictions")

print("\nAll analysis images have been saved to disk for inclusion in your report.")