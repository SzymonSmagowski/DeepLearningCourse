import copy
import math
import random
import time
from pathlib import Path
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.display import Audio, display
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

TARGET_SR = 16_000     # Speech-Commands is originally 16 kHz; keep it that way.
AUDIO_DURATION = 1.0   # seconds  (all clips in this dataset are 1 s)

def load_audio(path, target_sr=TARGET_SR):
    """
    Returns (audio_float32, sample_rate).
    Resamples on the fly if the original clip sr differs from target_sr.
    """
    audio, sr = librosa.load(str(path), sr=None)          # keep native sr first
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def normalize_audio_duration(audio, sr, duration=AUDIO_DURATION):
    """Center-crop (or zero-pad) the waveform to exactly <duration> seconds."""
    frame_len = int(sr * duration)

    if len(audio) >= frame_len:
        energy = librosa.util.frame(audio, frame_length=frame_len, hop_length=1)
        loudest_start = np.argmax(np.sum(energy**2, axis=0))
        audio = audio[loudest_start : loudest_start + frame_len]
    else:
        pad = frame_len - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")
    return audio


def extract_mfcc(audio, sr, n_mfcc=40):
    """Returns MFCCs shaped (n_mfcc, time)."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=512, hop_length=160,  # 10-ms hop @16 kHz
        fmin=20, fmax=sr // 2
    )
    return mfcc

def preprocess_dataset(dataset, name=""):
    """
    Reads every WAV → centers/normalises length → MFCC →
    returns (X, y) tensors ready for PyTorch.
    """
    features, labels = [], []

    for sample in tqdm(dataset, desc=f"Pre-processing {name}"):
        audio, sr = load_audio(sample["file_path"])
        audio = normalize_audio_duration(audio, sr)
        mfcc = extract_mfcc(audio, sr)

        features.append(mfcc)
        labels.append(sample["label_id"])

    X = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

def load_from_path(data_path):
    testing_hashset = set([
        file_name.strip() for file_name in open(data_path / 'testing_list.txt', 'r')
    ])
    validation_hashset = set([
        file_name.strip() for file_name in open(data_path / 'validation_list.txt', 'r')
    ])
    
    train_ = []
    valid_ = []
    test_ = []
    for root, _, files in os.walk(data_path):
        if root.endswith('_background_noise_'):
            continue
        root_path = Path(root)
        label = os.path.basename(root)
        for file in files:
            if not file.endswith('.wav'):
                continue
            labelled_file = label + '/' + file
            file_path = root_path / file
            data_entry = {
                'file_path': file_path,
                'label': label
            }
            if labelled_file in testing_hashset:
                test_.append(data_entry)
            elif labelled_file in validation_hashset:
                valid_.append(data_entry)
            else:
                train_.append(data_entry)

    return train_, valid_, test_

def show_random_example_per_class(dataset, n_cols=3):
    labels = sorted({e["label"] for e in dataset})
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 3))
    axes = axes.flat

    for ax, label in zip(axes, labels):
        ex = random.choice([e for e in dataset if e["label"] == label])
        wav, sr = load_audio(ex["file_path"])
        wav = normalize_audio_duration(wav, sr)           # 1-s crop/pad

        t = np.arange(len(wav)) / sr
        ax.plot(t, wav)
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amp")

        print(label)
        display(Audio(wav, rate=sr))
    
    # hide any empty subplots
    for ax in axes[len(labels):]:
        ax.axis("off")
    plt.tight_layout()


def plot_label_distribution(dataset, dataset_name=None):
    plt.figure(figsize=(10, 5))
    sns.histplot([entry['label'] for entry in train])
    plt.title(f"Labels distribution{(' ' + dataset_name) if dataset_name is not None else ''}")
    plt.show()

def run_epoch(device, model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, count = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss   = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(xb)
        correct    += (logits.argmax(1) == yb).sum().item()
        count      += len(xb)

    return total_loss / count, correct / count

def train_model(model,
                device,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler=None,
                epochs=30,
                early_stop=True,
                patience=3,
                verbose=True):
    """
    Trains <model> for up to <epochs>.  If early_stop is True, training stops
    when validation accuracy has not improved for <patience> epochs.

    Returns: history (list of dicts), best_val_acc
    """
    history = []
    best_val_acc = -float("inf")
    best_state   = None
    no_improve   = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(device, model, train_loader,
                                    criterion, optimizer)
        v_loss,  v_acc  = run_epoch(device, model, val_loader,   criterion)

        if scheduler and not isinstance(scheduler,
                                        torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        history.append({"epoch": ep,
                        "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": v_loss,   "val_acc": v_acc})

        if verbose:
            print(f"[{ep:02d}/{epochs}] "
                  f"train {tr_loss:.4f}/{tr_acc:.3f} – "
                  f"val {v_loss:.4f}/{v_acc:.3f}")

        # ----- early stopping bookkeeping -----
        if v_acc > best_val_acc + 1e-6:       # significant improvement
            best_val_acc = v_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if early_stop and no_improve >= patience:
                if verbose:
                    print(f"⏹️  Early stopping after {ep} epochs "
                          f"(no val-improve for {patience} epochs).")
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_acc


def eval_model(device, model, test_loader, label, id2label):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            all_preds.append(pred)
            all_true.append(yb)
    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()
    
    acc = (all_preds == all_true).mean()
    print(f"Test accuracy: {acc:.3%}")
    
    cm = confusion_matrix(all_true, all_preds, labels=list(id2label.keys()))
    disp = ConfusionMatrixDisplay(cm, display_labels=[id2label[i] for i in id2label])
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    if label is not None:
        ax.set_title(f"Confusion matrix – {label}")
    plt.show()


# Add these functions to your existing utils.py file

def save_results_to_json(results, filename):
    """
    Save training results to JSON file.
    
    Args:
        results (dict): Dictionary containing results
        filename (str): Path to save the JSON file
    """
    import json
    import os
    from pathlib import Path
    
    # Create directory if it doesn't exist
    os.makedirs(Path(filename).parent, exist_ok=True)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {filename}")


def plot_training_history(history, title="Training History", figsize=(12, 5), save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
    """
    import matplotlib.pyplot as plt
    
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
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
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
    
    plt.show()


def create_results_visualizations(results_file, output_dir=None):
    """
    Create visualizations from a results JSON file.
    
    Args:
        results_file (str): Path to results JSON file
        output_dir (str, optional): Directory to save visualizations
    """
    import json
    import os
    import numpy as np
    from pathlib import Path
    
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
    class_names = [id2label[str(i)] for i in range(len(id2label))]
    plot_confusion_matrix(
        cm, 
        class_names=class_names,
        title=f"Confusion Matrix - {model_name}",
        save_path=f"{output_dir}/{base_filename}_cm.png"
    )
    
    print(f"Visualizations saved to {output_dir}")