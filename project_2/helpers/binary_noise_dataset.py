import copy
import math
import random
import time
from pathlib import Path
import os
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter, defaultdict

from helpers.utils import extract_mfcc, load_audio, normalize_audio_duration, TARGET_SR, AUDIO_DURATION


def load_from_path_with_noise(data_path):
    """
    Load dataset including background noise files as a separate class.
    Similar to load_from_path but doesn't skip the _background_noise_ folder.
    
    Args:
        data_path (Path): Path to the dataset root
    
    Returns:
        train_, valid_, test_: Lists of data entries with both speech and background noise
    """
    testing_hashset = set([
        file_name.strip() for file_name in open(data_path / 'testing_list.txt', 'r')
    ])
    validation_hashset = set([
        file_name.strip() for file_name in open(data_path / 'validation_list.txt', 'r')
    ])
    
    # Initialize data lists
    train_ = []
    valid_ = []
    test_ = []
    
    for root, _, files in os.walk(data_path):
        root_path = Path(root)
        label = os.path.basename(root)
        
        # Handle background noise differently
        if label == '_background_noise_':
            # Split noise files into train/valid/test
            # We'll use a deterministic approach based on file names
            for file in files:
                if not file.endswith('.wav'):
                    continue
                
                file_path = root_path / file
                # Set binary label for noise
                data_entry = {
                    'file_path': file_path,
                    'label': 'noise',  # Binary label
                    'original_label': '_background_noise_'  # Original folder name
                }
                
                # Deterministic split based on file name hash
                file_hash = hash(file) % 100
                if file_hash < 70:  # 70% to train
                    train_.append(data_entry)
                elif file_hash < 85:  # 15% to validation
                    valid_.append(data_entry)
                else:  # 15% to test
                    test_.append(data_entry)
        else:
            # Regular speech files
            for file in files:
                if not file.endswith('.wav'):
                    continue
                
                labelled_file = label + '/' + file
                file_path = root_path / file
                
                # Set binary label for speech
                data_entry = {
                    'file_path': file_path,
                    'label': 'speech',  # Binary label
                    'original_label': label  # Original class label
                }
                
                # Use the same split as in the original dataset
                if labelled_file in testing_hashset:
                    test_.append(data_entry)
                elif labelled_file in validation_hashset:
                    valid_.append(data_entry)
                else:
                    train_.append(data_entry)
    
    return train_, valid_, test_


def chunk_noise_file(audio, sr, chunks=10, max_var=0.08):
    """
    Chunk a long audio file into segments of approximately 1 second with slight variations.
    
    Args:
        audio (np.ndarray): Audio data
        sr (int): Sample rate
        chunks (int): Number of chunks to create
        max_var (float): Maximum variation in chunk length (±max_var)
        
    Returns:
        list: List of audio chunks
    """
    chunks_list = []
    file_duration = len(audio) / sr
    
    # Determine how many 1-second chunks we can extract
    available_chunks = int(file_duration / AUDIO_DURATION)
    num_chunks = min(available_chunks, chunks)
    
    # If file is shorter than 1 second, just return it as is
    if available_chunks < 1:
        return [audio]
    
    # Create chunks with random variations
    for _ in range(num_chunks):
        # Random variation in chunk length
        var_factor = 1.0 + random.uniform(-max_var, max_var)
        chunk_length = int(sr * AUDIO_DURATION * var_factor)
        
        # Find a random starting point
        max_start = max(0, len(audio) - chunk_length)
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)
        
        # Extract chunk
        chunk = audio[start_idx:start_idx + chunk_length]
        chunks_list.append(chunk)
    
    return chunks_list


def preprocess_binary_dataset(dataset, name="", chunks_per_noise_file=10):
    """
    Preprocesses dataset for binary classification (speech vs. noise).
    
    Args:
        dataset (list): List of data entries with file_path and label
        name (str): Name for progress bar
        chunks_per_noise_file (int): Number of chunks to extract from each noise file
        
    Returns:
        X, y: PyTorch tensors for features and labels
    """
    features, labels = [], []
    label_map = {'speech': 1, 'noise': 0}  # Binary labels
    
    for sample in tqdm(dataset, desc=f"Pre-processing {name}"):
        audio, sr = load_audio(sample["file_path"])
        
        if sample["label"] == "noise":
            # For noise files, create multiple chunks
            chunks = chunk_noise_file(audio, sr, chunks=chunks_per_noise_file)
            
            for chunk in chunks:
                # Normalize length (pad/crop to 1 second)
                norm_chunk = normalize_audio_duration(chunk, sr)
                
                # Extract MFCC features
                mfcc = extract_mfcc(norm_chunk, sr)
                
                # Add to dataset
                features.append(mfcc)
                labels.append(label_map[sample["label"]])
        else:
            # For speech files, process normally
            audio = normalize_audio_duration(audio, sr)
            mfcc = extract_mfcc(audio, sr)
            
            features.append(mfcc)
            labels.append(label_map[sample["label"]])
    
    X = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def create_binary_dataset(data_path, batch_size=32, seed=42):
    """
    Creates a binary classification dataset (speech vs. noise).
    
    Args:
        data_path (str/Path): Path to the dataset
        batch_size (int): Batch size for the data loaders
        seed (int): Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training/validation/testing
        id2label: Dictionary mapping from label ID to label name
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data with noise files included
    print(f"Loading binary dataset from {data_path}...")
    data_path = Path(data_path)
    train_, valid_, test_ = load_from_path_with_noise(data_path)
    
    # Print dataset statistics
    train_speech = sum(1 for item in train_ if item['label'] == 'speech')
    train_noise = sum(1 for item in train_ if item['label'] == 'noise')
    print(f"Train set: {len(train_)} samples ({train_speech} speech, {train_noise} noise)")
    
    val_speech = sum(1 for item in valid_ if item['label'] == 'speech')
    val_noise = sum(1 for item in valid_ if item['label'] == 'noise')
    print(f"Validation set: {len(valid_)} samples ({val_speech} speech, {val_noise} noise)")
    
    test_speech = sum(1 for item in test_ if item['label'] == 'speech')
    test_noise = sum(1 for item in test_ if item['label'] == 'noise')
    print(f"Test set: {len(test_)} samples ({test_speech} speech, {test_noise} noise)")
    
    # Extract features
    print("Extracting features...")
    X_train, y_train = preprocess_binary_dataset(train_, "train")
    X_val, y_val = preprocess_binary_dataset(valid_, "valid")
    X_test, y_test = preprocess_binary_dataset(test_, "test")
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'noise': 0, 'speech': 1}


def compute_class_weights(y_train):
    """
    Compute class weights to handle class imbalance.
    
    Args:
        y_train (Tensor): Training labels
        
    Returns:
        Tensor: Class weights
    """
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights


def augment_audio(audio, sr, augmentation_methods=['pitch', 'speed', 'noise', 'stretch']):
    """
    Apply random audio augmentation techniques.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        augmentation_methods (list): List of augmentation methods to choose from
        
    Returns:
        np.ndarray: Augmented audio
    """
    # Choose a random augmentation method
    method = random.choice(augmentation_methods)
    
    if method == 'pitch':
        # Pitch shift by up to ±4 semitones
        n_steps = random.uniform(-4.0, 4.0)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    elif method == 'speed':
        # Speed up or slow down by up to ±15%
        speed_factor = random.uniform(0.85, 1.15)
        # Use librosa.effects.time_stretch for speed change
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    elif method == 'noise':
        # Add random Gaussian noise with SNR between 10-20dB
        noise_level = random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    elif method == 'stretch':
        # Time stretch without changing pitch by up to ±10%
        stretch_factor = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    return audio  # Return original if no method matches


def create_balanced_binary_dataset(data_path, batch_size=32, seed=42, 
                                  target_ratio=1.0, oversampling_factor=5, 
                                  undersampling_factor=0.3,
                                  augment=True):
    """
    Creates a balanced binary classification dataset (speech vs. noise).
    Uses oversampling of noise and undersampling of speech to achieve balance.
    
    Args:
        data_path (str/Path): Path to the dataset
        batch_size (int): Batch size for the data loaders
        seed (int): Random seed for reproducibility
        target_ratio (float): Target ratio of noise to speech samples (1.0 = equal)
        oversampling_factor (int): How many times to oversample each noise sample
        undersampling_factor (float): Fraction of speech samples to keep
        augment (bool): Whether to apply audio augmentation to oversampled noise
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training/validation/testing
        id2label: Dictionary mapping from label ID to label name
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data with noise files included
    print(f"Loading binary dataset from {data_path}...")
    data_path = Path(data_path)
    train_, valid_, test_ = load_from_path_with_noise(data_path)
    
    # Count original samples
    train_speech_original = sum(1 for item in train_ if item['label'] == 'speech')
    train_noise_original = sum(1 for item in train_ if item['label'] == 'noise')
    
    print(f"Original train set: {len(train_)} samples "
          f"({train_speech_original} speech, {train_noise_original} noise)")
    print(f"Original imbalance ratio: {train_speech_original / max(1, train_noise_original):.2f} speech:noise")
    
    # Separate speech and noise samples
    train_speech = [item for item in train_ if item['label'] == 'speech']
    train_noise = [item for item in train_ if item['label'] == 'noise']
    
    # Group speech samples by original label to ensure each word type is preserved
    speech_by_original_label = defaultdict(list)
    for item in train_speech:
        speech_by_original_label[item['original_label']].append(item)
    
    # Calculate how many samples to keep from each original label
    total_speech_to_keep = int(train_speech_original * undersampling_factor)
    labels_count = len(speech_by_original_label)
    samples_per_label = max(1, total_speech_to_keep // labels_count)
    
    # Undersample speech while preserving all original words
    undersampled_speech = []
    for label, samples in speech_by_original_label.items():
        # Keep at least 1 sample per label, but not more than available
        keep_count = min(len(samples), samples_per_label)
        # Randomly select samples to keep
        undersampled_speech.extend(random.sample(samples, keep_count))
    
    # Calculate how many noise samples we need after oversampling
    # to achieve the target ratio with the undersampled speech
    target_noise_count = int(len(undersampled_speech) * target_ratio)
    
    # If we need more noise samples than we have after applying the oversampling factor
    # adjust the oversampling factor accordingly
    required_oversampling_factor = math.ceil(target_noise_count / max(1, len(train_noise)))
    actual_oversampling_factor = max(oversampling_factor, required_oversampling_factor)
    
    print(f"Undersampled speech count: {len(undersampled_speech)}")
    print(f"Required noise count for balance: {target_noise_count}")
    print(f"Using oversampling factor: {actual_oversampling_factor}")
    
    # Oversample the noise
    oversampled_noise = []
    for _ in range(actual_oversampling_factor):
        oversampled_noise.extend(train_noise)
    
    # If we have too many noise samples after oversampling, randomly select the target amount
    if len(oversampled_noise) > target_noise_count:
        oversampled_noise = random.sample(oversampled_noise, target_noise_count)
    
    # Create balanced training set
    balanced_train = undersampled_speech + oversampled_noise
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_train)
    
    # Print final balance
    final_speech = sum(1 for item in balanced_train if item['label'] == 'speech')
    final_noise = sum(1 for item in balanced_train if item['label'] == 'noise')
    print(f"Final balanced train set: {len(balanced_train)} samples "
          f"({final_speech} speech, {final_noise} noise)")
    print(f"Final ratio: {final_speech / max(1, final_noise):.2f} speech:noise")
    
    # Keep validation and test sets as is
    val_speech = sum(1 for item in valid_ if item['label'] == 'speech')
    val_noise = sum(1 for item in valid_ if item['label'] == 'noise')
    print(f"Validation set: {len(valid_)} samples ({val_speech} speech, {val_noise} noise)")
    
    test_speech = sum(1 for item in test_ if item['label'] == 'speech')
    test_noise = sum(1 for item in test_ if item['label'] == 'noise')
    print(f"Test set: {len(test_)} samples ({test_speech} speech, {test_noise} noise)")
    
    # Extract features with augmentation for oversampled noise
    print("Extracting features...")
    
    def preprocess_balanced_dataset(dataset, name=""):
        """Preprocess dataset with augmentation for oversampled noise"""
        features, labels = [], []
        label_map = {'speech': 1, 'noise': 0}
        noise_count = defaultdict(int)  # Track noise file counts for augmentation
        
        for sample in tqdm(dataset, desc=f"Pre-processing {name}"):
            audio, sr = load_audio(sample["file_path"])
            
            # For noise samples in the training set that are oversampled (duplicates)
            if sample["label"] == "noise" and name == "train":
                # Track how many times we've seen this file
                file_key = str(sample["file_path"])
                noise_count[file_key] += 1
                
                # If it's being oversampled and augmentation is enabled, apply augmentation
                if noise_count[file_key] > 1 and augment:
                    audio = augment_audio(audio, sr)
                
                # Chunk the noise audio
                chunks = chunk_noise_file(audio, sr)
                
                for chunk in chunks:
                    # Normalize length
                    norm_chunk = normalize_audio_duration(chunk, sr)
                    
                    # Extract MFCC
                    mfcc = extract_mfcc(norm_chunk, sr)
                    
                    # Add to dataset
                    features.append(mfcc)
                    labels.append(label_map[sample["label"]])
            
            # For speech files or non-training noise, process normally
            else:
                if sample["label"] == "noise":
                    # Chunk the noise audio
                    chunks = chunk_noise_file(audio, sr)
                    
                    for chunk in chunks:
                        norm_chunk = normalize_audio_duration(chunk, sr)
                        mfcc = extract_mfcc(norm_chunk, sr)
                        features.append(mfcc)
                        labels.append(label_map[sample["label"]])
                else:
                    # Process speech normally
                    audio = normalize_audio_duration(audio, sr)
                    mfcc = extract_mfcc(audio, sr)
                    features.append(mfcc)
                    labels.append(label_map[sample["label"]])
        
        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y
    
    # Process the datasets
    X_train, y_train = preprocess_balanced_dataset(balanced_train, "train")
    X_val, y_val = preprocess_binary_dataset(valid_, "valid")
    X_test, y_test = preprocess_binary_dataset(test_, "test")
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Print final class distribution after preprocessing
    train_speech_final = (y_train == 1).sum().item()
    train_noise_final = (y_train == 0).sum().item()
    print(f"Final processed train set: {len(y_train)} samples "
          f"({train_speech_final} speech, {train_noise_final} noise)")
    print(f"Final processed ratio: {train_speech_final / max(1, train_noise_final):.2f} speech:noise")
    
    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'noise': 0, 'speech': 1}

def create_balanced_binary_dataset_fast(data_path, batch_size=32, seed=42, 
                               target_ratio=1.0, oversampling_factor=5, 
                               undersampling_factor=0.3):
    """
    Creates a balanced binary classification dataset (speech vs. noise) with optimized processing.
    Uses simple augmentation techniques that are computationally efficient.
    
    Args:
        data_path (str/Path): Path to the dataset
        batch_size (int): Batch size for the data loaders
        seed (int): Random seed for reproducibility
        target_ratio (float): Target ratio of noise to speech samples (1.0 = equal)
        oversampling_factor (int): How many times to oversample each noise sample
        undersampling_factor (float): Fraction of speech samples to keep
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training/validation/testing
        id2label: Dictionary mapping from label ID to label name
    """
    import random
    import math
    import numpy as np
    import torch
    from pathlib import Path
    from collections import defaultdict
    from torch.utils.data import DataLoader, TensorDataset
    from helpers.utils import extract_mfcc, load_audio, normalize_audio_duration
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data with noise files included
    print(f"Loading binary dataset from {data_path}...")
    data_path = Path(data_path)
    train_, valid_, test_ = load_from_path_with_noise(data_path)
    
    # Count original samples
    train_speech_original = sum(1 for item in train_ if item['label'] == 'speech')
    train_noise_original = sum(1 for item in train_ if item['label'] == 'noise')
    
    print(f"Original train set: {len(train_)} samples "
          f"({train_speech_original} speech, {train_noise_original} noise)")
    print(f"Original imbalance ratio: {train_speech_original / max(1, train_noise_original):.2f} speech:noise")
    
    # Separate speech and noise samples
    train_speech = [item for item in train_ if item['label'] == 'speech']
    train_noise = [item for item in train_ if item['label'] == 'noise']
    
    # Group speech samples by original label to ensure each word type is preserved
    speech_by_original_label = defaultdict(list)
    for item in train_speech:
        speech_by_original_label[item['original_label']].append(item)
    
    # Calculate how many samples to keep from each original label
    total_speech_to_keep = int(train_speech_original * undersampling_factor)
    labels_count = len(speech_by_original_label)
    samples_per_label = max(1, total_speech_to_keep // labels_count)
    
    # Undersample speech while preserving all original words
    undersampled_speech = []
    for label, samples in speech_by_original_label.items():
        # Keep at least 1 sample per label, but not more than available
        keep_count = min(len(samples), samples_per_label)
        # Randomly select samples to keep
        undersampled_speech.extend(random.sample(samples, keep_count))
    
    # Calculate how many noise samples we need after oversampling
    target_noise_count = int(len(undersampled_speech) * target_ratio)
    
    # If we need more noise samples than we have after applying the oversampling factor
    # adjust the oversampling factor accordingly
    required_oversampling_factor = math.ceil(target_noise_count / max(1, len(train_noise)))
    actual_oversampling_factor = max(oversampling_factor, required_oversampling_factor)
    
    print(f"Undersampled speech count: {len(undersampled_speech)}")
    print(f"Required noise count for balance: {target_noise_count}")
    print(f"Using oversampling factor: {actual_oversampling_factor}")
    
    # Oversample noise more efficiently
    oversampled_noise = []
    # First round - add all original noise samples (no augmentation needed)
    oversampled_noise.extend(train_noise)
    
    # Add duplicates for remaining rounds (tagged for simple augmentation)
    for i in range(1, actual_oversampling_factor):
        for noise_sample in train_noise:
            # Create a copy with an augmentation flag
            augmented_sample = noise_sample.copy()
            augmented_sample['augment'] = True
            augmented_sample['augment_type'] = random.choice(['volume', 'noise', 'shift'])
            oversampled_noise.append(augmented_sample)
    
    # If we have too many noise samples after oversampling, randomly select the target amount
    if len(oversampled_noise) > target_noise_count:
        oversampled_noise = random.sample(oversampled_noise, target_noise_count)
    
    # Create balanced training set
    balanced_train = undersampled_speech + oversampled_noise
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_train)
    
    # Print final balance
    final_speech = sum(1 for item in balanced_train if item['label'] == 'speech')
    final_noise = sum(1 for item in balanced_train if item['label'] == 'noise')
    print(f"Final balanced train set: {len(balanced_train)} samples "
          f"({final_speech} speech, {final_noise} noise)")
    print(f"Final ratio: {final_speech / max(1, final_noise):.2f} speech:noise")
    
    # Keep validation and test sets as is
    val_speech = sum(1 for item in valid_ if item['label'] == 'speech')
    val_noise = sum(1 for item in valid_ if item['label'] == 'noise')
    print(f"Validation set: {len(valid_)} samples ({val_speech} speech, {val_noise} noise)")
    
    test_speech = sum(1 for item in test_ if item['label'] == 'speech')
    test_noise = sum(1 for item in test_ if item['label'] == 'noise')
    print(f"Test set: {len(test_)} samples ({test_speech} speech, {test_noise} noise)")
    
    # Process the datasets with optimized functions
    print("Extracting features with optimized processing...")
    
    def preprocess_fast(dataset, name=""):
        """Faster preprocessing for binary classification"""
        features, labels = [], []
        label_map = {'speech': 1, 'noise': 0}
        
        from tqdm import tqdm
        for sample in tqdm(dataset, desc=f"Pre-processing {name}"):
            audio, sr = load_audio(sample["file_path"])
            
            # Apply simple augmentation if flagged (for noise duplicates)
            if sample.get('augment', False):
                augment_type = sample.get('augment_type')
                audio = augment_audio_simple(audio, sr, augment_type)
            
            # For noise files, create a few simple chunks
            if sample["label"] == "noise":
                # Simple chunking: up to 3 equal chunks for efficiency
                duration = len(audio) / sr
                chunk_count = min(3, int(duration))
                
                if chunk_count <= 1:
                    # For very short files, just use as is
                    norm_audio = normalize_audio_duration(audio, sr)
                    mfcc = extract_mfcc(norm_audio, sr)
                    features.append(mfcc)
                    labels.append(label_map[sample["label"]])
                else:
                    # Create equal-length chunks
                    chunk_length = len(audio) // chunk_count
                    for i in range(chunk_count):
                        start = i * chunk_length
                        end = start + chunk_length
                        chunk = audio[start:end]
                        
                        # Process chunk
                        norm_chunk = normalize_audio_duration(chunk, sr)
                        mfcc = extract_mfcc(norm_chunk, sr)
                        features.append(mfcc)
                        labels.append(label_map[sample["label"]])
            else:
                # Process speech samples normally
                norm_audio = normalize_audio_duration(audio, sr)
                mfcc = extract_mfcc(norm_audio, sr)
                features.append(mfcc)
                labels.append(label_map[sample["label"]])
        
        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y
    
    # Process datasets
    X_train, y_train = preprocess_fast(balanced_train, "train")
    X_val, y_val = preprocess_fast(valid_, "valid")
    X_test, y_test = preprocess_fast(test_, "test")
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Print final class distribution after preprocessing
    train_speech_final = (y_train == 1).sum().item()
    train_noise_final = (y_train == 0).sum().item()
    print(f"Final processed train set: {len(y_train)} samples "
          f"({train_speech_final} speech, {train_noise_final} noise)")
    print(f"Final processed ratio: {train_speech_final / max(1, train_noise_final):.2f} speech:noise")
    
    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'noise': 0, 'speech': 1}


def augment_audio_simple(audio, sr, augmentation_type=None):
    """
    Apply simple audio augmentation with lower computational cost.
    
    Args:
        audio (np.ndarray): Audio signal
        sr (int): Sample rate
        augmentation_type (str, optional): Specific augmentation to apply, 
                                         if None, a random one is selected
    
    Returns:
        np.ndarray: Augmented audio
    """
    import random
    import numpy as np
    
    # If no specific augmentation requested, randomly choose one
    if augmentation_type is None:
        augmentation_type = random.choice(['volume', 'noise', 'shift', 'none'])
    
    if augmentation_type == 'volume':
        # Simple volume change (faster than pitch shifting)
        volume_factor = random.uniform(0.75, 1.25)
        return audio * volume_factor
    
    elif augmentation_type == 'noise':
        # Add simple Gaussian noise
        noise_level = random.uniform(0.002, 0.01)  # Lower range for subtle effect
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    elif augmentation_type == 'shift':
        # Simple time shifting (faster than time stretching)
        shift_amount = int(random.uniform(-0.1, 0.1) * len(audio))
        if shift_amount > 0:
            # Shift right
            return np.pad(audio[:-shift_amount], (shift_amount, 0), mode='constant')
        else:
            # Shift left
            shift_amount = abs(shift_amount)
            return np.pad(audio[shift_amount:], (0, shift_amount), mode='constant')
    
    # No augmentation case or fallback
    return audio