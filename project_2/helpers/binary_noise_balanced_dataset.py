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
    chunks = []
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
        chunks.append(chunk)
    
    return chunks


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


def create_balanced_word_dataset(data_path, batch_size=32, seed=42, 
                             samples_per_word=100, chunks_per_noise=3,
                             enforce_test_balance=True):
    """
    Creates a binary classification dataset with samples from each word class.
    Fixes the issue with test set balance and improves chunking strategy.
    
    Args:
        data_path (str/Path): Path to the dataset
        batch_size (int): Batch size for the data loaders
        seed (int): Random seed for reproducibility
        samples_per_word (int): Number of samples from each word class
        chunks_per_noise (int): Number of chunks to extract from each noise file
        enforce_test_balance (bool): Whether to ensure test set has both classes
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training/validation/testing
        id2label: Dictionary mapping from label ID to label name
    """
    import random
    import numpy as np
    import torch
    from pathlib import Path
    from collections import defaultdict
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from helpers.utils import extract_mfcc, load_audio, normalize_audio_duration
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data with noise files included
    print(f"Loading binary dataset from {data_path}...")
    data_path = Path(data_path)
    train_, valid_, test_ = load_from_path_with_noise(data_path)
    
    # Separate speech and noise samples
    train_speech = [item for item in train_ if item['label'] == 'speech']
    train_noise = [item for item in train_ if item['label'] == 'noise']
    
    print(f"Original train set: {len(train_)} samples "
          f"({len(train_speech)} speech, {len(train_noise)} noise)")
    
    # Group speech samples by original word label
    speech_by_word = defaultdict(list)
    for item in train_speech:
        speech_by_word[item['original_label']].append(item)
    
    # Sample from each word class
    sampled_speech = []
    for word, samples in speech_by_word.items():
        # Skip background noise class if it somehow got mixed in
        if word == '_background_noise_':
            continue
            
        # Take up to samples_per_word from each class
        take_count = min(len(samples), samples_per_word)
        sampled_speech.extend(random.sample(samples, take_count))
    
    print(f"Sampled speech: {len(sampled_speech)} total from {len(speech_by_word)} word classes")
    
    # Prepare validation and test sets FIRST to ensure balance
    valid_speech = [item for item in valid_ if item['label'] == 'speech']
    valid_noise = [item for item in valid_ if item['label'] == 'noise']
    
    test_speech = [item for item in test_ if item['label'] == 'speech']
    test_noise = [item for item in test_ if item['label'] == 'noise']
    
    # Use a moderate size for test and validation sets
    max_val_samples = min(500, len(valid_speech), len(test_speech))
    
    # Ensure we have at least some of each class
    if enforce_test_balance:
        if len(valid_noise) == 0 or len(test_noise) == 0:
            # If no noise in validation/test, borrow some from train
            print("WARNING: No noise in validation/test, borrowing from train")
            if len(valid_noise) == 0 and len(train_noise) > 0:
                valid_noise = random.sample(train_noise, min(5, len(train_noise)))
            if len(test_noise) == 0 and len(train_noise) > 0:
                test_noise = random.sample(train_noise, min(5, len(train_noise)))
    
    # Sample validation set
    sampled_valid_speech = random.sample(valid_speech, min(max_val_samples, len(valid_speech)))
    sampled_valid_noise = random.sample(valid_noise, min(max_val_samples // chunks_per_noise, len(valid_noise)))
    
    # Sample test set
    sampled_test_speech = random.sample(test_speech, min(max_val_samples, len(test_speech)))
    sampled_test_noise = random.sample(test_noise, min(max_val_samples // chunks_per_noise, len(test_noise)))
    
    # Ensure we have at least one of each class in test set
    if enforce_test_balance:
        if len(sampled_test_speech) == 0:
            print("WARNING: No speech in test set, adding from train")
            sampled_test_speech = random.sample(train_speech, min(5, len(train_speech)))
        if len(sampled_test_noise) == 0:
            print("WARNING: No noise in test set, adding from train")
            sampled_test_noise = random.sample(train_noise, min(5, len(train_noise)))
    
    # Calculate how many noise samples we need for train set
    target_train_noise = len(sampled_speech)
    
    # Calculate how many train noise files to process
    noise_files_needed = (target_train_noise // chunks_per_noise) + 1
    
    # Ensure we have enough noise files, duplicate if needed
    if noise_files_needed > len(train_noise):
        duplicated_noise = []
        copies_needed = noise_files_needed // len(train_noise) + 1
        
        for _ in range(copies_needed):
            duplicated_noise.extend(train_noise)
        
        sampled_train_noise = random.sample(duplicated_noise, noise_files_needed)
    else:
        sampled_train_noise = random.sample(train_noise, noise_files_needed)
    
    # Combine for datasets
    train_data = sampled_speech + sampled_train_noise
    valid_data = sampled_valid_speech + sampled_valid_noise
    test_data = sampled_test_speech + sampled_test_noise
    
    # Shuffle datasets
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)
    
    print(f"Train data: {len(train_data)} samples "
          f"({len(sampled_speech)} speech, {len(sampled_train_noise)} noise files)")
    print(f"Validation data: {len(valid_data)} samples "
          f"({len(sampled_valid_speech)} speech, {len(sampled_valid_noise)} noise files)")
    print(f"Test data: {len(test_data)} samples "
          f"({len(sampled_test_speech)} speech, {len(sampled_test_noise)} noise files)")
    
    # Fixed chunk_noise_file function with better explanation
    def chunk_noise_file(audio, sr, chunks=chunks_per_noise, max_var=0.05):
        """
        Extract up to <chunks> separate ~1-second segments from audio file
        
        Args:
            audio: Audio file data
            sr: Sample rate
            chunks: Maximum number of ~1s chunks to extract
            max_var: Maximum variation in chunk length (±5%)
            
        Returns:
            List of audio chunks (each ~1 second long)
        """
        chunks_list = []
        file_duration = len(audio) / sr
        
        # For very short files, just use as is
        if file_duration < 0.5:  # If less than half a second
            return [audio]
            
        # Determine how many chunks we can extract (at most 1 per second)
        max_possible_chunks = max(1, int(file_duration))
        n_chunks = min(max_possible_chunks, chunks)
        
        for _ in range(n_chunks):
            # Create a chunk of ~1 second with slight variation
            chunk_duration = 1.0 * (1.0 + random.uniform(-max_var, max_var))
            chunk_samples = int(sr * chunk_duration)
            
            # Ensure we don't exceed file length
            if chunk_samples >= len(audio):
                chunks_list.append(audio)
                continue
                
            # Pick a random starting point
            max_start = len(audio) - chunk_samples
            start = random.randint(0, max_start)
            
            # Extract chunk
            chunk = audio[start:start + chunk_samples]
            chunks_list.append(chunk)
            
        return chunks_list
    
    # Improved preprocessing function
    def preprocess_balanced_dataset(dataset, name="", target_noise_samples=None, 
                                   verbose=True, discard_extra=True):
        """Process dataset with improved noise handling"""
        features, labels = [], []
        label_map = {'speech': 1, 'noise': 0}
        noise_count = 0
        
        # Track class distribution
        class_counts = {"speech": 0, "noise": 0}
        
        for sample in tqdm(dataset, desc=f"Pre-processing {name}"):
            # Load audio
            audio, sr = load_audio(sample["file_path"])
            
            if sample["label"] == "noise":
                # Stop if we've reached target and discard_extra is True
                if target_noise_samples and noise_count >= target_noise_samples and discard_extra:
                    continue
                
                # Create chunks
                chunks = chunk_noise_file(audio, sr, chunks=chunks_per_noise)
                
                # Limit if needed
                if target_noise_samples and noise_count + len(chunks) > target_noise_samples and discard_extra:
                    chunks = chunks[:target_noise_samples - noise_count]
                
                # Process each chunk
                for chunk in chunks:
                    # Normalize duration (center or pad to 1 second)
                    norm_chunk = normalize_audio_duration(chunk, sr)
                    
                    # Extract features
                    mfcc = extract_mfcc(norm_chunk, sr)
                    
                    # Add to dataset
                    features.append(mfcc)
                    labels.append(label_map[sample["label"]])
                    noise_count += 1
                    class_counts["noise"] += 1
                    
                    # Stop if reached target
                    if target_noise_samples and noise_count >= target_noise_samples and discard_extra:
                        break
            else:
                # Process speech
                norm_audio = normalize_audio_duration(audio, sr)
                mfcc = extract_mfcc(norm_audio, sr)
                features.append(mfcc)
                labels.append(label_map[sample["label"]])
                class_counts["speech"] += 1
        
        # Print class distribution
        if verbose:
            print(f"{name} set: {class_counts['speech']} speech, {class_counts['noise']} noise")
        
        # Convert to tensors
        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y
    
    # Process datasets
    print("Extracting features...")
    X_train, y_train = preprocess_balanced_dataset(train_data, "train", 
                                                 target_noise_samples=len(sampled_speech))
    X_val, y_val = preprocess_balanced_dataset(valid_data, "validation")
    X_test, y_test = preprocess_balanced_dataset(test_data, "test")
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Double-check class balance
    train_speech_count = (y_train == 1).sum().item()
    train_noise_count = (y_train == 0).sum().item()
    val_speech_count = (y_val == 1).sum().item()
    val_noise_count = (y_val == 0).sum().item()
    test_speech_count = (y_test == 1).sum().item()
    test_noise_count = (y_test == 0).sum().item()
    
    print(f"Final train balance: {train_speech_count} speech, {train_noise_count} noise")
    print(f"Final validation balance: {val_speech_count} speech, {val_noise_count} noise")
    print(f"Final test balance: {test_speech_count} speech, {test_noise_count} noise")
    
    # Verify both classes exist in all datasets
    if enforce_test_balance:
        if train_speech_count == 0 or train_noise_count == 0:
            raise ValueError(f"Train set missing a class: {train_speech_count} speech, {train_noise_count} noise")
        if val_speech_count == 0 or val_noise_count == 0:
            raise ValueError(f"Validation set missing a class: {val_speech_count} speech, {val_noise_count} noise")
        if test_speech_count == 0 or test_noise_count == 0:
            raise ValueError(f"Test set missing a class: {test_speech_count} speech, {test_noise_count} noise")
    
    # Create TensorDatasets and DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'noise': 0, 'speech': 1}


def create_balanced_dataset_with_augmentation(data_path, batch_size=32, seed=42, 
                                     samples_per_word=100, chunks_per_noise=60,
                                     enforce_balance=True):
    """
    Creates a balanced binary classification dataset with aggressive noise augmentation.
    Ensures speech and noise classes are perfectly balanced in all splits.
    
    Args:
        data_path (str/Path): Path to the dataset
        batch_size (int): Batch size for the data loaders
        seed (int): Random seed for reproducibility
        samples_per_word (int): Number of samples to take from each word class
        chunks_per_noise (int): Maximum number of chunks per noise file
        enforce_balance (bool): Ensure exact class balance in all splits
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training/validation/testing
        id2label: Dictionary mapping from label ID to label name
    """
    import random
    import numpy as np
    import torch
    from pathlib import Path
    from collections import defaultdict
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from helpers.utils import extract_mfcc, load_audio, normalize_audio_duration
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data with noise files included
    print(f"Loading binary dataset from {data_path}...")
    data_path = Path(data_path)
    train_, valid_, test_ = load_from_path_with_noise(data_path)
    
    # Separate speech and noise samples
    train_speech = [item for item in train_ if item['label'] == 'speech']
    train_noise = [item for item in train_ if item['label'] == 'noise']
    
    valid_speech = [item for item in valid_ if item['label'] == 'speech']
    valid_noise = [item for item in valid_ if item['label'] == 'noise']
    
    test_speech = [item for item in test_ if item['label'] == 'speech']
    test_noise = [item for item in test_ if item['label'] == 'noise']
    
    print(f"Original train set: {len(train_)} samples "
          f"({len(train_speech)} speech, {len(train_noise)} noise)")
    print(f"Original validation set: {len(valid_)} samples "
          f"({len(valid_speech)} speech, {len(valid_noise)} noise)")
    print(f"Original test set: {len(test_)} samples "
          f"({len(test_speech)} speech, {len(test_noise)} noise)")
    
    # Combine all noise files to maximize available noise sources
    all_noise_files = train_noise + valid_noise + test_noise
    print(f"Total noise files available: {len(all_noise_files)}")
    
    # Group speech samples by original word label
    speech_by_word = defaultdict(list)
    for item in train_speech:
        speech_by_word[item['original_label']].append(item)
    
    # Sample from each word class 
    sampled_speech = []
    for word, samples in speech_by_word.items():
        # Skip background noise class if it got mixed in
        if word == '_background_noise_':
            continue
            
        # Take up to samples_per_word from each class
        take_count = min(len(samples), samples_per_word)
        sampled_speech.extend(random.sample(samples, take_count))
    
    print(f"Sampled speech: {len(sampled_speech)} total from {len(speech_by_word)} word classes")
    
    # For validation and test, use a fixed size
    max_val_test_speech = 500
    
    # Sample validation and test speech
    sampled_valid_speech = random.sample(valid_speech, min(max_val_test_speech, len(valid_speech)))
    sampled_test_speech = random.sample(test_speech, min(max_val_test_speech, len(test_speech)))
    
    # Function to augment audio
    def augment_audio(audio, sr, aug_type=None):
        """Apply basic audio augmentation"""
        import numpy as np
        
        if aug_type is None:
            aug_type = random.choice(['volume', 'noise', 'shift', 'reverse', 'pitch'])
        
        if aug_type == 'volume':
            # Volume change
            volume_factor = random.uniform(0.6, 1.4)
            return audio * volume_factor
        
        elif aug_type == 'noise':
            # Add random noise
            noise_level = random.uniform(0.001, 0.02)
            noise = np.random.normal(0, noise_level, len(audio))
            return audio + noise
        
        elif aug_type == 'shift':
            # Time shift
            shift_factor = random.uniform(-0.2, 0.2)
            shift_amount = int(len(audio) * shift_factor)
            if shift_amount > 0:
                # Shift right
                return np.pad(audio[:-shift_amount], (shift_amount, 0), mode='constant')
            else:
                # Shift left
                shift_amount = abs(shift_amount)
                return np.pad(audio[shift_amount:], (0, shift_amount), mode='constant')
        
        elif aug_type == 'reverse':
            # Time reversal
            return audio[::-1]
        
        elif aug_type == 'pitch':
            try:
                # Simple "fake" pitch shift using resampling and time normalization
                pitch_factor = random.uniform(0.85, 1.15)
                # Resample
                resampled = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * pitch_factor)),
                    np.arange(len(audio)),
                    audio
                )
                # Fix length to original
                if len(resampled) > len(audio):
                    return resampled[:len(audio)]
                else:
                    return np.pad(resampled, (0, len(audio) - len(resampled)), mode='constant')
            except:
                # Fall back to volume change if resampling fails
                return audio * random.uniform(0.7, 1.3)
            
        return audio
    
    # Modified chunk_noise_file to include augmentation
    def chunk_noise_file_with_aug(audio, sr, chunks=chunks_per_noise, max_var=0.05):
        """
        Extract chunks from audio file with augmentation
        
        Args:
            audio: Audio file data
            sr: Sample rate
            chunks: Maximum number of chunks to extract
            max_var: Maximum variation in chunk length
            
        Returns:
            List of audio chunks with augmentation
        """
        chunks_list = []
        file_duration = len(audio) / sr
        
        # For very short files, just use as is with augmentation
        if file_duration < 0.5:  # If less than half a second
            # Create multiple augmented versions
            for i in range(min(chunks, 5)):
                aug_audio = augment_audio(audio, sr)
                chunks_list.append(aug_audio)
            return chunks_list
            
        # Determine how many chunks we can extract
        max_possible_chunks = max(1, int(file_duration * 2))  # Allow more overlap
        n_chunks = min(max_possible_chunks, chunks)
        
        for i in range(n_chunks):
            # Create a chunk of ~1 second with slight variation
            chunk_duration = 1.0 * (1.0 + random.uniform(-max_var, max_var))
            chunk_samples = int(sr * chunk_duration)
            
            # Ensure we don't exceed file length
            if chunk_samples >= len(audio):
                aug_audio = augment_audio(audio, sr)
                chunks_list.append(aug_audio)
                continue
                
            # Pick a random starting point
            max_start = len(audio) - chunk_samples
            start = random.randint(0, max_start)
            
            # Extract chunk
            chunk = audio[start:start + chunk_samples]
            
            # Apply augmentation to every other chunk for variety
            if i % 2 == 1:
                chunk = augment_audio(chunk, sr)
                
            chunks_list.append(chunk)
            
        return chunks_list
    
    # Function to process noise for a specific split
    def process_noise_for_split(noise_files, target_count, name=""):
        """Process noise files to create exactly target_count samples"""
        if not noise_files:
            print(f"WARNING: No noise files for {name}!")
            return []
            
        features = []
        total_processed = 0
        
        # First round - process each file normally
        for sample in tqdm(noise_files, desc=f"Processing noise for {name} (round 1)"):
            audio, sr = load_audio(sample["file_path"])
            
            # Chunk the file
            chunks = chunk_noise_file_with_aug(audio, sr, chunks=min(10, chunks_per_noise))
            
            for chunk in chunks:
                # Normalize length
                norm_chunk = normalize_audio_duration(chunk, sr)
                
                # Extract MFCC
                mfcc = extract_mfcc(norm_chunk, sr)
                
                # Add to features
                features.append(mfcc)
                total_processed += 1
                
                # Check if we've reached target
                if total_processed >= target_count:
                    break
                    
            # Check if we've reached target
            if total_processed >= target_count:
                break
        
        # If we still need more, create heavily augmented versions
        if total_processed < target_count:
            print(f"Need more noise samples for {name} ({total_processed}/{target_count})")
            
            # Repeat with more aggressive augmentation
            round_num = 2
            while total_processed < target_count and round_num <= 10:
                print(f"Round {round_num} augmentation for {name} noise")
                
                # Re-process files with different augmentations
                for sample in tqdm(noise_files, desc=f"Processing noise for {name} (round {round_num})"):
                    audio, sr = load_audio(sample["file_path"])
                    
                    # Apply initial file-level augmentation
                    audio = augment_audio(audio, sr, aug_type=random.choice(['volume', 'noise', 'reverse']))
                    
                    # Chunk with augmentation
                    chunks = chunk_noise_file_with_aug(audio, sr, chunks=min(20, chunks_per_noise))
                    
                    for chunk in chunks:
                        # Apply additional chunk-level augmentation
                        chunk = augment_audio(chunk, sr, aug_type=random.choice(['shift', 'pitch', 'noise']))
                        
                        # Normalize length
                        norm_chunk = normalize_audio_duration(chunk, sr)
                        
                        # Extract MFCC
                        mfcc = extract_mfcc(norm_chunk, sr)
                        
                        # Add to features
                        features.append(mfcc)
                        total_processed += 1
                        
                        # Check if we've reached target
                        if total_processed >= target_count:
                            break
                            
                    # Check if we've reached target
                    if total_processed >= target_count:
                        break
                
                round_num += 1
        
        # If we still haven't reached target, just duplicate some existing features
        if total_processed < target_count:
            print(f"WARNING: Still need more noise samples for {name} ({total_processed}/{target_count})")
            print(f"Duplicating existing features to reach target")
            
            # How many more we need
            still_needed = target_count - total_processed
            
            # Duplicate existing features
            if features:
                additional = random.choices(features, k=still_needed)
                features.extend(additional)
        
        # Ensure we have exactly the right number
        if len(features) > target_count:
            features = features[:target_count]
            
        return features
    
    # Process speech samples
    def process_speech(speech_samples, name=""):
        """Process speech samples"""
        features = []
        
        for sample in tqdm(speech_samples, desc=f"Processing speech for {name}"):
            audio, sr = load_audio(sample["file_path"])
            
            # Normalize duration
            norm_audio = normalize_audio_duration(audio, sr)
            
            # Extract MFCC
            mfcc = extract_mfcc(norm_audio, sr)
            
            # Add to features
            features.append(mfcc)
            
        return features
    
    # Process the datasets
    print("Processing speech samples...")
    train_speech_features = process_speech(sampled_speech, "train")
    valid_speech_features = process_speech(sampled_valid_speech, "validation")
    test_speech_features = process_speech(sampled_test_speech, "test")
    
    print("Creating noise samples with augmentation...")
    # Use all available noise files for maximum variety
    train_noise_features = process_noise_for_split(all_noise_files, len(train_speech_features), "train")
    valid_noise_features = process_noise_for_split(all_noise_files, len(valid_speech_features), "validation")
    test_noise_features = process_noise_for_split(all_noise_files, len(test_speech_features), "test")
    
    # Create balanced datasets
    print("Creating final balanced datasets...")
    
    # Train set
    X_train = torch.tensor(np.stack(train_speech_features + train_noise_features), dtype=torch.float32)
    y_train = torch.tensor(
        [1] * len(train_speech_features) + [0] * len(train_noise_features), 
        dtype=torch.long
    )
    
    # Validation set
    X_val = torch.tensor(np.stack(valid_speech_features + valid_noise_features), dtype=torch.float32)
    y_val = torch.tensor(
        [1] * len(valid_speech_features) + [0] * len(valid_noise_features), 
        dtype=torch.long
    )
    
    # Test set
    X_test = torch.tensor(np.stack(test_speech_features + test_noise_features), dtype=torch.float32)
    y_test = torch.tensor(
        [1] * len(test_speech_features) + [0] * len(test_noise_features), 
        dtype=torch.long
    )
    
    # Shuffle datasets
    def shuffle_dataset(X, y):
        indices = torch.randperm(len(y))
        return X[indices], y[indices]
    
    X_train, y_train = shuffle_dataset(X_train, y_train)
    X_val, y_val = shuffle_dataset(X_val, y_val)
    X_test, y_test = shuffle_dataset(X_test, y_test)
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, {y_val.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    # Print class balance
    train_speech_count = (y_train == 1).sum().item()
    train_noise_count = (y_train == 0).sum().item()
    val_speech_count = (y_val == 1).sum().item()
    val_noise_count = (y_val == 0).sum().item()
    test_speech_count = (y_test == 1).sum().item()
    test_noise_count = (y_test == 0).sum().item()
    
    print(f"Final train balance: {train_speech_count} speech, {train_noise_count} noise")
    print(f"Final validation balance: {val_speech_count} speech, {val_noise_count} noise")
    print(f"Final test balance: {test_speech_count} speech, {test_noise_count} noise")
    
    # Create TensorDatasets and DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'noise': 0, 'speech': 1}