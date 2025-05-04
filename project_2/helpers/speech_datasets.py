# speech_datasets.py
# -----------------------------------------------------------------------------
# Tiny, dependency-light helpers for Google Speech-Commands v0.01.
#
# Public helpers:
#   get_task1_dataloaders  – mutlicclass + unknown + silence
#   get_task2_dataloaders  – silence vs command
#   get_task3_dataloaders  – known vs unknown (silence dropped)
#   get_task4_dataloaders  – pure multi-class (known commands only)
#
# All four share the same call signature:
#
#   train_dl, val_dl, test_dl, label2id = get_taskX_dataloaders(
#       data_dir      = "~/datasets/speech_commands_v0.01",
#       known_commands= ["yes","no","up","down","left","right","on","off","stop","go"],
#       batch_size    = 128,
#       seed          = 42,
#       balance_val   = False,
#   )
#
# -----------------------------------------------------------------------------
import os, random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import librosa
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------#
#                             global constants                                 #
# -----------------------------------------------------------------------------#
TARGET_SR      = 16_000         # dataset native sample-rate
AUDIO_DURATION = 1.0            # seconds   (all speech clips are ≈1 s)

# -----------------------------------------------------------------------------#
#                               audio helpers                                  #
# -----------------------------------------------------------------------------#
def load_audio(path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Return mono float32 waveform & sr, resampling to target_sr if needed."""
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr  = target_sr
    return wav.astype(np.float32), sr


def normalize_audio_duration(audio: np.ndarray, sr: int,
                             duration: float = AUDIO_DURATION) -> np.ndarray:
    """Centre-crop (or zero-pad) waveform to exactly <duration> seconds."""
    frame_len = int(sr * duration)
    if len(audio) >= frame_len:                       # crop – keep loudest 1 s
        energy = librosa.util.frame(audio, frame_length=frame_len, hop_length=1)
        start  = np.argmax(np.sum(energy**2, axis=0))
        audio  = audio[start:start + frame_len]
    else:                                             # pad
        pad  = frame_len - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    return audio


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Return MFCC array (n_mfcc × frames) as float32."""
    return librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=512, hop_length=160, fmin=20, fmax=sr // 2
    ).astype(np.float32)

# -----------------------------------------------------------------------------#
#                         background-noise chunking                            #
# -----------------------------------------------------------------------------#
def chunk_noise_file(audio: np.ndarray,
                     sr: int,
                     max_var: float= 0.08,
                     rng: random.Random | None = None) -> List[np.ndarray]:
    """
    Slice a long noise wav into up to <chunks> segments of ≈1 s
    with slight ±max_var length variation.
    """
    if rng is None:
        rng = random
    out: List[np.ndarray] = []

    n_chunks = int(len(audio) / (sr * AUDIO_DURATION))
    if n_chunks < 1:                           # shorter than 1 s – use whole clip
        return [audio]

    for _ in range(n_chunks):
        var_factor   = 1.0 + rng.uniform(-max_var, max_var)
        chunk_len    = int(sr * AUDIO_DURATION * var_factor)
        max_start    = max(0, len(audio) - chunk_len)
        start_sample = 0 if max_start == 0 else rng.randint(0, max_start)
        out.append(audio[start_sample:start_sample + chunk_len])

    return out

def _augment_audio(audio: np.ndarray,
                   rng: random.Random,
                   sr: int = TARGET_SR) -> np.ndarray:
    """Cheap augmentation for *silence*."""
    aug_type = rng.choice(["gain", "noise", "shift", "none"])

    if aug_type == "gain":                           # ±25 % loudness
        return audio * rng.uniform(0.75, 1.25)

    if aug_type == "noise":                          # low-level Gaussian noise
        std = rng.uniform(0.002, 0.01)               # choose SNR each call
        noise = np.random.normal(0.0, std, size=audio.shape)
        return audio + noise

    if aug_type == "shift":                          # ≤±50 ms circular shift
        max_shift = int(0.05 * sr)
        s = rng.randint(-max_shift, max_shift)
        return np.roll(audio, s)

    return audio


def _prep_silence_datasets(
        noise_paths: Sequence[Path],
        label2id: Dict[str, int],
        *,
        rng:   random.Random,
        chunks_per_file: int = 0,   # 0 ⇒ use “all possible” chunks
        aug_factor: int = 5         # how many augmented copies per chunk
) -> dict[str, TensorDataset]:
    """
    Turn long `_background_noise_` WAVs into TensorDatasets for
    **train / val / test** in a 50 / 25 / 25 split, with augmentations
    applied *after* the split.
    """
    # 1) make ≈1 s chunks from every noise WAV -------------------------------
    raw_chunks: List[np.ndarray] = []
    for p in noise_paths:
        wav, sr = load_audio(p)
        chunks  = chunk_noise_file(wav, sr, rng=rng)
        if chunks_per_file:
            chunks = rng.sample(chunks, min(len(chunks), chunks_per_file))
        raw_chunks.extend(chunks)

    rng.shuffle(raw_chunks)
    n = len(raw_chunks)
    n_train, n_val = int(0.5 * n), int(0.25 * n)
    split_idxs = dict(
        train=raw_chunks[:n_train],
        val=  raw_chunks[n_train:n_train + n_val],
        test= raw_chunks[n_train + n_val:]
    )

    # 2) augment *inside* each split ----------------------------------------
    out: dict[str, TensorDataset] = {}
    silence_id = label2id["silence"]

    for split, chunk_list in split_idxs.items():
        feats, labels = [], []
        for ch in chunk_list:
            for _ in range(1 + aug_factor):      # original + aug copies
                ch_aug = _augment_audio(ch, rng)
                ch_aug = normalize_audio_duration(ch_aug, sr)
                mfcc   = extract_mfcc(ch_aug, sr)
                feats.append(mfcc)
                labels.append(silence_id)
        X = torch.tensor(np.stack(feats), dtype=torch.float32)
        y = torch.tensor(labels,         dtype=torch.long)
        out[split] = TensorDataset(X, y)

    return out

# -----------------------------------------------------------------------------#
#                         miscellaneous utilities                              #
# -----------------------------------------------------------------------------#
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _list_wavs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.wav")]


_CACHE: Dict[str, List[str]] = {"val": [], "test": []}

def _read_split_lists(root: Path):
    if not _CACHE["val"]:
        _CACHE["val"]  = (root / "validation_list.txt").read_text().splitlines()
        _CACHE["test"] = (root / "testing_list.txt").read_text().splitlines()


def _split_official(root: Path, wav: Path) -> str:
    rel = wav.relative_to(root).as_posix()
    if rel in _CACHE["test"]:
        return "test"
    if rel in _CACHE["val"]:
        return "val"
    return "train"


def _balance_tensor_dataset(ds: TensorDataset, seed: int) -> TensorDataset:
    """
    Down-sample classes so every label appears ``min_count`` times.
    """
    X, y = ds.tensors
    y_np = y.numpy()
    class_to_idx: Dict[int, List[int]] = {}
    for idx, cls in enumerate(y_np):
        class_to_idx.setdefault(int(cls), []).append(idx)

    min_count = min(len(v) for v in class_to_idx.values())
    rng = random.Random(seed)
    keep: List[int] = []
    for idxs in class_to_idx.values():
        keep.extend(rng.sample(idxs, min_count))
    keep.sort()

    X_bal = X[keep]
    y_bal = y[keep]
    return TensorDataset(X_bal, y_bal)

# -----------------------------------------------------------------------------#
#                        core pre-loading / building                           #
# -----------------------------------------------------------------------------#
def _preload_split(paths: Sequence[Path],
                   label_func,
                   label2id: Dict[str, int],
                   desc: str,
                   seed: int) -> TensorDataset:
    """
    Convert every wav file in <paths> into one or many MFCC tensors.
    """
    feats, labels = [], []
    rng = random.Random(seed)

    for p in tqdm(paths, desc=f"loading {desc}", leave=False):
        is_noise = p.parent.name == "_background_noise_"
        wav, sr  = load_audio(p)

        if is_noise:                                 # break long file into 1 s bits
            for chunk in chunk_noise_file(wav, sr, rng=rng):
                chunk = normalize_audio_duration(chunk, sr)
                mfcc  = extract_mfcc(chunk, sr)
                feats.append(mfcc)
                labels.append(label2id[label_func(p)])
        else:                                        # normal 1-s speech clip
            wav   = normalize_audio_duration(wav, sr)
            mfcc  = extract_mfcc(wav, sr)
            feats.append(mfcc)
            labels.append(label2id[label_func(p)])

    X = torch.tensor(np.stack(feats), dtype=torch.float32)
    y = torch.tensor(labels,        dtype=torch.long)
    return TensorDataset(X, y)

# -----------------------------------------------------------------------------#
#                           master builder                                     #
# -----------------------------------------------------------------------------#
def _make_dataloaders(
        sc_root     : Path,
        label_func,                              # maps Path → cls OR "__skip__"
        classes     : Sequence[str],
        *,
        batch_size  : int,
        seed        : int,
        balance_val : bool,
        noise_aug   : int  = 10,                  # copies per raw silence chunk
        noise_cpf   : int  = 0                  # chunks_per_file (0 = all)
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:

    _seed_all(seed)
    rng      = random.Random(seed)
    label2id = {c: i for i, c in enumerate(sorted(classes))}

    speech_paths, noise_paths = [], []
    for wav in _list_wavs(sc_root):
        if wav.parent.name == "_background_noise_":
            noise_paths.append(wav)
        else:
            cls = label_func(wav)
            if cls != "__skip__" and cls in label2id:
                speech_paths.append((wav, cls))

    # ------------ build speech TensorDatasets via official split ------------#
    split_to_paths = {"train": [], "val": [], "test": []}
    for wav, cls in speech_paths:
        split_to_paths[_split_official(sc_root, wav)].append(wav)

    ds_speech = {
        split: _preload_split(
            paths, label_func, label2id,
            split, seed
        )
        for split, paths in split_to_paths.items()
    }

    # ------------- build silence TensorDatasets w/ 70-15-15 -----------------#
    ds_silence = (
        _prep_silence_datasets(
            noise_paths, label2id,
            rng=rng,
            chunks_per_file=noise_cpf,
            aug_factor=noise_aug
        ) if "silence" in classes else {}
    )

    # -------------------- concatenate where relevant ------------------------#
    def _concat(ds_a: TensorDataset, ds_b: TensorDataset | None):
        if ds_b is None:
            return ds_a
        Xa, ya = ds_a.tensors
        Xb, yb = ds_b.tensors
        return TensorDataset(torch.cat([Xa, Xb]), torch.cat([ya, yb]))

    ds_train = _concat(ds_speech["train"], ds_silence.get("train"))
    ds_val   = _concat(ds_speech["val"],   ds_silence.get("val"))
    ds_test  = _concat(ds_speech["test"],  ds_silence.get("test"))

    # balance train (and optionally val)
    ds_train = _balance_tensor_dataset(ds_train, seed)
    if balance_val and len(ds_val) > 0:
        ds_val = _balance_tensor_dataset(ds_val, seed)

    def _dl(ds, shuf):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuf)

    return _dl(ds_train, True), _dl(ds_val, False), _dl(ds_test, False), label2id

# -----------------------------------------------------------------------------#
#                        public task-specific helpers                          #
# -----------------------------------------------------------------------------#
def get_task1_dataloaders(*,
        data_dir      : str | Path,
        known_commands: Sequence[str],
        batch_size    : int = 64,
        seed          : int = 42,
        balance_val   : bool = False,
):
    """
    Task 1: multiclass + `unknown` + `silence`
    """
    sc_root = Path(data_dir).expanduser()
    _read_split_lists(sc_root)

    silence, unknown = "silence", "unknown"
    classes          = list(known_commands) + [unknown, silence]

    def mapper(p: Path):
        if p.parent.name == "_background_noise_":
            return silence
        return p.parent.name if p.parent.name in known_commands else unknown

    return _make_dataloaders(
        sc_root, mapper, classes,
        batch_size=batch_size, seed=seed,
        balance_val=balance_val 
    )


def get_task2_dataloaders(*,
        data_dir    : str | Path,
        batch_size  : int = 64,
        seed        : int = 42,
        balance_val : bool = False,
):
    """
    Task 2: binary *silence vs command*
    """
    sc_root = Path(data_dir).expanduser()
    _read_split_lists(sc_root)

    classes = ["command", "silence"]

    def mapper(p: Path):
        return "silence" if p.parent.name == "_background_noise_" else "command"

    return _make_dataloaders(
        sc_root, mapper, classes,
        batch_size=batch_size, seed=seed,
        balance_val=balance_val
    )


def get_task3_dataloaders(*,
        data_dir      : str | Path,
        known_commands: Sequence[str],
        batch_size    : int = 64,
        seed          : int = 42,
        balance_val   : bool = False,
):
    """
    Task 3: binary *known vs unknown* (silence dropped)
    """
    sc_root = Path(data_dir).expanduser()
    _read_split_lists(sc_root)

    classes = ["known", "unknown"]

    def mapper(p: Path):
        if p.parent.name == "_background_noise_":
            return "__skip__"
        return "known" if p.parent.name in known_commands else "unknown"

    return _make_dataloaders(
        sc_root, mapper, classes,
        batch_size=batch_size, seed=seed,
        balance_val=balance_val
    )


def get_task4_dataloaders(*,
        data_dir      : str | Path,
        known_commands: Sequence[str],
        batch_size    : int = 64,
        seed          : int = 42,
        balance_val   : bool = False,
):
    """
    Task 4: multi-class classification over *known_commands* only.
    """
    sc_root = Path(data_dir).expanduser()
    _read_split_lists(sc_root)

    classes = list(known_commands)

    def mapper(p: Path):
        return p.parent.name if p.parent.name in known_commands else "__skip__"

    return _make_dataloaders(
        sc_root, mapper, classes,
        batch_size=batch_size, seed=seed,
        balance_val=balance_val
    )
