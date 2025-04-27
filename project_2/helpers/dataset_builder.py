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
from collections import Counter
from sklearn.model_selection import train_test_split


class DatasetBuilder:
    """
    Fluent builder for Speech-Commands splits **with label-id preprocessing**.
    """
    def __init__(self, train, val, test, seed=0):
        self._orig   = dict(train=copy.deepcopy(train),
                            val=copy.deepcopy(val),
                            test=copy.deepcopy(test))
        self._train  = self._orig["train"]
        self._val    = self._orig["val"]
        self._test   = self._orig["test"]
        self.seed    = seed

        # cfg
        self._trim_ratio      = None
        self._unknown_ratio   = None
        self._labels_keep     = None
        self._drop_nonlisted  = False
        self._include_unknown = True
        self._pending_noise   = False

    # ---------------- configuration ---------------- #
    def trim(self, ratio: float):                         # keep %
        self._trim_ratio = ratio;               return self

    def unknown_ratio(self, ratio: float):                # target %
        self._unknown_ratio = ratio;           return self

    def keep_labels(self, labels, make_rest_unknown=True):
        self._labels_keep    = set(labels)
        self._drop_nonlisted = not make_rest_unknown
        return self

    def include_unknown(self, flag: bool):
        self._include_unknown = flag;          return self

    def add_background_noise(self, *args, **kwargs):
        self._pending_noise = True;            return self  # stub

    # ---------------- internals -------------------- #
    def _apply_trim(self, split):
        if self._trim_ratio is None:
            return split
        X = [e["file_path"] for e in split]
        y = [e["label"]     for e in split]
        Xk, _, yk, _ = train_test_split(
            X, y, stratify=y, test_size=1-self._trim_ratio,
            random_state=self.seed)
        return [dict(file_path=p, label=l) for p, l in zip(Xk, yk)]

    def _apply_keep_labels(self, split):
        if self._labels_keep is None:
            return split
        out = []
        for e in split:
            if e["label"] in self._labels_keep:
                out.append(e)
            elif not self._drop_nonlisted:          # convert to unknown
                e = copy.deepcopy(e)
                e["label"] = "unknown"
                out.append(e)
        return out

    def _apply_unknown_ratio(self, split):
        if self._unknown_ratio is None:
            return split
        known   = [e for e in split if e["label"] != "unknown"]
        unknown = [e for e in split if e["label"] == "unknown"]
        tgt = int(len(known)*self._unknown_ratio /
                  max(1e-6, 1-self._unknown_ratio))
        if tgt >= len(unknown):
            return known + unknown
        random.seed(self.seed)
        return known + random.sample(unknown, tgt)

    def _maybe_drop_unknown(self, split):
        return split if self._include_unknown else [
            e for e in split if e["label"] != "unknown"
        ]

    def _preprocess_labels(self, split):
        """Add old_label / label_id, remap unknowns."""
        unique = sorted({e["label"] for e in split})
        self.id2label = {i: lbl for i, lbl in enumerate(unique)}
        self.label2id = {lbl: i for i, lbl in self.id2label.items()}
        for e in split:
            e["old_label"] = e["label"]
            idx = self.label2id.get(e["label"])
            e["label_id"]  = idx
            e["label"]     = self.id2label[idx]
        return split

    # ---------------- build ------------------------ #
    def build(self, verbose=True):
        splits = dict(train=self._train, val=self._val, test=self._test)

        for name, split in splits.items():
            split[:] = self._apply_keep_labels(split)
            split[:] = self._apply_trim(split)
            split[:] = self._apply_unknown_ratio(split)
            split[:] = self._maybe_drop_unknown(split)
            split[:] = self._preprocess_labels(split)

            if verbose:
                cnt = Counter(e["label"] for e in split)
                print(f"{name:>5}: {len(split):4d}  | {dict(cnt)}")

        if self._pending_noise:
            print("⚠️  add_background_noise() not implemented yet – stub only.")

        return splits["train"], splits["val"], splits["test"]
