#!/usr/bin/env python3
"""
Draw the confusion-matrix stored in a chain-of-responsibility result JSON.

Usage
-----
python plot_chain_cm.py  /path/to/chain_result_20250505_213501.json  \
        --labels down,go,left,no,off,on,right,silence,stop,unknown,up,yes  \
        --outfile chain_cm.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot chain confusion matrix")
    parser.add_argument("json_path", type=Path,
                        help="Path to chain_result_<timestamp>.json")
    parser.add_argument("--labels", type=str, default="",
                        help="Comma-separated list of class names "
                             "(leave empty to label 0…N-1)")
    parser.add_argument("--outfile", type=Path, default=None,
                        help="If given, save PNG instead of showing interactively")
    args = parser.parse_args()

    # ─── load JSON ─────────────────────────────────────────────────────
    with args.json_path.open() as f:
        meta = json.load(f)

    cm = np.asarray(meta["chain_confusion_matrix"])
    n = cm.shape[0]

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != n:
            raise ValueError(f"{len(labels)} labels supplied but matrix is {n}x{n}")
    else:
        labels = [str(i) for i in range(n)]

    acc = meta.get("chain_test_acc", None)

    # ─── plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("predicted")
    plt.ylabel("true")
    title = "Chain confusion matrix"
    if acc is not None:
        title += f" (acc {acc:.3%})"
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if args.outfile:
        args.outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.outfile, dpi=150)
        print("saved to", args.outfile)
    else:
        plt.show()


if __name__ == "__main__":
    main()
