#!/usr/bin/env python3
"""
Parameter-influence analysis

Creates box-plots of test-accuracy versus every varying hyper-parameter
for each (task, architecture) pair found under ./results.

Output PNGs:  output/param_influence_images/<task>/<task>_<ARCH>_<param>.png
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_json(p: Path):
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return None


def extract_config_and_acc(rec: dict):
    """Return (config, test_acc) if present, else (None, None)."""
    if "config" in rec and "test_acc" in rec:
        return rec["config"], rec["test_acc"]
    if "config" in rec and "result" in rec and "test_acc" in rec["result"]:
        return rec["config"], rec["result"]["test_acc"]
    return None, None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sort_labels(labels):
    """
    Try to interpret each label as a float; if successful, sort numerically.
    Otherwise fall back to alphabetical ordering of the original strings.
    """
    def key_fn(lbl):
        try:
            return float(lbl)
        except ValueError:
            return lbl  # string sort

    return sorted(labels, key=key_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(results_root: Path = Path("results"),
         out_root: Path = Path("output/param_influence_images"),
         max_boxes: int = 15):

    tasks = [f"task{i}" for i in range(1, 5)]
    sns.set_theme(style="whitegrid")

    for task in tasks:
        task_out_dir = out_root / task
        ensure_dir(task_out_dir)

        for arch_dir in sorted(results_root.iterdir()):
            if not arch_dir.is_dir():
                continue

            runs_dir = arch_dir / task
            if not runs_dir.exists():
                continue

            # Collect accuracies per parameter value
            buckets = defaultdict(lambda: defaultdict(list))
            for jfile in runs_dir.glob("*.json"):
                if jfile.name == "best_config.json":
                    continue
                rec = load_json(jfile)
                if not rec:
                    continue
                cfg, acc = extract_config_and_acc(rec)
                if cfg is None:
                    continue
                for k, v in cfg.items():
                    buckets[k][str(v)].append(acc)

            # Build a box-plot for every parameter that varies
            for param, values_dict in buckets.items():
                if len(values_dict) < 2 or len(values_dict) > max_boxes:
                    continue

                labels_sorted = sort_labels(list(values_dict.keys()))
                data_sorted   = [values_dict[lbl] for lbl in labels_sorted]

                plt.figure(figsize=(max(8, 1.2 * len(labels_sorted)), 6))
                sns.boxplot(data=data_sorted)
                plt.xticks(ticks=np.arange(len(labels_sorted)),
                           labels=labels_sorted,
                           rotation=45, ha="right", fontsize=8)

                plt.xlabel(param)           # ← x-axis label (no title)
                plt.ylabel("test accuracy")

                out_path = task_out_dir / f"{task}_{arch_dir.name}_{param}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

                total_runs = sum(len(v) for v in data_sorted)
                print(f"✅  {out_path}  (n={total_runs})")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        sys.exit("usage: python param_influence.py [RESULTS_DIR]")
    root_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("results")
    main(root_dir)
