#!/usr/bin/env python3
"""
Diagnostic figures for TASK-1
• left = train / val loss curves
• right = confusion-matrix heat-map
The figure itself has NO title; the winning config is printed to stdout.
"""

import json
import pathlib
import sys
import os
import pprint
from typing import Dict, Any, List

import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────
def load(path: pathlib.Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def base_cfg(best: Dict[str, Any]) -> Dict[str, Any]:
    """Return the hyper-parameter set that was part of the search (not the upscaled)."""
    return best["config"] if "config" in best else best["best_config"]


def base_test_acc(best: Dict[str, Any]) -> float:
    """Return the test-accuracy associated with that base config."""
    return float(best["result"]["test_acc"] if "result" in best
                 else best["best_result"]["test_acc"])


def find_run(task_dir: pathlib.Path, cfg: Dict[str, Any]) -> pathlib.Path:
    """Locate *one* search-run JSON whose ['config'] matches cfg."""
    for p in task_dir.glob("*.json"):
        if p.name == "best_config.json":
            continue
        try:
            if load(p).get("config") == cfg:
                return p
        except json.JSONDecodeError:
            continue
    raise FileNotFoundError("No matching run found")


# ────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────
def plot_loss(ax, hist):
    """train / val loss on the given axes"""
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], label="train loss")
    ax.plot(epochs, hist["val_loss"],   label="val loss")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)


def plot_acc(ax, hist):
    """train / val accuracy on the given axes"""
    epochs = range(1, len(hist["train_acc"]) + 1)
    ax.plot(epochs, hist["train_acc"], "--", label="train acc")
    ax.plot(epochs, hist["val_acc"],   "--", label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("accuracy curves")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)

def plot_cm(ax, cm: List[List[int]], labels: List[str]):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("confusion matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    vmax = max(map(max, cm))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j],
                    ha="center", va="center",
                    color="white" if cm[i][j] > vmax * 0.6 else "black",
                    fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main(root: pathlib.Path):
    tasks = [f"task{i}" for i in range(1, 5)]
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False, sort_dicts=False)

    # make the top-level output directory
    out_root = pathlib.Path("output/history_and_cm")
    out_root.mkdir(exist_ok=True)

    for task_name in tasks:
        (out_root / task_name).mkdir(exist_ok=True)

        for arch_dir in sorted(root.iterdir()):
            if not arch_dir.is_dir():
                continue

            task_dir = arch_dir / task_name
            best_fp  = task_dir / "best_config.json"
            if not best_fp.exists():
                continue  # this architecture did not run this task

            best_json = load(best_fp)
            cfg       = base_cfg(best_json)
            t_acc     = base_test_acc(best_json)

            try:
                run_fp = find_run(task_dir, cfg)
            except FileNotFoundError:
                print(f"⚠️  {arch_dir.name} / {task_name}: base run not found")
                continue

            run = load(run_fp)
            hist = run.get("history")
            cm   = run.get("confusion_matrix")
            id2l = run.get("id2label")

            if not hist or cm is None or not id2l:
                print(f"⚠️  {arch_dir.name} / {task_name}: missing history or CM")
                continue

            # ─── figure ────────────────────────────────────────────────
            fig = plt.figure(figsize=(14, 8))
            gs  = fig.add_gridspec(2, 2, width_ratios=[3, 3])

            ax_loss = fig.add_subplot(gs[0, 0])
            ax_acc  = fig.add_subplot(gs[1, 0], sharex=ax_loss)
            ax_cm   = fig.add_subplot(gs[:, 1])

            plot_loss(ax_loss, hist)
            plot_acc(ax_acc,   hist)

            labels = [lbl for lbl, idx in sorted(id2l.items(), key=lambda x: x[1])]
            plot_cm(ax_cm, cm, labels)

            plt.tight_layout()
            out_path = out_root / task_name / f"{task_name}_{arch_dir.name}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()

            # ─── console output ───────────────────────────────────────
            print(f"\n▶ {task_name} • {arch_dir.name}  (test_acc = {t_acc:.4f})")
            pp.pprint(cfg)
            print(f"✅  saved {out_path}")


# entry-point -----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit("usage: python diagnostics_all_tasks.py [RESULTS_DIR]")
    results_root = pathlib.Path(sys.argv[1]) if len(sys.argv) == 2 else pathlib.Path("results")
    main(results_root)
