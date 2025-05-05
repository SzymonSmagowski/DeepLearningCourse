#!/usr/bin/env python3
"""
Compare HPO results for multiple architectures and tasks.

• Expects the directory layout exactly as printed in the prompt:
      results/<ARCH>/<TASK>/best_config.json

• Produces:
  – Four PNG bar-charts named  task1_arch_comparison.png … task4_arch_comparison.png
  – A console dump with every architecture’s winning config for every task.

If you want the plots to pop up interactively, remove the line that calls
plt.savefig() and leave the plt.show() line uncommented.
"""

import json
import pprint
import pathlib
import sys
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt


###############################################################################
# Helpers
###############################################################################
def load_best_config(json_path: pathlib.Path) -> Tuple[dict, float]:
    """
    Parse a best_config.json file and return
        (chosen_config_dict, chosen_test_acc_float)

    Handles both of the formats described in the prompt.
    """
    with json_path.open() as f:
        data = json.load(f)

    # Format (a) – single config
    if "config" in data and "result" in data:
        return data["config"], float(data["result"]["test_acc"])

    # Format (b) – best + larger
    if "best_config" in data and "best_result" in data:
        best_cfg   = data["best_config"]
        best_acc   = float(data["best_result"]["test_acc"])
        larger_cfg = data.get("larger_config")
        larger_acc = float(data.get("larger_result", {}).get("test_acc", -1))

        if larger_cfg is not None and larger_acc > best_acc:
            return larger_cfg, larger_acc
        else:
            return best_cfg, best_acc

    # Unknown schema
    raise ValueError(f"Unrecognised schema in {json_path}")


###############################################################################
# Main logic
###############################################################################
def main(results_root: pathlib.Path = pathlib.Path("results")) -> None:
    if not results_root.exists():
        sys.exit(f"Directory {results_root} not found.")

    tasks = [f"task{i}" for i in range(1, 5)]

    # Discover architecture folders (top-level dirs in results/)
    architectures = sorted(
        d.name for d in results_root.iterdir() if d.is_dir()
    )

    # Collect accuracies and configs
    # results[task][arch] = (acc, cfg)
    results: Dict[str, Dict[str, Tuple[float, dict]]] = {t: {} for t in tasks}

    for arch in architectures:
        for task in tasks:
            json_file = results_root / arch / task / "best_config.json"
            if not json_file.exists():
                print(f"⚠️  Missing {json_file}", file=sys.stderr)
                continue
            cfg, acc = load_best_config(json_file)
            results[task][arch] = (acc, cfg)

    ###########################################################################
    # Plotting
    ###########################################################################
    for task in tasks:
        archs = list(results[task].keys())
        accs  = [results[task][a][0] for a in archs]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(archs, accs)
        plt.ylabel("Test accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")

        # Annotate the bars with the accuracy value
        for bar, score in zip(bars, accs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        out_name = f"output/{task}_arch_comparison.png"
        plt.savefig(out_name, dpi=150)
        # plt.show()   # ← uncomment if you prefer interactive windows
        print(f"✅  Saved {out_name}")

    ###########################################################################
    # Console dump of chosen configs
    ###########################################################################
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False, sort_dicts=False)
    for task in tasks:
        print("\n", "=" * 80, sep="")
        print(f"{task.upper()}")
        for arch in architectures:
            if arch not in results[task]:
                continue
            acc, cfg = results[task][arch]
            print(f"\n▶ {arch}  (test_acc = {acc:.4f})")
            pp.pprint(cfg)


###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit("Usage: python compare_results.py [RESULTS_DIR]")
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) == 2 else pathlib.Path("results")
    main(root)
