#!/usr/bin/env python3
"""
colabify_configs.py

Usage:
    python colabify_configs.py --prefix /content/drive/MyDrive/project3
    # optional:  --config-dir path/to/configs   (defaults to ./configs)

It walks the config directory, patches every YAML file so it is Colab-friendly,
and overwrites them in place.
"""
import argparse
import os
import sys
from pathlib import Path

try:
    import yaml       # pip install pyyaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

def patch_cfg(path: Path, prefix: Path) -> None:
    """Load YAML, fix values, and write back."""
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    # 1. Force CUDA
    cfg["device"] = "cuda"

    # 2. Prepend prefix to every *_dir key
    for key, val in list(cfg.items()):
        if key.endswith("_dir") and isinstance(val, str):
            if not Path(val).as_posix().startswith(prefix.as_posix()):
                cfg[key] = os.path.join(prefix, val)

    # 3. Save (keep original key order – PyYAML ≥ 6.0)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main() -> None:
    p = argparse.ArgumentParser(description="Make YAML configs Colab-friendly")
    p.add_argument("--prefix", required=True,
                   help="Prefix inside MyDrive, e.g. /content/drive/MyDrive/project3")
    p.add_argument("--config-dir", default="configs",
                   help="Directory to scan for YAML files (default: ./configs)")
    args = p.parse_args()

    prefix = Path(args.prefix).expanduser()
    cfg_dir = Path(args.config_dir).expanduser()

    if not cfg_dir.is_dir():
        sys.exit(f"{cfg_dir} is not a directory")

    yaml_paths = [p for p in cfg_dir.rglob("*") if p.suffix in {".yml", ".yaml"}]
    if not yaml_paths:
        sys.exit(f"No YAML files found in {cfg_dir}")

    for yp in yaml_paths:
        patch_cfg(yp, prefix)
        print(f"[OK] patched {yp.relative_to(cfg_dir.parent)}")

if __name__ == "__main__":
    main()

