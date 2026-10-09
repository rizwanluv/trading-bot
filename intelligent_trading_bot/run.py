#!/usr/bin/env python3
"""
===============================================================================
  PERSONAL INTELLIGENT TRADING BOT - Single Entry Point
===============================================================================
  This is the main runner for your combined project.
  All original Python modules + personal config are already inside this folder.

  Usage examples:
      python run.py download
      python run.py merge
      python run.py features
      python run.py labels
      python run.py train
      python run.py predict
      python run.py signals
      python run.py output
      python run.py simulate
      python run.py predict_rolling
      python run.py server          # start live service
      python run.py all             # run full offline pipeline
===============================================================================
"""

import sys
import os
import argparse
from pathlib import Path

# Make sure the project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/config-personal-1min.jsonc"


def run_module(module_name: str, config: str):
    """Dynamically run a scripts.* or service.* module with -c config"""
    import runpy
    sys.argv = [module_name, "-c", config]
    runpy.run_module(module_name, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(
        description="Personal Intelligent Trading Bot - Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=[
            "download", "merge", "features", "labels",
            "train", "predict", "signals", "output",
            "simulate", "predict_rolling", "server", "all"
        ],
        help="Which part of the pipeline to run"
    )
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})"
    )

    args = parser.parse_args()
    config_path = args.config

    if not Path(config_path).exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print("Please edit configs/config-personal-1min.jsonc first.")
        sys.exit(1)

    print(f"→ Using config : {config_path}")
    print(f"→ Command      : {args.command}")
    print("-" * 60)

    if args.command == "all":
        # Full offline pipeline in correct order
        steps = [
            "scripts.download",
            "scripts.merge",
            "scripts.features",
            "scripts.labels",
            "scripts.train",
            "scripts.predict",
            "scripts.signals",
            "scripts.output",
        ]
        for step in steps:
            print(f"\n>>> Running {step} ...")
            run_module(step, config_path)
        print("\n✅ Full offline pipeline finished.")
        return

    # Map short command → full module
    module_map = {
        "download": "scripts.download",
        "merge": "scripts.merge",
        "features": "scripts.features",
        "labels": "scripts.labels",
        "train": "scripts.train",
        "predict": "scripts.predict",
        "signals": "scripts.signals",
        "output": "scripts.output",
        "simulate": "scripts.simulate",
        "predict_rolling": "scripts.predict_rolling",
        "server": "service.server",
    }

    module = module_map[args.command]
    run_module(module, config_path)


if __name__ == "__main__":
    main()
