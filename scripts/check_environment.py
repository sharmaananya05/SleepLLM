"""
scripts/check_environment.py
================================
A quick sanity-check script: run this FIRST on a new machine (or after
setting up a fresh conda env) to confirm PyTorch, CUDA, and all required
packages are correctly installed, before running any training.

Usage:
    python scripts/check_environment.py
"""

import sys

def main() -> None:
    print("=" * 60)
    print("SleepLLM environment check")
    print("=" * 60)

    print(f"Python version: {sys.version.split()[0]}")

    try:
        import torch
        print(f"torch version:  {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:            {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM:           {props.total_memory / 1e9:.1f} GB")
    except ImportError:
        print("torch: NOT INSTALLED -- run: pip install -r requirements.txt")
        return

    required_packages = ["transformers", "yaml", "pytest", "Levenshtein", "datasets"]
    print("\nChecking required packages:")
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg} -- run: pip install -r requirements.txt")

    print("\nChecking project structure:")
    import os
    required_dirs = ["config", "models", "memory", "sleep", "distillation", "dreaming", "trainer", "evaluation", "utils", "tests"]
    for d in required_dirs:
        status = "OK" if os.path.isdir(d) else "MISSING"
        print(f"  [{status}] {d}/")

    print("=" * 60)
    print("If everything above says OK, you're ready to run:")
    print("  pytest tests/ -v")
    print("  python main.py train --steps 20")
    print("=" * 60)


if __name__ == "__main__":
    main()
