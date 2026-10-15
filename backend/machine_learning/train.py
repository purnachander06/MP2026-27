import os
import sys
import subprocess
from pathlib import Path

# Automatic Virtual Environment Hand-off
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

venv_py = None
for parent in [CURRENT_DIR, BACKEND_DIR, ROOT_DIR]:
    cand = parent / ".venv" / "Scripts" / "python.exe"
    if cand.exists():
        venv_py = cand.resolve()
        break

if venv_py and Path(sys.executable).resolve() != venv_py:
    cmd = [str(venv_py)] + sys.argv
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        sys.exit(0)
import os
import sys
import argparse
from pathlib import Path

# Automatically inject backend directory and workspace root into sys.path
ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

for path in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Resolve dataset paths (supports processed/ at root or inside backend)
def find_path(rel_path: str) -> Path:
    candidates = [
        ROOT_DIR / rel_path,
        BACKEND_DIR / rel_path,
        ML_DIR / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return ROOT_DIR / rel_path

from src.detection.train import train_and_evaluate_baseline

def main():
    parser = argparse.ArgumentParser(description="Centralized Machine Learning Baseline Training (IoT-23)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-tune-threshold", action="store_true", help="Skip ROC threshold tuning")
    args = parser.parse_args()

    train_csv = find_path("processed/train.csv")
    val_csv = find_path("processed/val.csv")
    test_csv = find_path("processed/test.csv")
    full_csv = find_path("processed/iot23_sampled.csv")
    models_dir = BACKEND_DIR / "models"
    experiments_dir = ROOT_DIR / "experiments" / "baseline"

    print("=" * 65)
    print("  Machine Learning Training: Centralized Baselines (RF + LR)")
    print("=" * 65)
    print(f"  Training Data   : {train_csv}")
    print(f"  Validation Data : {val_csv}")
    print(f"  Test Data       : {test_csv}")
    print(f"  Models Output   : {models_dir}")
    print("=" * 65)

    if not train_csv.exists():
        print(f"Error: Training dataset not found at {train_csv}.")
        print("Please verify the processed/ folder exists in the project root.")
        sys.exit(1)

    train_and_evaluate_baseline(
        train_path=train_csv,
        val_path=val_csv,
        test_path=test_csv,
        full_dataset_path=full_csv,
        models_dir=models_dir,
        experiments_dir=experiments_dir,
        random_seed=args.seed,
        tune_threshold=not args.no_tune_threshold,
    )
    print("\n[SUCCESS] Machine learning training completed successfully!")
    print(f"Serialized pipelines saved to: {models_dir}")

if __name__ == "__main__":
    main()