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

FL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = FL_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

for path in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

def find_path(rel_path: str) -> Path:
    candidates = [ROOT_DIR / rel_path, BACKEND_DIR / rel_path, FL_DIR / rel_path]
    for p in candidates:
        if p.exists():
            return p
    return ROOT_DIR / rel_path

from src.federated.config import FederatedConfig
from src.federated.train import run_federated_experiment

def main():
    parser = argparse.ArgumentParser(description="Standard FedAvg Federated Training")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients")
    args = parser.parse_args()

    cfg = FederatedConfig(num_rounds=args.rounds, num_clients=args.clients)
    cfg.train_path = find_path("processed/train.csv")
    cfg.val_path = find_path("processed/val.csv")
    cfg.test_path = find_path("processed/test.csv")
    cfg.metadata_path = find_path("processed/feature_metadata.json")
    cfg.models_dir = BACKEND_DIR / "models" / "federated"
    cfg.experiments_dir = ROOT_DIR / "experiments" / "federated" / "fedavg"

    print("=" * 65)
    print("  Running Standard FedAvg Training")
    print("=" * 65)
    run_federated_experiment(cfg)
    print("\n[SUCCESS] FedAvg checkpoint saved to models/federated/fedavg_model.pt")

if __name__ == "__main__":
    main()