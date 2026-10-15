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
    candidates = [
        ROOT_DIR / rel_path,
        BACKEND_DIR / rel_path,
        FL_DIR / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return ROOT_DIR / rel_path

from src.federated.config import FederatedConfig, get_fedopt_config, get_gsfs_config
from src.federated.train_gsfs import run_gsfs_experiment
from src.federated.train import run_federated_experiment
from src.federated.train_fedopt import run_fedopt_experiment

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training for IoT Security")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="gsfs", 
        choices=["gsfs", "fedavg", "fedopt"],
        help="Federated algorithm to run: gsfs (default, basepaper proposed), fedavg, or fedopt"
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds (default: 5)")
    parser.add_argument("--clients", type=int, default=4, help="Number of edge IoT clients (default: 4)")
    args = parser.parse_args()

    train_csv = find_path("processed/train.csv")
    val_csv = find_path("processed/val.csv")
    test_csv = find_path("processed/test.csv")
    meta_json = find_path("processed/feature_metadata.json")

    print("=" * 65)
    print(f"  Federated Learning Training - Algorithm: {args.algorithm.upper()}")
    print("=" * 65)
    print(f"  Rounds           : {args.rounds}")
    print(f"  Clients          : {args.clients}")
    print(f"  Training Dataset : {train_csv}")
    print(f"  Models Output    : {BACKEND_DIR / 'models' / 'federated'}")
    print("=" * 65)

    if not train_csv.exists():
        print(f"Error: Dataset not found at {train_csv}.")
        sys.exit(1)

    if args.algorithm == "gsfs":
        cfg = get_gsfs_config(num_rounds=args.rounds, num_clients=args.clients)
        cfg.train_path = train_csv
        cfg.val_path = val_csv
        cfg.test_path = test_csv
        cfg.metadata_path = meta_json
        cfg.models_dir = BACKEND_DIR / "models" / "federated"
        cfg.experiments_dir = ROOT_DIR / "experiments" / "federated" / "gsfs"
        run_gsfs_experiment(cfg)
    elif args.algorithm == "fedopt":
        cfg = get_fedopt_config(num_rounds=args.rounds, num_clients=args.clients)
        cfg.train_path = train_csv
        cfg.val_path = val_csv
        cfg.test_path = test_csv
        cfg.metadata_path = meta_json
        cfg.models_dir = BACKEND_DIR / "models" / "federated"
        cfg.experiments_dir = ROOT_DIR / "experiments" / "federated" / "fedopt"
        run_fedopt_experiment(cfg)
    else:
        cfg = FederatedConfig(num_rounds=args.rounds, num_clients=args.clients)
        cfg.train_path = train_csv
        cfg.val_path = val_csv
        cfg.test_path = test_csv
        cfg.metadata_path = meta_json
        cfg.models_dir = BACKEND_DIR / "models" / "federated"
        cfg.experiments_dir = ROOT_DIR / "experiments" / "federated" / "fedavg"
        run_federated_experiment(cfg)

    print("\n[SUCCESS] Federated learning training completed!")
    print(f"Trained checkpoints saved to: {BACKEND_DIR / 'models' / 'federated'}")

if __name__ == "__main__":
    main()