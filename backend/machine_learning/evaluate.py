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
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

for path in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import scripts.run_final_experiments as runner

if __name__ == "__main__":
    print("=" * 65)
    print("  Evaluating All Machine Learning and Federated Baselines")
    print("=" * 65)
    runner.main()