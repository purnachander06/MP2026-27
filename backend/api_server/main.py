import os
import sys
import subprocess
import argparse
from pathlib import Path

# 1. Automatic Virtual Environment Hand-off
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

# 2. Argument Parsing for Environment Selection
parser = argparse.ArgumentParser(description="IoT Threat Detection FastAPI Server")
parser.add_argument("--env", type=str, default=os.getenv("APP_ENV", "development"), choices=["development", "production"], help="Select environment: development (.env) or production (.env.production)")
parser.add_argument("--supabase", action="store_true", help="Shortcut to use production Supabase database")
parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
args, unknown = parser.parse_known_args()

if args.supabase:
    args.env = "production"

os.environ["APP_ENV"] = args.env

# 3. Load Selected Environment Variables
target_env = BACKEND_DIR / ".env.production" if args.env == "production" else BACKEND_DIR / ".env"
if target_env.exists():
    with open(target_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v

# 4. Path Setup
for path in [str(BACKEND_DIR), str(ROOT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

if "MODELS_DIR" not in os.environ:
    models_path = BACKEND_DIR / "models"
    if models_path.exists():
        os.environ["MODELS_DIR"] = str(models_path)

import uvicorn

def main():
    print("=" * 65)
    print("  IoT Threat Detection & LLM Explanation Service - Backend API")
    print("=" * 65)
    print(f"  Environment       : {args.env.upper()} ({target_env.name})")
    print(f"  Python Runtime    : {sys.executable}")
    print(f"  Backend Directory : {BACKEND_DIR}")
    print(f"  Models Directory  : {os.environ.get('MODELS_DIR')}")
    print(f"  Host              : http://{args.host}:{args.port}")
    print(f"  Swagger Docs      : http://{args.host}:{args.port}/docs")
    print("=" * 65)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info",
        app_dir=str(BACKEND_DIR),
    )

if __name__ == "__main__":
    main()