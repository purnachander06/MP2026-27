"""FastAPI application configuration and settings with automatic .env loader."""

import os
from pathlib import Path
from typing import List

# Automatically load .env from backend directory if present
def _load_dotenv():
    backend_dir = Path(__file__).resolve().parent.parent.parent
    app_env = os.getenv("APP_ENV", "").lower().strip()
    env_file = backend_dir / ".env.production" if app_env == "production" else backend_dir / ".env"
    if not env_file.exists():
        env_file = backend_dir / ".env"
    
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v

_load_dotenv()

SUPABASE_DEFAULT = "postgresql+asyncpg://postgres.mxiyhuombyyfjytvwfgk:Sup%40base112@aws-0-ap-southeast-2.pooler.supabase.com:5432/postgres?ssl=require"

class ApiConfig:
    """Application settings read from environment variables."""

    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", 8000))
    DEBUG: bool = os.getenv("API_DEBUG", "false").lower() in ("true", "1")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    DEFAULT_DETECTOR: str = os.getenv("DEFAULT_DETECTOR", "random_forest")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", 100))

    # CORS Allowed Origins
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000",
        ).replace("[", "").replace("]", "").replace('"', "").replace("'", "").split(",")
        if origin.strip()
    ]

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", 15.0))

    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", SUPABASE_DEFAULT)
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", 5))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", 5))
    DATABASE_POOL_TIMEOUT: float = float(os.getenv("DATABASE_POOL_TIMEOUT", 15.0))
    DATABASE_ECHO: bool = os.getenv("DATABASE_ECHO", "false").lower() in ("true", "1")


settings = ApiConfig()