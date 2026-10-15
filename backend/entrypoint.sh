#!/bin/sh
set -e

echo "[Entrypoint] Starting IoT Threat Detection API container..."

# Check if migrations should run
if [ "$RUN_MIGRATIONS" != "false" ]; then
    echo "[Entrypoint] Running database migrations (alembic upgrade head)..."
    alembic upgrade head
    echo "[Entrypoint] Database migrations applied successfully."
fi

# Execute the primary process passed via CMD
echo "[Entrypoint] Launching ASGI server on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}..."
exec "$@"
