# Multi-stage or optimized production build for FastAPI backend
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for healthchecks, libpq for postgres compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy database migration files and configurations
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic

# Copy source code and trained model checkpoints
COPY src /app/src
COPY models /app/models
COPY entrypoint.sh /app/entrypoint.sh

# Create non-root user and assign permissions
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    chmod +x /app/entrypoint.sh && \
    chown -R appuser:appuser /app

USER appuser

# Expose default API port
EXPOSE 8000

# Container healthcheck using standard Python library to avoid curl dependency issues
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
