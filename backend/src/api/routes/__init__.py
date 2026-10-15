"""Route package exports for FastAPI service."""

from src.api.routes import health, predictions, explanations, analyze, history, analytics

__all__ = ["health", "predictions", "explanations", "analyze", "history", "analytics"]
