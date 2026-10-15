"""Main FastAPI application for IoT Threat Detection & LLM Explanation Service."""

import logging
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.routes import health, predictions, explanations, analyze, history, analytics

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("IoTThreatApi")


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"IoT Threat Detection & Explanation API started. "
        f"Default detector: '{settings.DEFAULT_DETECTOR}', LLM Provider: '{settings.LLM_PROVIDER}'. "
        f"Interactive documentation at http://{settings.HOST}:{settings.PORT}/docs"
    )
    yield
    logger.info("IoT Threat Detection & Explanation API shutdown.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="IoT Threat Detection & LLM Explanation Service",
        description=(
            "Research-grade, production-minded API connecting IoT-23 intrusion detection models "
            "(Centralized Random Forest, Logistic Regression, FedAvg, FedOpt, GSFS) to "
            "LLM-assisted, evidence-grounded cybersecurity threat explanations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS for explicit trusted origins and regex for all local development ports
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing & correlation middleware
    @app.middleware("http")
    async def add_timing_and_correlation(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Global unhandled exception handler to prevent leaking stack traces
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred while processing the request.",
                "path": request.url.path,
            },
        )

    # Include API Routers
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(explanations.router)
    app.include_router(analyze.router)
    app.include_router(history.router)
    app.include_router(analytics.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
