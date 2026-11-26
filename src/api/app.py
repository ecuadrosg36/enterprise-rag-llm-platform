"""
FastAPI Application.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_config
from src.core.logger import setup_logger, correlation_id_var
from src.core.errors import RAGError
from src.api.routes import health, rag
from src.api.middleware import RequestLoggingMiddleware, setup_metrics

# Setup logger
logger = setup_logger(__name__)

# Load config
config = get_config()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description="Enterprise RAG Platform API",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Fail closed: a production API must not run without a key.
    if not os.getenv("RAG_API_KEY"):
        if config.environment == "prod":
            raise RuntimeError("RAG_API_KEY must be set when ENV=prod.")
        logger.warning("RAG_API_KEY is not set: /rag and /embed are open (dev only).")

    # CORS: only the origins listed in CORS_ORIGINS (comma-separated); none by default.
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )

    # Logging Middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Metrics
    setup_metrics(app)

    # Global Exception Handler
    @app.exception_handler(RAGError)
    async def rag_exception_handler(request: Request, exc: RAGError):
        # Details stay in the logs: provider errors can echo request data or config.
        logger.error(f"{exc.__class__.__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": exc.__class__.__name__,
                "message": "Request failed. Quote request_id when reporting it.",
                "request_id": correlation_id_var.get(),
            },
        )

    # Include Routers
    app.include_router(health.router)
    app.include_router(rag.router)

    return app


app = create_app()
