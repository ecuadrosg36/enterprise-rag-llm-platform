"""
FastAPI Application.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from src.core.config import get_config
from src.core.logger import setup_logger
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
        redoc_url="/redoc"
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Logging Middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Metrics
    setup_metrics(app)
    
    # Global Exception Handler
    @app.exception_handler(RAGError)
    async def rag_exception_handler(request: Request, exc: RAGError):
        return JSONResponse(
            status_code=500,  # Could map specific errors to 4xx
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
                "details": exc.details
            }
        )
    
    # Include Routers
    app.include_router(health.router)
    app.include_router(rag.router)
    
    return app


app = create_app()
