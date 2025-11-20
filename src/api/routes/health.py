"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from src.core.config import Config
from src.api.dependencies import get_settings
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Config = Depends(get_settings)):
    """
    Basic health check endpoint.
    Returns service status and version.
    """
    return HealthResponse(
        status="ok", version=settings.app_version, environment=settings.environment
    )
