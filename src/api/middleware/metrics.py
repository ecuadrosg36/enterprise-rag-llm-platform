"""
Metrics Middleware.

Prometheus metrics collection.
"""

from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI


def setup_metrics(app: FastAPI):
    """
    Initialize Prometheus metrics.
    
    Exposes /metrics endpoint.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health", "/docs", "/redoc"],
        inprogress_name="rag_api_inprogress",
        inprogress_labels=True,
    )
    
    # Add standard metrics
    instrumentator.instrument(app)
    
    # Expose endpoint
    instrumentator.expose(app, include_in_schema=True, tags=["Observability"])
