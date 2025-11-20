"""
Logging Middleware.

Handles request/response logging, correlation IDs, and timing.
"""

import time
import uuid
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.logger import setup_logger, correlation_id_var


logger = setup_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request logging.
    
    - Generates/Extracts Correlation ID
    - Logs Request Details (Method, Path, Client IP)
    - Logs Response Details (Status, Duration)
    - Handles Request/Response Body Logging (optional, use with care)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_body: bool = False,
        max_body_size: int = 1024
    ):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Setup Correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id_var.set(request_id)
        
        start_time = time.time()
        
        # 2. Log Request
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Optional: Log Request Body (careful with PII/Secrets)
        # Note: Reading body in middleware consumes the stream. 
        # FastAPI/Starlette doesn't make this easy without caching.
        # For now, we skip body logging to avoid stream consumption issues
        # or use a specialized library if needed.
        
        logger.info(f"Request started: {request.method} {request.url.path}", extra=log_data)
        
        try:
            # 3. Process Request
            response = await call_next(request)
            
            # 4. Log Response
            duration_ms = (time.time() - start_time) * 1000
            
            log_data.update({
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            })
            
            logger.info(
                f"Request completed: {response.status_code}",
                extra=log_data
            )
            
            # Add Correlation ID to Response Headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # 5. Log Failure
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {e}",
                extra={
                    "error": str(e),
                    "duration_ms": duration_ms,
                    **log_data
                }
            )
            raise
