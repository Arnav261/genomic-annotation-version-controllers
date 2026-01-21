"""
Middleware and dependency: basic API-key auth, request logging, and security headers.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
from app.database import SessionLocal, APIKey
from app.config import settings


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Expect header `X-API-Key`. Validate presence in DB and active flag.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Allow open paths (health, docs, static)
        path = request.url.path
        open_paths = ["/health", "/docs", "/openapi.json", "/", "/web"]
        if any(path.startswith(p) for p in open_paths):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key:
            return JSONResponse({"detail": "Missing API key"}, status_code=status.HTTP_401_UNAUTHORIZED)

        db = SessionLocal()
        try:
            key_obj = db.query(APIKey).filter(APIKey.key == api_key, APIKey.is_active == True).first()
            if not key_obj:
                return JSONResponse({"detail": "Invalid API key"}, status_code=status.HTTP_401_UNAUTHORIZED)
        finally:
            db.close()

        response = await call_next(request)
        # Add some security headers
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", "default-src 'self' 'unsafe-inline' https:;")
        return response