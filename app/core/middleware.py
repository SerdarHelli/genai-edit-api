import uuid

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


def setup_middlewares(app: FastAPI):
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add logging middleware
    @app.middleware("http")
    async def enrich_logs_with_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        user_agent = request.headers.get("User-Agent", "unknown")
        user_id = request.headers.get("X-User-ID", "anonymous")

        try:
            with logger.contextualize(
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                user_agent=user_agent,
                user_id=user_id
            ):
                response = await call_next(request)
                return response

        except Exception as exc:
            logger.exception(f"Middleware-level exception: {type(exc).__name__} - {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error (middleware)",
                    "error": f"{type(exc).__name__}: {str(exc)}"
                }
            )
