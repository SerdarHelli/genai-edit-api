from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import traceback

def setup_exception_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def python_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        user_agent = request.headers.get("User-Agent", "unknown")
        user_id = request.headers.get("X-User-ID", "anonymous")

        try:
            body = await request.body()
            decoded_body = body.decode("utf-8", errors="ignore")
        except Exception:
            decoded_body = "<unavailable>"

        with logger.contextualize(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            user_agent=user_agent,
            user_id=user_id
        ):
            traceback_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.error(f"Unhandled Exception: {type(exc).__name__} - {exc}\nBody: {decoded_body}\n{traceback_str}")

        return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            body = await request.body()
            decoded_body = body.decode("utf-8", errors="ignore")
        except Exception:
            decoded_body = "<unavailable>"

        with logger.contextualize(request_id=request_id):
            logger.error(f"Validation error: {exc.errors()} | Body: {decoded_body}")

        return JSONResponse(status_code=422, content={"detail": exc.errors()})
