from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import traceback


def get_request_context(request: Request) -> dict:
    """Extract contextual logging information from the request."""
    return {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "path": str(request.url.path),
        "method": request.method,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "user_id": request.headers.get("X-User-ID", "anonymous"),
    }


async def read_request_body(request: Request) -> str:
    try:
        return (await request.body()).decode("utf-8", errors="ignore")
    except Exception:
        return "<unavailable>"


async def python_exception_handler(request: Request, exc: Exception):
    context = get_request_context(request)
    body = await read_request_body(request)
    traceback_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    with logger.contextualize(**context):
        logger.error(
            f"Unhandled Exception: {type(exc).__name__} - {exc}\n"
            f"Body: {body}\n"
            f"{traceback_str}"
        )

    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    context = get_request_context(request)
    body = await read_request_body(request)

    with logger.contextualize(**context):
        logger.error(f"Validation error | Errors: {exc.errors()} | Body: {body}")

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
