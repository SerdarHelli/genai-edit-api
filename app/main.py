from fastapi import FastAPI
from loguru import logger

from app.plugin import register_all_versions
from app.core.middleware import setup_middlewares
from app.core.exceptation_handlers import setup_exception_handlers

# Configure global logging
logger.add(
    "./logs.log",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message} | {extra}",
    level="INFO",
    enqueue=True,
    catch=True
)

app = FastAPI(
    title="SerdarHelli",
    description="SerdarHelli's assessments",
    version="1.0.0"
)

# Setup middlewares and exceptions
setup_middlewares(app)
setup_exception_handlers(app)
register_all_versions(app)

@app.get("/")
async def root():
    return {
        "message": "Multi editing SerdarHelli's API",
        "levels": ["/api/v1/edit", "/api/v2/edit", "/api/v3/edit"]
    }

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for uptime monitoring.
    Returns 200 OK if the service is alive.
    """
    return {"status": "ok"}


#uvicorn app.main:app --host 0.0.0.0 --port 7860