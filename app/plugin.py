
from fastapi import FastAPI

from app.api.v1.routes import get_router as get_router_v1
from app.api.v2.routes import get_router as get_router_v2
from app.api.v3.routes import get_router as get_router_v3

def register_all_versions(app: FastAPI):

# plugin.py
    app.include_router(get_router_v1(), prefix="/api/v1", tags=["Level 1"])
    app.include_router(get_router_v2(), prefix="/api/v2", tags=["Level 2"])
    app.include_router(get_router_v3(), prefix="/api/v3", tags=["Level 3"])


