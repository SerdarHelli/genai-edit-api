from fastapi import APIRouter
from app.api.v1.endpoints import edit

def get_router() -> APIRouter:
    router = APIRouter()
    router.include_router(edit.router, prefix="/edit", tags=["Level 1"])
    return router
