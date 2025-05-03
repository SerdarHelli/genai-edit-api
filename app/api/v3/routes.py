from fastapi import APIRouter
from app.api.v3.endpoints import edit

def get_router() -> APIRouter:
    router = APIRouter()
    router.include_router(edit.router, prefix="/edit", tags=["Level 3"])
    return router
