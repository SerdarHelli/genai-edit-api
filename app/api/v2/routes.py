from fastapi import APIRouter
from app.api.v2.endpoints import edit

def get_router() -> APIRouter:
    router = APIRouter()
    router.include_router(edit.router, prefix="/edit", tags=["Level 2"])
    return router
