from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.services.pipeline_loader import LazyPipelineLoader
from app.utils.image_utils import read_image_bytes
import io
from loguru import logger

router = APIRouter()
loader = LazyPipelineLoader()

@router.post("/level2")
async def level2_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    similarity_level: float = Form(0.8)
):
    try:
        logger.info(f"Level 2 request received with prompt='{prompt}', similarity_level={similarity_level}")
        instruct_pipe = loader.get("instruct")
        input_img = read_image_bytes(image)
        result = instruct_pipe(prompt=prompt, image=input_img, strength=1 - similarity_level)
        buf = io.BytesIO()
        result.images[0].save(buf, format="JPEG")
        buf.seek(0)
        logger.info("Level 2 image edited successfully.")
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception:
        logger.exception("Error in Level 2 editing")
        raise HTTPException(status_code=500, detail="Image editing failed")
    
