from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from app.services.pipeline_loader import get_img2img_pipeline
from app.utils.image_utils import read_image_bytes
import io
from loguru import logger


router = APIRouter()

@router.post("/level1")
async def level1_generate(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    similarity_level: float = Form(0.8)
):
    try:
        logger.info(f"Level 1 request received with similarity_level={similarity_level}")
        pipe = get_img2img_pipeline()
        input_img = read_image_bytes(image)
        result = pipe(prompt=prompt, image=input_img, strength=1 - similarity_level)
        buf = io.BytesIO()
        result.images[0].save(buf, format="JPEG")
        buf.seek(0)
        logger.info("Level 1 image generated successfully.")
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        logger.exception("Error in Level 1 generation")
        raise HTTPException(status_code=500, detail="Image generation failed")
    
