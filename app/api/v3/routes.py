from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
from loguru import logger
from app.services.pipeline_loader import get_controlnet_img2img_pipeline
from app.utils.image_utils import read_image_bytes, extract_edges

router = APIRouter(prefix="/edit")

@router.post("/level3")
async def level3_guided_edit(
    baseline: UploadFile = File(...),
    annotated: UploadFile = File(...),
    similarity_level: float = Form(0.8),
    prompt: str = Form("")
):
    try:
        logger.info(f"Level 3 request received | similarity={similarity_level}, prompt='{prompt}'")
        pipe = get_controlnet_img2img_pipeline()

        base_img = read_image_bytes(baseline)
        ann_img = read_image_bytes(annotated)

        control_image = extract_edges(ann_img)

        result = pipe(
            prompt=prompt,
            image=base_img,
            control_image=control_image,
            strength=1 - similarity_level,
            guidance_scale=7.5,
            num_inference_steps=30
        )

        buf = io.BytesIO()
        result.images[0].save(buf, format="JPEG")
        buf.seek(0)
        logger.info("Level 3 image generated successfully.")
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception:
        logger.exception("Error in Level 3 guided edit")
        raise HTTPException(status_code=500, detail="Guided edit generation failed")
    
def get_router() -> APIRouter:
    return router