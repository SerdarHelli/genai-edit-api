from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
import io

from app.utils.image_utils import read_image_bytes, to_canny, get_diff_mask
from app.services.pipeline_loader import get_controlnet_img2img_pipeline

def get_router() -> APIRouter:
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

            base_img = read_image_bytes(baseline)
            ann_img = read_image_bytes(annotated)

            # Generate control images
            canny_image = to_canny(ann_img).convert("RGB")
            diff_mask = get_diff_mask(base_img, ann_img).convert("RGB")

            # Get preloaded pipeline
            pipe = get_controlnet_img2img_pipeline()

            # Run inference
            result = pipe(
                prompt=prompt,
                image=base_img,
                control_image=[canny_image, diff_mask],
                strength=1 - similarity_level,
                guidance_scale=8.5,
                num_inference_steps=30
            )

            # Return result
            buf = io.BytesIO()
            result.images[0].save(buf, format="JPEG")
            buf.seek(0)

            logger.info("Level 3 image generated successfully.")
            return StreamingResponse(buf, media_type="image/jpeg")

        except Exception:
            logger.exception("Error during Level 3 guided generation")
            raise HTTPException(status_code=500, detail="Guided edit generation failed")

    return router
