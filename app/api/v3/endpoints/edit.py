
import io

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import torch

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.utils.image_utils import read_image_bytes, to_canny, get_diff_mask
from app.utils.prompt_utils  import enhance_prompt


router = APIRouter()



logger.info(f"Loading pipeline: controlnet_dual")
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" ) 

canny_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
mlsd_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[canny_net, mlsd_net],
    torch_dtype=torch.float16
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
logger.info(f"Pipeline controlnet_dual loaded successfully")

@router.post("/level3")
async def level3_guided_edit(
    baseline: UploadFile = File(...),
    annotated: UploadFile = File(...),
    similarity_level: float = Form(0.5),
    prompt: str = Form("")
):
    try:
        logger.info(f"Level 3 request received | similarity={similarity_level}, prompt='{prompt}'")

        base_img = read_image_bytes(baseline)
        ann_img = read_image_bytes(annotated)
        prompt = enhance_prompt(prompt)

        # Generate control images
        canny_image = to_canny(ann_img).convert("RGB")
        diff_mask = get_diff_mask(base_img, ann_img).convert("RGB")

        # Get preloaded pipeline

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

