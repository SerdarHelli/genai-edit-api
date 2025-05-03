import io

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from diffusers import StableDiffusionImg2ImgPipeline
import torch

from app.utils.image_utils import read_image_bytes
from app.utils.prompt_utils  import enhance_prompt

router = APIRouter()

logger.info(f"Loading pipeline: img2img")
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" ) 

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to(device)

logger.info(f"Pipeline img2img loaded successfully.")



@router.post("/level1")
async def level1_generate(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    similarity_level: float = Form(0.5)
):
    try:
        prompt = enhance_prompt(prompt)
        logger.info(f"Level 1 request received with similarity_level={similarity_level}")
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
    
