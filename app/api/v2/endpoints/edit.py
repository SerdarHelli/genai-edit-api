import io

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.utils.prompt_utils  import enhance_prompt
from app.utils.image_utils import read_image_bytes

router = APIRouter()



device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" ) 

logger.info(f"Loading pipeline: instruct_pipe")

instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16
        ).to(device)
logger.info(f"Pipeline instruct_pipe loaded successfully.")




@router.post("/level2")
async def level2_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    similarity_level: float = Form(0.5)
):
    try:
        logger.info(f"Level 2 request received with prompt='{prompt}', similarity_level={similarity_level}")
        prompt = enhance_prompt(prompt)
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
    
