from diffusers import StableDiffusionImg2ImgPipeline,StableDiffusionInstructPix2PixPipeline,ControlNetModel,StableDiffusionControlNetImg2ImgPipeline
from loguru import logger
import torch

_img2img_pipe = None
_instruct_pipe = None
_controlnet_pipe = None

def get_img2img_pipeline():
    global _img2img_pipe
    if _img2img_pipe is None:
        logger.info("Loading img2img pipeline for v1...")
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to("cuda")
        logger.info("img2img pipeline loaded successfully.")
    return _img2img_pipe

def get_instruct_pix2pix_pipeline():
    global _instruct_pipe
    if _instruct_pipe is None:
        logger.info("Loading InstructPix2Pix pipeline for v2...")
        _instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16
        ).to("cuda")
        logger.info("InstructPix2Pix pipeline loaded successfully.")
    return _instruct_pipe


def get_controlnet_img2img_pipeline():
    global _controlnet_pipe
    if _controlnet_pipe is None:
        logger.info("Loading ControlNet Img2Img pipeline for v3...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        _controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")
        _controlnet_pipe.enable_model_cpu_offload()
        logger.info("ControlNet Img2Img pipeline loaded successfully.")
    return _controlnet_pipe
