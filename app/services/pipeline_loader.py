from diffusers import StableDiffusionImg2ImgPipeline,StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
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
    """
    Lazy-load and cache a dual-ControlNet Stable Diffusion Img2Img pipeline
    using Canny and MLSD maps for visual guidance.
    
    Returns:
        StableDiffusionControlNetImg2ImgPipeline: the initialized pipeline.
    """
    global _controlnet_pipe

    if _controlnet_pipe is None:
        logger.info("Loading dual-ControlNet Img2Img pipeline (Canny + MLSD)...")

        canny_net = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        mlsd_net = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        )

        _controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[canny_net, mlsd_net],
            torch_dtype=torch.float16
        ).to("cuda")

        _controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(_controlnet_pipe.scheduler.config)
        _controlnet_pipe.enable_model_cpu_offload()
        _controlnet_pipe.enable_xformers_memory_efficient_attention()

        logger.info("Dual-ControlNet Img2Img pipeline loaded and ready.")

    return _controlnet_pipe
