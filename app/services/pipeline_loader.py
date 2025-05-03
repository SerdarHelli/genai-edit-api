from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from loguru import logger
import torch


class LazyPipelineLoader:
    def __init__(self):
        self._pipelines = {}

    def get(self, name: str):
        if name not in self._pipelines:
            load_method = getattr(self, f"_load_{name}", None)
            if load_method is None:
                raise ValueError(f"No loader defined for pipeline '{name}'")
            logger.info(f"Loading pipeline: {name}")
            self._pipelines[name] = load_method()
            logger.info(f"Pipeline '{name}' loaded successfully.")
        return self._pipelines[name]

    # === Loaders ===

    def _load_img2img(self):
        return StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to("cuda")

    def _load_instruct(self):
        return StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16
        ).to("cuda")

    def _load_controlnet_dual(self):
        canny_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        mlsd_net = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[canny_net, mlsd_net],
            torch_dtype=torch.float16
        ).to("cuda")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        return pipe
