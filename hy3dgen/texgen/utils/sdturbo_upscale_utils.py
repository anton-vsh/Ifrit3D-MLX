# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# Generative alternative to RealESRGAN_Upscaler (realesrgan_utils.py): reuses
# the SD Turbo checkpoint already bundled at models/sd-turbo (same one used
# for text-to-image and the existing input-image upscale feature in app.py),
# rather than a plain restoration CNN. A low img2img strength adds plausible
# fine detail (skin texture, fabric weave) instead of only sharpening/
# denoising existing pixels — at the cost of being non-deterministic per
# view, since each of the ~6 views is repainted independently.

import os
import threading

import torch
from PIL import Image

_SD_LOCK = threading.Lock()


class SDTurboUpscaler():
    """Per-view generative upscale via SD Turbo img2img, applied before UV
    baking. Mirrors RealESRGAN_Upscaler's __call__(image) -> image interface
    so it's a drop-in alternative behind the same HY3D_USE_SUPER_RES gate."""

    def __init__(self, config, progress_callback=None):
        self.device = config.device
        self.texture_size = config.texture_size
        self.strength = float(os.environ.get('HY3D_SUPER_RES_STRENGTH', '0.3'))
        self.steps = int(os.environ.get('HY3D_SUPER_RES_STEPS', '8'))

        from diffusers import AutoPipelineForImage2Image

        model_path = os.environ.get(
            'HY3D_SD_TURBO_PATH',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), 'models', 'sd-turbo'),
        )
        if progress_callback is not None:
            progress_callback(0.0, "Loading SD Turbo for texture upscale...")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to(self.device)

    # guidance_scale=0.0 (below) disables classifier-free guidance, which is
    # how SD Turbo is meant to run — it does NOT mean the prompt is ignored:
    # it's still text-encoded and fed into the UNet's cross-attention every
    # step. But with CFG off, the prompt's actual influence on the output is
    # weak relative to the img2img source image, so a purely generic quality
    # phrase barely moves fine detail — verified: naming the actual subject
    # (e.g. "a cat, highly detailed...") measurably recovers detail a
    # generic-only prompt does not (sharper eyes, distinct whisker strands
    # on a real test case), because there's now an actual content signal for
    # the (weak) text conditioning to reinforce rather than only style words.
    DEFAULT_PROMPT = "highly detailed, sharp, photorealistic texture"

    @torch.no_grad()
    def __call__(self, image: Image.Image, subject: str = None) -> Image.Image:
        prompt = f"{subject}, {self.DEFAULT_PROMPT}" if subject else self.DEFAULT_PROMPT
        upsampled = image.convert('RGB').resize(
            (self.texture_size, self.texture_size), Image.LANCZOS,
        )
        with _SD_LOCK:
            out = self.pipe(
                prompt=prompt,
                image=upsampled,
                strength=self.strength,
                height=self.texture_size,
                width=self.texture_size,
                num_inference_steps=self.steps,
                guidance_scale=0.0,
            ).images[0]
        return out
