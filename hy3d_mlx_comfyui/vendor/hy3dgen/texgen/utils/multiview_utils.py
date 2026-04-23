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

import os
import random
import shutil

import numpy as np
import torch
from typing import List
from diffusers import DiffusionPipeline
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler


class Multiview_Diffusion_Net():
    def __init__(self, config) -> None:
        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        current_file_path = os.path.abspath(__file__)
        pipeline_dir_name = 'hunyuanpaintpbr' if config.pipe_name == 'hunyuanpaintpbr' else 'hunyuanpaint'
        custom_pipeline_path = os.path.join(os.path.dirname(current_file_path), '..', pipeline_dir_name)
        self.is_pbr = config.pipe_name == 'hunyuanpaintpbr'

        if self.is_pbr:
            local_unet_dir = os.path.join(os.path.dirname(current_file_path), '..', 'hunyuanpaintpbr', 'unet')
            model_unet_dir = os.path.join(multiview_ckpt_path, 'unet')
            for module_file in ['attn_processor.py', 'modules.py']:
                local_file = os.path.join(local_unet_dir, module_file)
                model_file = os.path.join(model_unet_dir, module_file)
                if os.path.exists(local_file) and os.path.exists(model_unet_dir):
                    shutil.copy2(local_file, model_file)

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
        )

        if config.pipe_name == 'hunyuanpaint':
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing='trailing',
            )
        elif config.pipe_name == 'hunyuanpaint-turbo':
            pipeline.scheduler = LCMScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing='trailing',
            )
            pipeline.set_turbo(True)
        elif config.pipe_name == 'hunyuanpaintpbr':
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        pipeline.set_progress_bar_config(disable=True)

        # Fix for Tencent's 2.1 PBR pipeline bug (expects 'gen' but model has 'albedo')
        if hasattr(pipeline, "unet"):
            inner_unet = getattr(pipeline.unet, "unet", pipeline.unet)
            if not hasattr(inner_unet, "learned_text_clip_gen"):
                if hasattr(inner_unet, "learned_text_clip_albedo"):
                    inner_unet.learned_text_clip_gen = inner_unet.learned_text_clip_albedo
                elif hasattr(inner_unet, "learned_text_clip_ref"):
                    inner_unet.learned_text_clip_gen = inner_unet.learned_text_clip_ref

        backend = getattr(config, 'diffusion_backend', 'pytorch')
        if backend == 'mlx' and self.is_pbr:
            # Keep the heavy PyTorch pipeline on CPU for 2.1 MLX mode.
            # MLX handles the UNet compute; keeping PyTorch weights off MPS avoids unified-memory kills.
            self.pipeline = pipeline
        else:
            self.pipeline = pipeline.to(self.device)

        self.dino_v2 = None
        if self.is_pbr and hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            try:
                from ..hunyuanpaintpbr.unet.modules import Dino_v2
                dino_ckpt_path = os.path.join(multiview_ckpt_path, "mvd_std")
                if os.path.exists(dino_ckpt_path):
                    self.dino_v2 = Dino_v2(dino_ckpt_path).to(torch.float16)
                    self.dino_v2 = self.dino_v2.to(self.device)
            except Exception as e:
                print(f"[WARN] Failed to load Dino_v2 for 2.1 PBR pipeline: {e}")

        # Optional MLX backend: keep diffusers pipeline, replace UNet forward with MLX.
        if backend == 'mlx':
            if torch.device(self.device).type != 'mps':
                raise RuntimeError(
                    f"[MLX] backend=mlx requires MPS device, got: {self.device}"
                )
            try:
                from ..mlx.hybrid_unet import HybridMLXUNet

                self._mlx_hybrid = HybridMLXUNet.patch_pipeline(
                    self.pipeline,
                    model_path=multiview_ckpt_path,
                    weights_path=getattr(config, 'mlx_weights_path', None),
                    profile=getattr(config, 'mlx_profile', None),
                )

                # Free MPS memory: MLX owns the UNet weights now.
                if self.is_pbr:
                    self.pipeline.unet.to('cpu')
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except Exception as e:
                raise RuntimeError(f"[MLX] Failed to enable MLX backend: {e}") from e

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    def __call__(self, input_images, control_images, camera_info):

        self.seed_everything(0)

        if not isinstance(input_images, List):
            input_images = [input_images]

        input_images = [input_image.resize((self.view_size, self.view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view

        if self.is_pbr:
            kwargs["images_normal"] = normal_image
            kwargs["images_position"] = position_image
            if self.dino_v2 is not None:
                kwargs["dino_hidden_states"] = self.dino_v2(input_images[0])
            else:
                kwargs["dino_hidden_states"] = torch.zeros(
                    (1, 1, 1536),
                    dtype=getattr(self.pipeline.unet, "dtype", torch.float16),
                    device=self.device,
                )
            mvd_image = self.pipeline(
                input_images[0:1],
                num_inference_steps=30,
                guidance_scale=3.0,
                **kwargs,
            ).images

            # Current Hunyuan3D-MLX texture baker expects a list of albedo multiview images.
            return mvd_image[:num_view]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        mvd_image = self.pipeline(input_images, num_inference_steps=30, **kwargs).images

        return mvd_image
