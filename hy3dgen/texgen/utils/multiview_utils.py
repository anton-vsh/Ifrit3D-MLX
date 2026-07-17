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
        # hybrid_unet.py's MLX UNet wrapper infers batch size from raw tensor
        # size and has no notion of classifier-free-guidance batch doubling.
        # The non-turbo ("hunyuanpaint") pipeline doubles the latent batch for
        # CFG when guidance_scale > 1, which desyncs that inference and
        # scrambles the uncond/cond split — producing colored-noise output.
        # Turbo never hits this (it disables CFG doubling outright via
        # is_turbo), so only "hunyuanpaint" + mlx needs CFG forced off here.
        self._force_no_cfg = backend == 'mlx' and config.pipe_name == 'hunyuanpaint'
        # Move the whole pipeline to MPS even for PBR+MLX: the VAE and text
        # encoder still run as real PyTorch fp16 ops and are catastrophically
        # slow on CPU (Apple Silicon has no fast fp16 CPU kernels — a single
        # VAE encode() that takes ~1s on MPS can take 30+ minutes on CPU,
        # indistinguishable from a hang). Only the UNet's PyTorch weights are
        # evicted back to CPU below, once MLX has taken over its forward pass.
        self.pipeline = pipeline.to(self.device)

        self.dino_v2 = None
        if self.is_pbr and hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            try:
                from ..hunyuanpaintpbr.unet.modules import Dino_v2
                # Tencent's reference hy3dpaint/textureGenPipeline.py loads DINOv2
                # from the "facebook/dinov2-giant" HF repo directly (not bundled
                # inside the paint checkpoint) — fall back to that when a local
                # "mvd_std" copy isn't present, since without it the model gets
                # a zero-tensor placeholder instead of real image features and
                # generates content unrelated to the reference image.
                dino_ckpt_path = os.path.join(multiview_ckpt_path, "mvd_std")
                if not os.path.exists(dino_ckpt_path):
                    dino_ckpt_path = "facebook/dinov2-giant"
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
            pbr_albedo_only = self.is_pbr and getattr(config, 'pbr_albedo_only', False)
            if pbr_albedo_only:
                # Pipeline pre/post-processing (batch construction, learned-embedding
                # lookup) reads this generically rather than hardcoding 2 channels —
                # see hunyuanpaintpbr/pipeline.py's n_pbr = len(self.unet.pbr_setting).
                self.pipeline.unet.pbr_setting = ["albedo"]
            try:
                from ..mlx.hybrid_unet import HybridMLXUNet

                self._mlx_hybrid = HybridMLXUNet.patch_pipeline(
                    self.pipeline,
                    model_path=multiview_ckpt_path,
                    weights_path=getattr(config, 'mlx_weights_path', None),
                    profile=getattr(config, 'mlx_profile', None),
                    pbr_albedo_only=pbr_albedo_only,
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

    def __call__(self, input_images, control_images, camera_info, seed=0, progress_callback=None):

        self.seed_everything(seed)

        if not isinstance(input_images, List):
            input_images = [input_images]

        input_images = [input_image.resize((self.view_size, self.view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(seed))

        num_inference_steps = 30
        step_kwargs = {}
        if progress_callback is not None:
            def _on_step_end(pipe, step, timestep, callback_kwargs):
                # Turbo/LCM schedulers resolve fewer actual timesteps than the
                # requested num_inference_steps; `_num_timesteps` (set by the
                # pipeline right before its denoising loop) reflects what will
                # really run, so the displayed total doesn't get stuck early.
                total = getattr(pipe, "_num_timesteps", num_inference_steps)
                progress_callback((step + 1) / total, f"Multiview diffusion — step {step + 1}/{total}")
                return callback_kwargs

            step_kwargs["callback_on_step_end"] = _on_step_end

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
                num_inference_steps=num_inference_steps,
                guidance_scale=3.0,
                **kwargs,
                **step_kwargs,
            ).images

            # Current Ifrit3D-MLX texture baker expects a list of albedo multiview images.
            return mvd_image[:num_view]

        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image

        if self._force_no_cfg:
            kwargs['guidance_scale'] = 1.0

        mvd_image = self.pipeline(input_images, num_inference_steps=num_inference_steps, **kwargs, **step_kwargs).images

        return mvd_image
