"""MLX inference pipeline for Hunyuan3D-Paint texture generation.

Replaces the PyTorch diffusers-based pipeline with a pure MLX implementation.
The 3D pipeline (rasterizer, UV, baking) stays in PyTorch — this handles only
the diffusion inference (UNet + VAE + DDIM denoising loop).

Inputs/outputs are numpy arrays at the boundary.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .base.config import AutoencoderConfig, DDIMConfig
from .base.sampler import DDIMSampler
from .base.vae import Autoencoder
from .unet import HunyuanUNet2p5D


def _cam_mapping(azim: float) -> float:
    """View-dependent guidance scale based on camera azimuth."""
    if 0 <= azim < 90:
        return float(azim) / 90.0 + 1
    elif 90 <= azim < 330:
        return 2.0
    else:
        return -float(azim) / 90.0 + 5.0


class MLXHunyuanPaintPipeline:
    """MLX inference pipeline for Hunyuan3D-Paint.

    Usage:
        pipeline = MLXHunyuanPaintPipeline.from_pretrained(model_path)
        images = pipeline(
            latents, prompt_embeds, ref_latents, embeds_normal, embeds_position,
            dino_hidden_states, camera_azims, num_steps=10, guidance_scale=3.0,
        )
    """

    def __init__(
        self,
        unet: HunyuanUNet2p5D,
        vae: Autoencoder,
        scheduler: DDIMSampler,
    ):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler

    @staticmethod
    def from_pretrained(model_path: str, weights_path: Optional[str] = None):
        """Load the MLX pipeline from converted weights.

        Args:
            model_path: Path to hunyuan3d-paintpbr-v2-1 directory
            weights_path: Path to converted MLX weights (default: model_path/mlx_weights)
        """
        if weights_path is None:
            weights_path = os.path.join(model_path, "mlx_weights")

        unet_npz = os.path.join(weights_path, "unet.npz")
        if not os.path.exists(unet_npz):
            raise FileNotFoundError(
                f"[MLX] Pre-converted MLX weights not found: {unet_npz}\n"
                "Convert local weights with: python -m hy3dgen.texgen.mlx.convert_weights "
                "--model-path <paint-model-subfolder>"
            )

        # Load configs
        with open(os.path.join(model_path, "unet", "config.json")) as f:
            unet_config = json.load(f)

        # Create UNet
        attention_head_dim = unet_config.get("attention_head_dim", [5, 10, 20, 20])
        block_out_channels = tuple(unet_config.get("block_out_channels", [320, 640, 1280, 1280]))
        cross_attention_dim = unet_config.get("cross_attention_dim", 1024)

        unet = HunyuanUNet2p5D(
            pbr_settings=["albedo", "mr"],
            cross_attention_dim=cross_attention_dim,
            out_channels=4,
            block_out_channels=block_out_channels,
            layers_per_block=tuple(unet_config.get("layers_per_block", 2) if isinstance(unet_config.get("layers_per_block"), list) else [unet_config.get("layers_per_block", 2)] * len(block_out_channels)),
            transformer_layers_per_block=(1,) * len(block_out_channels),
            num_attention_heads=tuple(attention_head_dim),
            norm_num_groups=unet_config.get("norm_num_groups", 32),
        )

        # Load UNet weights
        print("Loading MLX UNet weights...")
        unet_weights = dict(np.load(os.path.join(weights_path, "unet.npz"), allow_pickle=True))
        # Convert numpy arrays to mlx arrays
        unet_weights_mx = {k: mx.array(v) for k, v in unet_weights.items()}
        unet.load_weights(list(unet_weights_mx.items()))
        print(f"  Loaded {len(unet_weights)} UNet weight arrays")

        # Create VAE
        vae_config = AutoencoderConfig(
            in_channels=3, out_channels=3,
            latent_channels_out=8, latent_channels_in=4,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2, norm_num_groups=32,
            scaling_factor=0.18215,
        )
        vae = Autoencoder(vae_config)

        # Load VAE weights
        print("Loading MLX VAE weights...")
        vae_weights = dict(np.load(os.path.join(weights_path, "vae.npz"), allow_pickle=True))
        vae_weights_mx = {k: mx.array(v) for k, v in vae_weights.items()}
        vae.load_weights(list(vae_weights_mx.items()))
        print(f"  Loaded {len(vae_weights)} VAE weight arrays")

        # Create scheduler
        scheduler_config = DDIMConfig(
            prediction_type="v_prediction",
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        scheduler = DDIMSampler(scheduler_config)

        print("MLX pipeline ready!")
        return MLXHunyuanPaintPipeline(unet, vae, scheduler)

    def encode_images(self, images_np: np.ndarray) -> mx.array:
        """Encode images to VAE latent space.

        Args:
            images_np: [B, H, W, 3] float32 in [0, 1]

        Returns:
            latents: [B, H//8, W//8, 4] (NHWC)
        """
        x = mx.array(images_np)
        x = (x - 0.5) * 2.0  # normalize to [-1, 1]
        mean, _ = self.vae.encode(x)
        return mean  # already scaled by vae.scaling_factor

    def decode_latents(self, latents: mx.array) -> np.ndarray:
        """Decode latents to images.

        Args:
            latents: [B, H_lat, W_lat, 4] (NHWC)

        Returns:
            images: [B, H, W, 3] float32 in [0, 1]
        """
        images = self.vae.decode(latents)
        images = (images * 0.5 + 0.5)
        images = mx.clip(images, 0.0, 1.0)
        mx.eval(images)
        return np.array(images)

    def __call__(
        self,
        # All inputs as numpy arrays (NCHW from PyTorch pipeline)
        prompt_embeds_np: np.ndarray,
        negative_prompt_embeds_np: np.ndarray,
        ref_latents_np: np.ndarray,
        embeds_normal_np: np.ndarray,
        embeds_position_np: np.ndarray,
        dino_hidden_states_np: np.ndarray,
        camera_azims: List[float],
        height: int = 64,
        width: int = 64,
        num_in_batch: int = 6,
        n_pbr: int = 2,
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Run the diffusion denoising loop.

        All inputs are numpy arrays in NCHW format (from PyTorch preprocessing).
        Returns denoised latents as numpy array in NCHW format.
        """
        mx.random.seed(seed)

        # Convert inputs to MLX
        # Triple-batch prompt embeddings: [uncond, ref, full]
        prompt_embeds = mx.array(np.concatenate([
            negative_prompt_embeds_np, prompt_embeds_np, prompt_embeds_np
        ], axis=0))

        ref_latents = mx.array(ref_latents_np)
        embeds_normal = mx.array(embeds_normal_np)
        embeds_position = mx.array(embeds_position_np)
        dino_hidden_states = mx.array(dino_hidden_states_np)

        # Triple-batch the conditioning for CFG
        ref_latents_cfg = mx.repeat(ref_latents, 3, axis=0)
        embeds_normal_cfg = mx.repeat(embeds_normal, 3, axis=0)
        embeds_position_cfg = mx.repeat(embeds_position, 3, axis=0)
        # DINO: [zero, zero, real]
        B = dino_hidden_states.shape[0]
        dino_zeros = mx.zeros_like(dino_hidden_states)
        dino_cfg = mx.concatenate([dino_zeros, dino_zeros, dino_hidden_states], axis=0)

        # Setup scheduler
        self.scheduler.set_timesteps(num_steps)

        # Initialize latents
        num_latents = B * num_in_batch * n_pbr
        latents = mx.random.normal((num_latents, 4, height, width)).astype(mx.float16)

        # View-dependent guidance scale
        view_scales = np.array([_cam_mapping(a) for a in camera_azims], dtype=np.float32)
        view_scales = np.tile(view_scales, n_pbr)  # repeat for PBR materials
        view_scale = mx.array(view_scales).reshape(-1, 1, 1, 1)

        # Cache for condition embeddings across steps
        cache = {}

        print(f"Running {num_steps} denoising steps...")
        t_start = time.time()

        for i, t in enumerate(self.scheduler.timesteps.tolist()):
            t_step_start = time.time()
            t_mx = mx.array(float(t))

            # Reshape for CFG: [B*pbr*N, C, H, W] → [B, pbr, N, C, H, W]
            latents_6d = latents.reshape(B, n_pbr, num_in_batch, 4, height, width)

            # Triple for CFG: [3*B, pbr, N, C, H, W]
            latent_model_input = mx.repeat(latents_6d, 3, axis=0)

            # UNet forward
            noise_pred = self.unet(
                latent_model_input, t_mx, prompt_embeds,
                embeds_normal=embeds_normal_cfg,
                embeds_position=embeds_position_cfg,
                ref_latents=ref_latents_cfg,
                dino_hidden_states=dino_cfg,
                mva_scale=1.0,
                ref_scale=1.0,
                num_in_batch=num_in_batch,
                cache=cache,
            )

            # CFG guidance
            noise_pred_uncond = noise_pred[:num_latents]
            noise_pred_ref = noise_pred[num_latents:2*num_latents]
            noise_pred_full = noise_pred[2*num_latents:]

            noise_pred = (
                noise_pred_uncond
                + guidance_scale * view_scale * (noise_pred_ref - noise_pred_uncond)
                + guidance_scale * view_scale * (noise_pred_full - noise_pred_ref)
            )

            # DDIM step (NCHW format)
            latents = self.scheduler.step(noise_pred, mx.array(t), latents[:, :4])

            mx.eval(latents)

            elapsed = time.time() - t_step_start
            print(f"  Step {i+1}/{num_steps} (t={t}): {elapsed:.1f}s")

        total = time.time() - t_start
        print(f"Denoising complete: {total:.1f}s total")

        # Return as numpy NCHW
        mx.eval(latents)
        return np.array(latents)
