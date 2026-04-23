# Adapted from ml-explore/mlx-examples/stable_diffusion
# Copyright © 2023 Apple Inc. — MIT License

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215


@dataclass
class UNetConfig:
    """UNet configuration matching Hunyuan3D-Paint's architecture.

    Default values match the standard SD 2.1 UNet used as the base
    in Hunyuan's UNet2p5DConditionModel. The wrapper adds 12 input
    channels (4 latent + 4 normal + 4 position) and custom attention.
    """
    in_channels: int = 4
    out_channels: int = 4
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: Tuple[int, ...] = (2, 2, 2, 2)
    transformer_layers_per_block: Tuple[int, ...] = (1, 1, 1, 1)
    num_attention_heads: Tuple[int, ...] = (5, 10, 20, 20)
    cross_attention_dim: Tuple[int, ...] = (1024,) * 4
    norm_num_groups: int = 32
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    projection_class_embeddings_input_dim: Optional[int] = None


@dataclass
class DDIMConfig:
    """DDIM scheduler configuration matching Hunyuan3D-Paint's setup."""
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_steps: int = 1000
    prediction_type: str = "v_prediction"
    rescale_betas_zero_snr: bool = True
    timestep_spacing: str = "trailing"
