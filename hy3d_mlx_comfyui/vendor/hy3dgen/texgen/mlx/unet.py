"""MLX port of UNet2p5DConditionModel from Hunyuan3D-Paint.

This module rebuilds the full UNet with Hunyuan's custom transformer blocks
(4 attention pathways, PBR materials, RoPE) replacing the standard SD attention.

The architecture mirrors the PyTorch checkpoint structure:
    unet.conv_in, unet.time_embedding, unet.down_blocks, unet.mid_block,
    unet.up_blocks, unet.conv_norm_out, unet.conv_out
    unet.learned_text_clip_{albedo,mr,ref}
    unet.image_proj_model_dino.{proj,norm}
    unet_dual.* (reference stream)
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base.unet import (
    ResnetBlock2D,
    TimestepEmbedding,
    UNetBlock2D,
    upsample_nearest,
)
from .transformer_block import MLXTransformerBlock


# ─── Hunyuan Transformer2D (replaces base with custom attention) ─────────────

class HunyuanTransformer2D(nn.Module):
    """2D Transformer with Hunyuan's custom multi-attention blocks."""

    def __init__(
        self,
        in_channels: int,
        model_dims: int,
        encoder_dims: int,
        num_heads: int,
        dim_head: int,
        num_layers: int = 1,
        norm_num_groups: int = 32,
        pbr_settings: Optional[List[str]] = None,
        use_ma: bool = True,
        use_ra: bool = True,
        use_mda: bool = True,
        use_dino: bool = True,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, model_dims)
        self.transformer_blocks = [
            MLXTransformerBlock(
                dim=model_dims,
                num_heads=num_heads,
                dim_head=dim_head,
                cross_attention_dim=encoder_dims,
                pbr_settings=pbr_settings,
                use_ma=use_ma,
                use_ra=use_ra,
                use_mda=use_mda,
                use_dino=use_dino,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(model_dims, in_channels)

    def __call__(
        self,
        x: mx.array,
        encoder_x: mx.array,
        attn_mask=None,
        encoder_attn_mask=None,
        cross_attention_kwargs: Optional[Dict] = None,
    ) -> mx.array:
        input_x = x
        B, H, W, C = x.shape

        x = self.norm(x).reshape(B, -1, C)
        x = self.proj_in(x)

        kwargs = cross_attention_kwargs or {}

        for i, block in enumerate(self.transformer_blocks):
            # Build layer name matching PyTorch naming convention
            # The parent block will set the layer_name_prefix
            layer_name = kwargs.get("layer_name_prefix", "") + f"_{i}"

            x = block(
                x,
                encoder_hidden_states=encoder_x,
                num_in_batch=kwargs.get("num_in_batch", 1),
                mode=kwargs.get("mode", "r"),
                mva_scale=kwargs.get("mva_scale", 1.0),
                ref_scale=kwargs.get("ref_scale", 1.0),
                condition_embed_dict=kwargs.get("condition_embed_dict"),
                dino_hidden_states=kwargs.get("dino_hidden_states"),
                position_voxel_indices=kwargs.get("position_voxel_indices"),
                layer_name=layer_name,
                n_pbr=kwargs.get("n_pbr", 2),
            )

        x = self.proj_out(x)
        x = x.reshape(B, H, W, C)
        return x + input_x


# ─── Hunyuan UNet Block (threads cross_attention_kwargs) ─────────────────────

class HunyuanUNetBlock2D(nn.Module):
    """UNet block that threads cross_attention_kwargs to Hunyuan transformers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        prev_out_channels: Optional[int] = None,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1280,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        add_upsample: bool = True,
        add_cross_attention: bool = True,
        pbr_settings: Optional[List[str]] = None,
        use_ma: bool = True,
        use_ra: bool = True,
        use_mda: bool = True,
        use_dino: bool = True,
        block_name: str = "",
    ):
        super().__init__()

        if prev_out_channels is None:
            in_channels_list = [in_channels] + [out_channels] * (num_layers - 1)
        else:
            in_channels_list = [prev_out_channels] + [out_channels] * (num_layers - 1)
            res_channels_list = [out_channels] * (num_layers - 1) + [in_channels]
            in_channels_list = [a + b for a, b in zip(in_channels_list, res_channels_list)]

        self.resnets = [
            ResnetBlock2D(
                in_channels=ic,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
            )
            for ic in in_channels_list
        ]

        dim_head = out_channels // num_attention_heads
        self.has_cross_attention = add_cross_attention

        if add_cross_attention:
            self.attentions = [
                HunyuanTransformer2D(
                    in_channels=out_channels,
                    model_dims=out_channels,
                    num_heads=num_attention_heads,
                    dim_head=dim_head,
                    num_layers=transformer_layers_per_block,
                    encoder_dims=cross_attention_dim,
                    pbr_settings=pbr_settings,
                    use_ma=use_ma,
                    use_ra=use_ra,
                    use_mda=use_mda,
                    use_dino=use_dino,
                )
                for j in range(num_layers)
            ]
            self._attn_names = [f"{block_name}_{j}" for j in range(num_layers)]

        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(
        self,
        x: mx.array,
        encoder_x: mx.array = None,
        temb: mx.array = None,
        attn_mask=None,
        encoder_attn_mask=None,
        residual_hidden_states=None,
        cross_attention_kwargs: Optional[Dict] = None,
    ):
        output_states = []

        for i in range(len(self.resnets)):
            if residual_hidden_states is not None:
                x = mx.concatenate([x, residual_hidden_states.pop()], axis=-1)

            x = self.resnets[i](x, temb)

            if self.has_cross_attention and "attentions" in self:
                # Set layer name prefix for condition_embed_dict keying
                kwargs = dict(cross_attention_kwargs or {})
                kwargs["layer_name_prefix"] = self._attn_names[i]
                x = self.attentions[i](
                    x, encoder_x, attn_mask, encoder_attn_mask,
                    cross_attention_kwargs=kwargs,
                )

            output_states.append(x)

        if "downsample" in self:
            x = self.downsample(x)
            output_states.append(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))
            output_states.append(x)

        return x, output_states


# ─── ImageProjModel (DINO projection) ───────────────────────────────────────

class ImageProjModel(nn.Module):
    """Projects DINO image embeddings into cross-attention space."""

    def __init__(
        self,
        cross_attention_dim: int = 1024,
        clip_embeddings_dim: int = 1536,
        clip_extra_context_tokens: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        num_token = 1
        embeds = image_embeds
        if embeds.ndim == 3:
            num_token = embeds.shape[1]
            embeds = embeds.reshape(-1, embeds.shape[-1])

        tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        tokens = self.norm(tokens)
        tokens = tokens.reshape(-1, num_token * self.clip_extra_context_tokens, self.cross_attention_dim)
        return tokens


# ─── Main UNet ───────────────────────────────────────────────────────────────

class HunyuanUNetModel(nn.Module):
    """Full Hunyuan3D-Paint UNet with custom attention.

    This is the base UNet (either main or dual). The top-level wrapper
    HunyuanUNet2p5D handles the dual-stream logic.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        transformer_layers_per_block: Tuple[int, ...] = (1, 1, 1, 1),
        num_attention_heads: Tuple[int, ...] = (5, 10, 20, 20),
        cross_attention_dim: int = 1024,
        norm_num_groups: int = 32,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D", "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
        ),
        pbr_settings: Optional[List[str]] = None,
        use_ma: bool = True,
        use_ra: bool = True,
        use_mda: bool = True,
        use_dino: bool = True,
        **kwargs,
    ):
        super().__init__()
        pbr_settings = pbr_settings or ["albedo", "mr"]
        temb_channels = block_out_channels[0] * 4

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)

        # Time embedding
        self.timesteps = nn.SinusoidalPositionalEncoding(
            block_out_channels[0],
            max_freq=1,
            min_freq=math.exp(
                -math.log(10000) + 2 * math.log(10000) / block_out_channels[0]
            ),
            scale=1.0, cos_first=True, full_turns=False,
        )
        self.time_embedding = TimestepEmbedding(block_out_channels[0], temb_channels)

        # Down blocks
        block_channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            HunyuanUNetBlock2D(
                in_channels=ic,
                out_channels=oc,
                temb_channels=temb_channels,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_attention_heads=num_attention_heads[i],
                cross_attention_dim=cross_attention_dim,
                resnet_groups=norm_num_groups,
                add_downsample=(i < len(block_out_channels) - 1),
                add_upsample=False,
                add_cross_attention="CrossAttn" in down_block_types[i],
                pbr_settings=pbr_settings,
                use_ma=use_ma, use_ra=use_ra, use_mda=use_mda, use_dino=use_dino,
                block_name=f"down_{i}",
            )
            for i, (ic, oc) in enumerate(zip(block_channels, block_channels[1:]))
        ]

        # Mid block
        self.mid_resnets_0 = ResnetBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            groups=norm_num_groups,
        )
        dim_head_mid = block_out_channels[-1] // num_attention_heads[-1]
        self.mid_attentions_0 = HunyuanTransformer2D(
            in_channels=block_out_channels[-1],
            model_dims=block_out_channels[-1],
            num_heads=num_attention_heads[-1],
            dim_head=dim_head_mid,
            num_layers=transformer_layers_per_block[-1],
            encoder_dims=cross_attention_dim,
            pbr_settings=pbr_settings,
            use_ma=use_ma, use_ra=use_ra, use_mda=use_mda, use_dino=use_dino,
        )
        self.mid_resnets_1 = ResnetBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            groups=norm_num_groups,
        )

        # Up blocks — match diffusers index convention exactly
        # Diffusers up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
        # up_blocks[0] = UpBlock2D (deepest, no attn), up_blocks[3] = CrossAttnUpBlock2D (shallowest)
        #
        # Channel logic from diffusers UNet2DConditionModel:
        # block_out_channels = [320, 640, 1280, 1280]
        # reversed_boc = [1280, 1280, 640, 320]
        # For up_block i:
        #   out_channels = reversed_boc[i]
        #   in_channels = reversed_boc[min(i+1, len-1)]  (next block's output, or same for last)
        #   prev_out_channels: comes from the skip connection schedule
        n_blocks = len(block_out_channels)
        reversed_boc = list(reversed(block_out_channels))

        up_blocks_list = []
        for i in range(n_blocks):
            # Mirror the diffusers channel computation
            out_ch = reversed_boc[i]
            in_ch = reversed_boc[min(i + 1, n_blocks - 1)]
            # prev_out_channels for skip connections
            prev_ch = reversed_boc[max(i - 1, 0)] if i > 0 else block_out_channels[-1]

            # Use diffusers index for config (layers_per_block etc are indexed by original block order)
            config_idx = n_blocks - 1 - i

            up_blocks_list.append(HunyuanUNetBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                temb_channels=temb_channels,
                prev_out_channels=prev_ch,
                num_layers=layers_per_block[config_idx] + 1,
                transformer_layers_per_block=transformer_layers_per_block[config_idx],
                num_attention_heads=num_attention_heads[config_idx],
                cross_attention_dim=cross_attention_dim,
                resnet_groups=norm_num_groups,
                add_downsample=False,
                add_upsample=(i < n_blocks - 1),
                add_cross_attention="CrossAttn" in up_block_types[i],
                pbr_settings=pbr_settings,
                use_ma=use_ma, use_ra=use_ra, use_mda=use_mda, use_dino=use_dino,
                block_name=f"up_{i}",
            ))
        self.up_blocks = up_blocks_list

        # Output
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # Class/Camera embedding (used for view-dependent conditioning)
        # Optional based on config
        self.use_camera_embedding = kwargs.get("use_camera_embedding", True)
        if self.use_camera_embedding:
            self.class_embedding = nn.Embedding(49, temb_channels)

    def __call__(
        self,
        x: mx.array,
        timestep: mx.array,
        encoder_x: mx.array,
        class_labels: Optional[mx.array] = None,
        cross_attention_kwargs: Optional[Dict] = None,
    ) -> mx.array:
        # Time embedding
        # Ensure timestep is a 1D array for SinusoidalPositionalEncoding
        if timestep.ndim == 0:
            timestep = mx.expand_dims(timestep, axis=0)
        temb = self.timesteps(timestep).astype(x.dtype)
        temb = self.time_embedding(temb)
        # Broadcast temb to batch size if needed
        if temb.shape[0] == 1 and x.shape[0] > 1:
            temb = mx.repeat(temb, x.shape[0], axis=0)

        # Add camera/class embedding if provided
        if class_labels is not None and self.use_camera_embedding:
            temb = temb + self.class_embedding(class_labels)

        # Input projection (NHWC in MLX)
        x = self.conv_in(x)
        
        # Expand encoder states to match visual batch if needed (B_total vs B)
        if encoder_x is not None:
            if encoder_x.shape[0] < x.shape[0]:
                repeats = x.shape[0] // encoder_x.shape[0]
                encoder_x = mx.repeat(encoder_x, repeats, axis=0)

        # Downsampling
        residuals = [x]
        for block in self.down_blocks:
            x, res = block(x, encoder_x=encoder_x, temb=temb,
                          cross_attention_kwargs=cross_attention_kwargs)
            residuals.extend(res)

        # Mid block
        x = self.mid_resnets_0(x, temb)
        kwargs = dict(cross_attention_kwargs or {})
        kwargs["layer_name_prefix"] = "mid_0"
        x = self.mid_attentions_0(x, encoder_x, cross_attention_kwargs=kwargs)
        x = self.mid_resnets_1(x, temb)

        # Upsampling
        for block in self.up_blocks:
            x, _ = block(x, encoder_x=encoder_x, temb=temb,
                        residual_hidden_states=residuals,
                        cross_attention_kwargs=cross_attention_kwargs)

        # Output
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


# ─── Top-level 2.5D Wrapper ─────────────────────────────────────────────────

class HunyuanUNet2p5D(nn.Module):
    """Top-level Hunyuan3D-Paint UNet with dual-stream reference extraction.

    This corresponds to UNet2p5DConditionModel in the PyTorch version.
    It manages:
    - Main UNet (unet) for denoising
    - Dual-stream UNet (unet_dual) for reference feature extraction
    - Learned text embeddings per PBR material
    - DINO feature projection
    """

    def __init__(
        self,
        pbr_settings: Optional[List[str]] = None,
        cross_attention_dim: int = 1024,
        **unet_kwargs,
    ):
        super().__init__()
        self.pbr_settings = pbr_settings or ["albedo", "mr"]

        self.use_mda = unet_kwargs.get("use_mda", True)
        self.use_dino = unet_kwargs.get("use_dino", True)

        # Main UNet (12 input channels: 4 latent + 4 normal + 4 position)
        self.unet = HunyuanUNetModel(
            in_channels=12, pbr_settings=self.pbr_settings, **unet_kwargs
        )

        # Dual-stream reference UNet (4 input channels: just latents)
        # Filter out keys we want to override for the dual stream
        dual_kwargs = {k: v for k, v in unet_kwargs.items() 
                       if k not in ["use_ma", "use_ra", "use_mda", "use_dino", "use_camera_embedding"]}
        self.unet_dual = HunyuanUNetModel(
            in_channels=4, pbr_settings=["albedo"],
            use_ma=False, use_ra=False, use_mda=False, use_dino=False,
            use_camera_embedding=False,
            **dual_kwargs
        )

        # Learned text embeddings
        self.learned_text_clip_albedo = mx.zeros((1, 77, cross_attention_dim))
        if "mr" in self.pbr_settings:
            self.learned_text_clip_mr = mx.zeros((1, 77, cross_attention_dim))
        self.learned_text_clip_ref = mx.zeros((1, 77, cross_attention_dim))

        # DINO projection
        if self.use_dino:
            self.image_proj_model_dino = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=1536,
                clip_extra_context_tokens=4,
            )

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        embeds_normal: Optional[mx.array] = None,
        embeds_position: Optional[mx.array] = None,
        ref_latents: Optional[mx.array] = None,
        dino_hidden_states: Optional[mx.array] = None,
        position_voxel_indices: Optional[Dict] = None,
        mva_scale = 1.0,  # float or mx.array [B] for per-batch scaling
        ref_scale = 1.0,  # float or mx.array [B] for per-batch scaling
        num_in_batch: int = 6,
        cache: Optional[Dict] = None,
        **kwargs,
    ) -> mx.array:
        """Forward pass.

        Args:
            sample: [B, N_pbr, N_gen, C, H, W] noisy latents (NCHW from PyTorch, will be transposed)
            timestep: scalar or [B] timestep
            encoder_hidden_states: [B, N_pbr, seq_len, dim] text embeddings
            embeds_normal: [B, N_gen, C, H, W] normal map latents
            embeds_position: [B, N_gen, C, H, W] position map latents
            ref_latents: [B, N_ref, C, H, W] reference image latents
            dino_hidden_states: [B, n_patches, dino_dim] DINO features
            position_voxel_indices: pre-computed voxel indices for RoPE
            mva_scale: multiview attention scale
            ref_scale: reference attention scale
            num_in_batch: number of generated views
            cache: dict for caching across denoising steps
        """
        if cache is None:
            cache = {}

        B, N_pbr, N_gen = sample.shape[0], sample.shape[1], sample.shape[2]
        
        # Extract optional conditioning info
        class_labels = kwargs.get("class_labels")
        camera_info_ref = kwargs.get("camera_info_ref")

        # ── 1. Input preparation ──
        # Concatenate normal/position embeddings along channel dim
        # sample is [B, N_pbr, N_gen, C, H, W] in NCHW
        inputs = [sample]
        
        def _match_batch(arr, target_b):
            if arr.shape[0] == target_b:
                return arr
            if arr.shape[0] == 1:
                return mx.repeat(arr, target_b, axis=0)
            return arr[:target_b]

        if embeds_normal is not None:
            # [B, N_gen, C, H, W] → [B, N_pbr, N_gen, C, H, W]
            en = _match_batch(embeds_normal, B)
            inputs.append(mx.repeat(mx.expand_dims(en, axis=1), N_pbr, axis=1))
        if embeds_position is not None:
            ep = _match_batch(embeds_position, B)
            inputs.append(mx.repeat(mx.expand_dims(ep, axis=1), N_pbr, axis=1))
        sample_cat = mx.concatenate(inputs, axis=3)  # concat along C

        # Flatten to [B*N_pbr*N_gen, C, H, W]
        flat_sample = sample_cat.reshape(-1, *sample_cat.shape[3:])
        # NCHW → NHWC for MLX
        flat_sample = flat_sample.transpose(0, 2, 3, 1)

        # Expand text embeddings: [B, N_pbr, seq_len, dim] → [B*N_pbr*N_gen, seq_len, dim]
        enc_hs = mx.repeat(mx.expand_dims(encoder_hidden_states, axis=2), N_gen, axis=2)
        enc_hs = enc_hs.reshape(-1, *enc_hs.shape[3:])

        # ── 2. Reference feature extraction (dual-stream) ──
        if "condition_embed_dict" in cache:
            condition_embed_dict = cache["condition_embed_dict"]
        elif ref_latents is not None:
            condition_embed_dict = {}
            N_ref = ref_latents.shape[1]

            # [B, N_ref, C, H, W] → [B*N_ref, C, H, W] → NHWC
            ref_flat = ref_latents.reshape(-1, *ref_latents.shape[2:])
            ref_flat = ref_flat.transpose(0, 2, 3, 1)

            # Reference text embedding
            # Make sure we use the right shape for concatenation (1, 77, 1024)
            ref_token = self.learned_text_clip_ref
            if ref_token.ndim == 2:
                ref_token = mx.expand_dims(ref_token, axis=0)

            ref_enc = mx.repeat(
                mx.expand_dims(ref_token, axis=0),
                B, axis=0
            )  # [B, 1, 77, 1024]
            ref_enc = mx.repeat(ref_enc, N_ref, axis=1)
            ref_enc = ref_enc.reshape(-1, *ref_enc.shape[2:])

            ref_timestep = mx.array(0.0)

            # Run dual UNet in write mode
            self.unet_dual(
                ref_flat, ref_timestep, ref_enc,
                class_labels=camera_info_ref,
                cross_attention_kwargs={
                    "mode": "w",
                    "num_in_batch": N_ref,
                    "condition_embed_dict": condition_embed_dict,
                    "n_pbr": 1,
                },
            )
            cache["condition_embed_dict"] = condition_embed_dict
        else:
            condition_embed_dict = None

        # ── 3. DINO projection ──
        if "dino_hidden_states_proj" in cache:
            dino_proj = cache["dino_hidden_states_proj"]
        elif dino_hidden_states is not None:
            dino_proj = self.image_proj_model_dino(dino_hidden_states)
            cache["dino_hidden_states_proj"] = dino_proj
        else:
            dino_proj = None

        # ── 4. Camera embedding ──
        # (Handled at the top now)

        # ── 5. Main UNet forward ──
        output = self.unet(
            flat_sample, timestep, enc_hs,
            class_labels=class_labels,
            cross_attention_kwargs={
                "mode": "r",
                "num_in_batch": N_gen,
                "dino_hidden_states": dino_proj,
                "condition_embed_dict": condition_embed_dict,
                "mva_scale": mva_scale,
                "ref_scale": ref_scale,
                "position_voxel_indices": position_voxel_indices,
                "n_pbr": N_pbr,
            },
        )

        # NHWC → NCHW for output
        output = output.transpose(0, 3, 1, 2)

        return output

    def load_weights(self, weights: List[Tuple[str, mx.array]], strict: bool = True):
        """Override to handle inconsistent text token shapes across versions."""
        processed = []
        for name, weight in weights:
            # Handle (77, 1024) vs (1, 77, 1024)
            if "learned_text_clip" in name:
                model_shape = getattr(self, name).shape
                if weight.shape != model_shape:
                    print(f"[MLX] Reshaping {name}: {weight.shape} -> {model_shape}")
                    weight = weight.reshape(model_shape)
            processed.append((name, weight))
        return super().load_weights(processed, strict=strict)
