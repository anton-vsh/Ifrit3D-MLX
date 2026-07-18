"""Hybrid UNet wrapper: PyTorch interface, MLX computation.

Replaces the PyTorch UNet forward pass with a wrapper that:
1) Converts PyTorch tensors -> numpy -> MLX arrays
2) Runs MLX UNet forward pass
3) Converts MLX output -> numpy -> PyTorch tensors

Supports two paint profiles:
- paint-2.0      (legacy non-PBR path used by this repo's local runtime)
- paint-pbr-2.1  (PBR profile, compatible with paintpbr checkpoints)
"""

import os
import time
from typing import Optional

import mlx.core as mx
import mlx.utils
import numpy as np
import torch

from .unet import HunyuanUNet2p5D
from ..hunyuanpaintpbr.unet.modules import calc_multires_voxel_idxs


PROFILE_PAINT_20 = "paint-2.0"
PROFILE_PAINT_PBR_21 = "paint-pbr-2.1"


def _infer_profile(model_path: str, profile: Optional[str] = None) -> str:
    if profile in {PROFILE_PAINT_20, PROFILE_PAINT_PBR_21}:
        return profile
    model_name = os.path.basename(model_path).lower()
    if "paintpbr" in model_name or "v2-1" in model_name:
        return PROFILE_PAINT_PBR_21
    return PROFILE_PAINT_20


def _resolve_weights_path(model_path: str, weights_path: Optional[str], profile: str) -> str:
    if weights_path:
        return weights_path

    parent = os.path.dirname(model_path)
    if profile == PROFILE_PAINT_20:
        candidates = [
            os.path.join(parent, "hunyuan3d-2.0-mlx"),
            os.path.join(model_path, "mlx_weights"),
        ]
    else:
        candidates = [
            os.path.join(parent, "hunyuan3d-2.1-mlx"),
            os.path.join(model_path, "mlx_weights"),
        ]

    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


class HybridMLXUNet:
    """Wraps MLX UNet to match PyTorch UNet2p5DConditionModel interface."""

    def __init__(
        self,
        model_path: str,
        weights_path: Optional[str] = None,
        profile: Optional[str] = None,
        pbr_albedo_only: bool = False,
    ):
        """Load MLX UNet from converted weights.

        Args:
            model_path: Paint model directory (contains unet/)
            weights_path: Directory containing unet.npz
            profile: paint-2.0 or paint-pbr-2.1 (auto if None)
            pbr_albedo_only: for paint-pbr-2.1, build/load only the "albedo"
                material branch and skip "mr" (metallic-roughness) entirely —
                roughly halves multiview diffusion compute for callers that
                only need a basic (non-PBR) texture.
        """
        self.profile = _infer_profile(model_path, profile)
        weights_path = _resolve_weights_path(model_path, weights_path, self.profile)

        unet_npz = os.path.join(weights_path, "unet.npz")
        if not os.path.exists(unet_npz):
            raise FileNotFoundError(
                f"[MLX] Converted MLX UNet weights not found: {unet_npz}\n"
                f"Convert local weights with: python -m hy3dgen.texgen.mlx.convert_weights "
                f"--model-path {model_path} --profile {self.profile}"
            )

        if self.profile == PROFILE_PAINT_20:
            # Legacy runtime path in this repo (single material, no DINO/MDA).
            self.pbr_settings = ["albedo"]
            self.mlx_unet = HunyuanUNet2p5D(
                pbr_settings=self.pbr_settings,
                cross_attention_dim=1024,
                out_channels=4,
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=(2, 2, 2, 2),
                transformer_layers_per_block=(1, 1, 1, 1),
                num_attention_heads=(5, 10, 20, 20),
                norm_num_groups=32,
                use_mda=False,
                use_dino=False,
                use_camera_embedding=True,
            )
        else:
            # Material-dimension attention (use_mda) and DINO conditioning stay
            # on regardless of pbr_albedo_only — pbr_settings is a genuine
            # variable-length construction param (see transformer_block.py),
            # not something use_mda hardcodes to exactly 2 entries.
            self.pbr_settings = ["albedo"] if pbr_albedo_only else ["albedo", "mr"]
            self.mlx_unet = HunyuanUNet2p5D(
                pbr_settings=self.pbr_settings,
                cross_attention_dim=1024,
                out_channels=4,
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=(2, 2, 2, 2),
                transformer_layers_per_block=(1, 1, 1, 1),
                num_attention_heads=(5, 10, 20, 20),
                norm_num_groups=32,
                use_mda=True,
                use_dino=True,
                use_camera_embedding=False,
            )

        print(f"[MLX] Loading UNet weights ({self.profile}, pbr_settings={self.pbr_settings}) from {weights_path} ...")
        t0 = time.time()
        raw = dict(np.load(unet_npz, allow_pickle=True))
        if pbr_albedo_only and self.profile != PROFILE_PAINT_20:
            # The checkpoint was converted for the full 2-channel (albedo+mr)
            # model. Filter to just the keys this smaller model actually has —
            # verified this is a clean subset (no partially-shared tensors):
            # filtering to the albedo-only model's own param names skips
            # exactly the "*_mr*"/"learned_text_clip_mr" keys with zero
            # missing/mismatched entries, then a strict=True load confirms
            # full coverage instead of silently leaving something zero-init.
            model_param_names = {name for name, _ in mlx.utils.tree_flatten(self.mlx_unet.parameters())}
            weights = [(k, mx.array(v)) for k, v in raw.items() if k in model_param_names]
            skipped = len(raw) - len(weights)
            self.mlx_unet.load_weights(weights, strict=True)
            print(f"[MLX] Loaded {len(weights)} weights in {time.time() - t0:.1f}s (skipped {skipped} mr-only keys)")
        else:
            self.mlx_unet.load_weights([(k, mx.array(v)) for k, v in raw.items()])
            print(f"[MLX] Loaded {len(raw)} weights in {time.time() - t0:.1f}s")
        del raw

        self._cache = {}
        self._call_count = 0

    @staticmethod
    def _to_np(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().numpy()
        return v

    @staticmethod
    def _scale_to_mlx(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.item())
            return mx.array(v.detach().cpu().float().numpy())
        return float(v)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        *args,
        return_dict=False,
        **kwargs,
    ):
        """Drop-in replacement for UNet2p5DConditionModel.forward()."""
        self._call_count += 1
        device = sample.device
        dtype = sample.dtype

        # Extract kwargs that the MLX UNet needs
        embeds_normal = kwargs.get("embeds_normal", kwargs.get("normal_imgs"))
        embeds_position = kwargs.get("embeds_position", kwargs.get("position_imgs"))
        ref_latents = kwargs.get("ref_latents")
        dino_hidden_states = kwargs.get("dino_hidden_states")
        # Raw (non-VAE-encoded) position maps — used only for 3D RoPE voxel
        # indices below, distinct from embeds_position (the encoded version).
        position_maps = kwargs.get("position_maps")
        
        # View/Camera info
        class_labels = kwargs.get("class_labels", kwargs.get("camera_info_gen"))
        camera_info_ref = kwargs.get("camera_info_ref")

        # Per-branch CFG scales pass through as mx.array [B] — e.g. the 2.1
        # pipeline's ref_scale=[0,1,1], where branch 0 is the *unconditional*
        # CFG branch with reference attention switched off. These must NOT be
        # collapsed to a scalar: doing so (old behavior: take the last
        # element) applied full conditioning to the unconditional branch too,
        # making cond≈uncond so the CFG update guidance_scale*(cond-uncond)
        # amplified quasi-random residual instead of a real guidance signal —
        # measured on a real mesh as 3.2x more blue-artifact pixels and 24%
        # more speckle vs guidance disabled. transformer_block.py's
        # _broadcast_scale already expands [B] arrays to the flattened
        # (b n_pbr n) batch exactly like the PyTorch reference does.
        mva_scale = self._scale_to_mlx(kwargs.get("mva_scale", 1.0))
        ref_scale = self._scale_to_mlx(kwargs.get("ref_scale", 1.0))

        sample_np = self._to_np(sample)
        enc_np = self._to_np(encoder_hidden_states)

        # PyTorch pipeline passes sample in various formats ([Total, C, H, W] or [B, Total, C, H, W])
        # We must normalize to [B, N_pbr, N_gen, C, H, W] for MLX UNet
        n_pbr = len(self.pbr_settings)
        
        if "num_in_batch" in kwargs:
            num_in_batch = kwargs["num_in_batch"]
        else:
            # Fallback: calculate from total views assuming 1 material
            num_in_batch = (sample_np.size // (sample_np.shape[-3] * sample_np.shape[-2] * sample_np.shape[-1])) // n_pbr

        # Calculate batch size dynamically
        # Each item is [C, H, W]
        item_size = sample_np.shape[-3] * sample_np.shape[-2] * sample_np.shape[-1]
        total_items = sample_np.size // item_size
        batch_size = total_items // (n_pbr * num_in_batch)

        # Reshape to standard 6D: [B, N_pbr, N_gen, C, H, W]
        sample_np = sample_np.reshape(batch_size, n_pbr, num_in_batch, *sample_np.shape[-3:])

        # 3D RoPE voxel indices for multiview cross-attention (position-aware
        # attention correlating the same 3D surface point across views).
        # use_position_rope=True is unconditional on the PyTorch side
        # (hunyuanpaintpbr/unet/modules.py) — without this, multiview
        # attention here was silently running position-blind, degrading
        # cross-view detail/consistency. calc_multires_voxel_idxs is pure
        # PyTorch tensor math (no model weights), so it's called directly on
        # the raw position_maps tensor before any MLX conversion, matching
        # exactly what the PyTorch UNet computes for the same input.
        position_voxel_indices = None
        if position_maps is not None:
            H = sample_np.shape[-2]
            pt_voxel_indices = calc_multires_voxel_idxs(
                position_maps,
                grid_resolutions=[H, H // 2, H // 4, H // 8],
                voxel_resolutions=[H * 8, H * 4, H * 2, H],
            )
            position_voxel_indices = {
                seq_len: {
                    "voxel_indices": mx.array(entry["voxel_indices"].cpu().numpy()),
                    "voxel_resolution": entry["voxel_resolution"],
                }
                for seq_len, entry in pt_voxel_indices.items()
            }

        # Normalize encoder hidden states to [B, N_pbr, Seq, Dim]
        # Pipeline passes [B_total, Seq, Dim]
        enc_item_size = enc_np.shape[-2] * enc_np.shape[-1]
        enc_total_items = enc_np.size // enc_item_size
        
        if enc_total_items == batch_size * n_pbr * num_in_batch:
            # [B, N_pbr, N_gen, Seq, Dim] -> [B, N_pbr, Seq, Dim] (take first view)
            enc_np = enc_np.reshape(batch_size, n_pbr, num_in_batch, *enc_np.shape[-2:])
            enc_np = enc_np[:, :, 0, :, :]
        else:
            # Assume [B_total, Seq, Dim] matches [B * N_pbr, Seq, Dim] or needs expansion
            enc_np = enc_np.reshape(-1, *enc_np.shape[-2:])
            if enc_np.shape[0] < batch_size * n_pbr:
                # Tile if needed
                repeats = (batch_size * n_pbr + enc_np.shape[0] - 1) // enc_np.shape[0]
                enc_np = np.repeat(enc_np, repeats, axis=0)[:batch_size * n_pbr]
            enc_np = enc_np.reshape(batch_size, n_pbr, *enc_np.shape[-2:])

        t_val = (
            float(timestep)
            if isinstance(timestep, (int, float))
            else float(timestep.item())
            if timestep.dim() == 0
            else float(timestep[0].item())
        )

        t0 = time.time()
        output = self.mlx_unet(
            mx.array(sample_np),
            mx.array(t_val),
            mx.array(enc_np),
            embeds_normal=mx.array(self._to_np(embeds_normal)) if embeds_normal is not None else None,
            embeds_position=mx.array(self._to_np(embeds_position)) if embeds_position is not None else None,
            ref_latents=mx.array(self._to_np(ref_latents)) if ref_latents is not None else None,
            dino_hidden_states=mx.array(self._to_np(dino_hidden_states)) if dino_hidden_states is not None else None,
            position_voxel_indices=position_voxel_indices,
            class_labels=mx.array(self._to_np(class_labels)).astype(mx.int64).reshape(-1) if class_labels is not None else None,
            camera_info_ref=mx.array(self._to_np(camera_info_ref)).astype(mx.int64).reshape(-1) if camera_info_ref is not None else None,
            mva_scale=mva_scale,
            ref_scale=ref_scale,
            num_in_batch=num_in_batch,
            cache=self._cache,
        )
        mx.eval(output)

        if self._call_count <= 5 or self._call_count % 5 == 0:
            print(f"[MLX] UNet step {self._call_count}: {time.time() - t0:.2f}s")

        output_np = np.array(output)
        output_pt = torch.from_numpy(output_np).to(dtype=dtype, device=device)

        if return_dict:
            return {"sample": output_pt}
        return (output_pt,)

    @staticmethod
    def patch_pipeline(
        pipeline,
        model_path: str,
        weights_path: Optional[str] = None,
        profile: Optional[str] = None,
        pbr_albedo_only: bool = False,
    ):
        """Patch an existing diffusers HunyuanPaintPipeline to use MLX UNet."""
        hybrid = HybridMLXUNet(
            model_path, weights_path=weights_path, profile=profile, pbr_albedo_only=pbr_albedo_only
        )

        original_forward = pipeline.unet.forward
        pipeline.unet._original_forward = original_forward
        pipeline.unet.forward = hybrid.forward
        pipeline.unet._mlx_hybrid = hybrid

        # Aliases for pipeline compatibility (2.1 PBR uses albedo/mr instead of gen)
        # Apply to both the wrapper and the inner unet just in case
        if hasattr(pipeline.unet, "unet"):
            inner_unet = pipeline.unet.unet
            if not hasattr(inner_unet, "learned_text_clip_gen"):
                if hasattr(inner_unet, "learned_text_clip_albedo"):
                    inner_unet.learned_text_clip_gen = inner_unet.learned_text_clip_albedo
                elif hasattr(inner_unet, "learned_text_clip_ref"):
                    inner_unet.learned_text_clip_gen = inner_unet.learned_text_clip_ref

        print(f"[MLX] Pipeline patched ({hybrid.profile}) — UNet forward now uses MLX")
        return hybrid
