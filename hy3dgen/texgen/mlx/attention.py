"""Hunyuan3D-Paint attention processors ported to MLX.

These replace the PyTorch attention processors in attn_processor.py.
Key difference: all weights are self-contained in each module (no reaching into
diffusers' Attention class). Weight names are designed to match PyTorch checkpoint
keys after the weight conversion script maps them.

All attention uses mx.fast.scaled_dot_product_attention — the whole reason for
this port. No chunking needed.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ─── Rotary Position Embeddings ──────────────────────────────────────────────

class RotaryEmbedding:
    """Rotary position embedding utilities for 3D spatial attention."""

    @staticmethod
    def get_1d_rotary_pos_embed(
        dim: int, pos: mx.array, theta: float = 10000.0
    ) -> Tuple[mx.array, mx.array]:
        """Compute 1D rotary position embeddings.

        Returns (cos_embeddings, sin_embeddings) with shape [len(pos), dim].
        """
        assert dim % 2 == 0
        half_dim = dim // 2
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(pos.dtype)[:half_dim] / dim))
        freqs = mx.outer(pos, freqs)  # [len(pos), half_dim]
        # repeat_interleave(2) equivalent: [c0 c0 c1 c1 ...] for dim pairing
        cos_emb = mx.repeat(mx.cos(freqs), 2, axis=1).astype(mx.float32)
        sin_emb = mx.repeat(mx.sin(freqs), 2, axis=1).astype(mx.float32)
        return cos_emb, sin_emb

    @staticmethod
    def get_3d_rotary_pos_embed(
        position: mx.array, embed_dim: int, voxel_resolution: int, theta: int = 10000
    ) -> Tuple[mx.array, mx.array]:
        """Compute 3D rotary position embeddings for spatial coordinates.

        Args:
            position: [..., 3] voxel indices
            embed_dim: head dimension
            voxel_resolution: resolution of the voxel grid
        """
        assert position.shape[-1] == 3
        dim_xy = embed_dim // 8 * 3
        dim_z = embed_dim // 8 * 2

        grid = mx.arange(voxel_resolution).astype(mx.float32)
        xy_cos, xy_sin = RotaryEmbedding.get_1d_rotary_pos_embed(dim_xy, grid, theta)
        z_cos, z_sin = RotaryEmbedding.get_1d_rotary_pos_embed(dim_z, grid, theta)

        # Flatten position for indexing
        orig_shape = position.shape[:-1]
        flat = position.reshape(-1, 3).astype(mx.int32)

        x_cos = xy_cos[flat[:, 0]]
        x_sin = xy_sin[flat[:, 0]]
        y_cos = xy_cos[flat[:, 1]]
        y_sin = xy_sin[flat[:, 1]]
        zc = z_cos[flat[:, 2]]
        zs = z_sin[flat[:, 2]]

        cos = mx.concatenate([x_cos, y_cos, zc], axis=-1)
        sin = mx.concatenate([x_sin, y_sin, zs], axis=-1)

        cos = cos.reshape(*orig_shape, embed_dim)
        sin = sin.reshape(*orig_shape, embed_dim)
        return cos, sin

    @staticmethod
    def apply_rotary_emb(
        x: mx.array, freqs_cis: Tuple[mx.array, mx.array]
    ) -> mx.array:
        """Apply rotary position embeddings.

        x: [B, H, N, D]
        freqs_cis: (cos, sin) each [B*N_pbrs, N, D] — will be unsqueezed for heads dim
        """
        cos, sin = freqs_cis
        # Add heads dimension: [B, 1, N, D]
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)

        # Split into real/imag pairs and rotate
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshape[..., 0]
        x_imag = x_reshape[..., 1]
        # Interleave [-imag, real] to match cos/sin frequency pairing
        x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)

        out = (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)
        return out


# ─── Attention Helper ────────────────────────────────────────────────────────

def _sdpa(query, key, value, mask=None):
    """Scaled dot-product attention using MLX's fused Metal kernel."""
    scale = query.shape[-1] ** -0.5
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)


def _reshape_for_attention(tensor, batch_size, num_heads, head_dim):
    """Reshape [B*N, L, inner_dim] → [B*N, H, L, D] for multi-head attention."""
    seq_len = tensor.shape[1]
    tensor = tensor.reshape(batch_size, seq_len, num_heads, head_dim)
    return tensor.transpose(0, 2, 1, 3)  # [B, H, L, D]


# ─── PoseRoPE Attention (Multiview) ─────────────────────────────────────────

class MLXPoseRoPEAttnProcessor(nn.Module):
    """Multiview attention with 3D Rotary Position Embeddings.

    Weight keys from checkpoint:
        attn_multiview.to_q.weight, attn_multiview.to_k.weight,
        attn_multiview.to_v.weight, attn_multiview.to_out.0.weight/bias
    """

    def __init__(self, query_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out_0 = nn.Linear(inner_dim, query_dim)  # to_out.0 in checkpoint

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_indices: Optional[Dict] = None,
        n_pbrs: int = 1,
    ) -> mx.array:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = _reshape_for_attention(query, batch_size, self.heads, self.dim_head)
        key = _reshape_for_attention(key, batch_size, self.heads, self.dim_head)
        value = _reshape_for_attention(value, batch_size, self.heads, self.dim_head)

        # Apply 3D RoPE
        if position_indices is not None:
            head_dim = self.dim_head
            if head_dim in position_indices:
                image_rotary_emb = position_indices[head_dim]
            else:
                # Compute RoPE from voxel indices
                voxel_indices = position_indices["voxel_indices"]
                # Expand for PBR: [B, L, 3] → [B*n_pbrs, L, 3]
                voxel_expanded = mx.repeat(
                    mx.expand_dims(voxel_indices, axis=1), n_pbrs, axis=1
                ).reshape(-1, *voxel_indices.shape[1:])

                image_rotary_emb = RotaryEmbedding.get_3d_rotary_pos_embed(
                    voxel_expanded, head_dim,
                    voxel_resolution=position_indices["voxel_resolution"],
                )
                position_indices[head_dim] = image_rotary_emb

            query = RotaryEmbedding.apply_rotary_emb(query, image_rotary_emb)
            key = RotaryEmbedding.apply_rotary_emb(key, image_rotary_emb)

        # Attention
        hidden_states = _sdpa(query, key, value, mask=attention_mask)

        # Reshape back: [B, H, L, D] → [B, L, H*D]
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.heads * self.dim_head
        )

        # Output projection
        hidden_states = self.to_out_0(hidden_states)
        return hidden_states


# ─── Self-Attention with PBR Materials ───────────────────────────────────────

class MLXSelfAttnProcessor(nn.Module):
    """Self-attention with separate Q/K/V per PBR material.

    "albedo" uses the base to_q/k/v/out projections.
    Other materials (e.g., "mr") use to_q_mr/k_mr/v_mr/out_mr.

    Weight keys from checkpoint:
        transformer.attn1.to_q.weight (albedo)
        transformer.attn1.processor.to_q_mr.weight (mr)
        etc.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        pbr_settings: List[str],
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.pbr_settings = pbr_settings
        cross_attention_dim = cross_attention_dim or query_dim

        # Albedo projections (base)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out_0 = nn.Linear(inner_dim, query_dim)

        # Per-material projections (non-albedo)
        for token in pbr_settings:
            if token != "albedo":
                setattr(self, f"to_q_{token}", nn.Linear(query_dim, inner_dim, bias=False))
                setattr(self, f"to_k_{token}", nn.Linear(cross_attention_dim, inner_dim, bias=False))
                setattr(self, f"to_v_{token}", nn.Linear(cross_attention_dim, inner_dim, bias=False))
                setattr(self, f"to_out_{token}_0", nn.Linear(inner_dim, query_dim))

    def _process_single(
        self,
        hidden_states: mx.array,
        token: str,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Process attention for a single PBR material."""
        batch_size = hidden_states.shape[0]

        if token == "albedo":
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            out_proj = self.to_out_0
        else:
            query = getattr(self, f"to_q_{token}")(hidden_states)
            key = getattr(self, f"to_k_{token}")(hidden_states)
            value = getattr(self, f"to_v_{token}")(hidden_states)
            out_proj = getattr(self, f"to_out_{token}_0")

        query = _reshape_for_attention(query, batch_size, self.heads, self.dim_head)
        key = _reshape_for_attention(key, batch_size, self.heads, self.dim_head)
        value = _reshape_for_attention(value, batch_size, self.heads, self.dim_head)

        attn_out = _sdpa(query, key, value, mask=attention_mask)

        # [B, H, L, D] → [B, L, H*D]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.heads * self.dim_head
        )
        return out_proj(attn_out)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Process attention for all PBR materials.

        Input hidden_states: [B, N_pbr, N_views, L, C]
        Output: same shape
        """
        B = hidden_states.shape[0]
        results = []

        for i, token in enumerate(self.pbr_settings):
            # Extract single PBR: [B, 1, N_views, L, C] → [B*N_views, L, C]
            pbr_hs = hidden_states[:, i:i+1]
            n_views = pbr_hs.shape[2]
            flat_hs = pbr_hs.reshape(-1, pbr_hs.shape[3], pbr_hs.shape[4])

            result = self._process_single(flat_hs, token, attention_mask)

            # Reshape back: [B*N_views, L, C] → [B, 1, N_views, L, C]
            result = result.reshape(B, 1, n_views, *result.shape[1:])
            results.append(result)

        return mx.concatenate(results, axis=1)


# ─── Reference Attention ─────────────────────────────────────────────────────

class MLXRefAttnProcessor(nn.Module):
    """Reference attention: shared Q/K, separate V per PBR material.

    Weight keys from checkpoint:
        attn_refview.to_q.weight, attn_refview.to_k.weight
        attn_refview.to_v.weight (albedo V)
        attn_refview.processor.to_v_mr.weight (mr V)
        attn_refview.to_out.0.weight/bias (albedo out)
        attn_refview.processor.to_out_mr.0.weight/bias (mr out)
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        pbr_settings: List[str],
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.pbr_settings = pbr_settings
        cross_attention_dim = cross_attention_dim or query_dim

        # Shared Q/K — Q comes from hidden_states (query_dim), K from encoder (query_dim for self-attn)
        # In the PyTorch code, RefAttn uses attn.to_q/to_k which both have query_dim input
        # because encoder_hidden_states is the same dim as hidden_states for ref attention
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)

        # Per-material V and output — V also uses query_dim (same source as K)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)  # albedo
        self.to_out_0 = nn.Linear(inner_dim, query_dim)  # albedo

        for token in pbr_settings:
            if token != "albedo":
                setattr(self, f"to_v_{token}", nn.Linear(query_dim, inner_dim, bias=False))
                setattr(self, f"to_out_{token}_0", nn.Linear(inner_dim, query_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Input hidden_states: [B, L, C]
        Output: [N_pbr, B, L, C] stacked per material
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = hidden_states.shape[0]

        # Shared Q/K
        query = self.to_q(hidden_states)
        
        # Align encoder batch size if needed
        if encoder_hidden_states is not None and encoder_hidden_states.shape[0] != batch_size:
            if encoder_hidden_states.shape[0] == 1:
                encoder_hidden_states = mx.repeat(encoder_hidden_states, batch_size, axis=0)
            else:
                encoder_hidden_states = encoder_hidden_states[:batch_size]

        key = self.to_k(encoder_hidden_states)

        # Concatenate all material V projections
        value_list = [self.to_v(encoder_hidden_states)]
        for token in self.pbr_settings:
            if token != "albedo":
                value_list.append(getattr(self, f"to_v_{token}")(encoder_hidden_states))
        value = mx.concatenate(value_list, axis=-1)

        # Reshape for multi-head attention
        query = _reshape_for_attention(query, batch_size, self.heads, self.dim_head)
        key = _reshape_for_attention(key, batch_size, self.heads, self.dim_head)
        # Value has n_pbr * inner_dim, so head_dim becomes n_pbr * dim_head
        value = _reshape_for_attention(
            value, batch_size, self.heads, self.dim_head * len(self.pbr_settings)
        )

        # Attention
        attn_out = _sdpa(query, key, value, mask=attention_mask)

        # Split output by PBR material along head_dim
        outputs = []
        for i, token in enumerate(self.pbr_settings):
            # Extract this material's slice: [B, H, L, dim_head]
            hs = attn_out[..., i * self.dim_head:(i + 1) * self.dim_head]
            # [B, H, L, D] → [B, L, H*D]
            hs = hs.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads * self.dim_head)

            if token == "albedo":
                hs = self.to_out_0(hs)
            else:
                hs = getattr(self, f"to_out_{token}_0")(hs)
            outputs.append(hs)

        return mx.stack(outputs, axis=1)  # [B, N_pbr, L, C]


# ─── Standard Cross-Attention (for text and DINO) ───────────────────────────

class MLXCrossAttnProcessor(nn.Module):
    """Standard cross-attention for text conditioning and DINO features.

    Used for attn2 (text cross-attention) and attn_dino.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out_0 = nn.Linear(inner_dim, query_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = _reshape_for_attention(query, batch_size, self.heads, self.dim_head)
        key = _reshape_for_attention(key, batch_size, self.heads, self.dim_head)
        value = _reshape_for_attention(value, batch_size, self.heads, self.dim_head)

        attn_out = _sdpa(query, key, value, mask=attention_mask)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.heads * self.dim_head
        )
        return self.to_out_0(attn_out)
