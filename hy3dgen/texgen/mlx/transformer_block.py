"""MLX port of Basic2p5DTransformerBlock from Hunyuan3D-Paint.

This is the core transformer block with 4 attention pathways:
1. MDA (Material-Dimension Attention) — per-PBR self-attention
2. RA (Reference Attention) — shared Q/K, per-material V
3. MA (Multiview Attention) — cross-view with 3D RoPE
4. Text cross-attention + DINO cross-attention
5. Feed-forward (GeGLU)

Weight keys follow the PyTorch checkpoint naming:
    transformer_blocks.N.transformer.norm1/2/3
    transformer_blocks.N.transformer.attn1.to_q/k/v/out
    transformer_blocks.N.transformer.attn1.processor.to_q_mr/...
    transformer_blocks.N.transformer.attn2.to_q/k/v/out
    transformer_blocks.N.transformer.linear1/2/3
    transformer_blocks.N.attn_multiview.to_q/k/v/out
    transformer_blocks.N.attn_refview.to_q/k/v/out + processor.to_v_mr/...
    transformer_blocks.N.attn_dino.to_q/k/v/out
"""

from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .attention import (
    MLXCrossAttnProcessor,
    MLXPoseRoPEAttnProcessor,
    MLXRefAttnProcessor,
    MLXSelfAttnProcessor,
)


class MLXTransformerBlock(nn.Module):
    """Enhanced transformer block for multiview 2.5D texture generation.

    Args:
        dim: Hidden state dimension
        num_heads: Number of attention heads
        dim_head: Dimension per head
        cross_attention_dim: Dimension of cross-attention context (CLIP/text)
        pbr_settings: List of PBR materials e.g. ["albedo", "mr"]
        use_ma: Enable multiview attention
        use_ra: Enable reference attention
        use_mda: Enable material-dimension attention
        use_dino: Enable DINO feature cross-attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        cross_attention_dim: int = 1024,
        pbr_settings: Optional[List[str]] = None,
        use_ma: bool = True,
        use_ra: bool = True,
        use_mda: bool = True,
        use_dino: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.use_mda = use_mda
        self.use_dino = use_dino
        self.pbr_settings = pbr_settings or ["albedo", "mr"]

        hidden_dim = dim * 4

        # === Norms (from transformer.norm1/2/3) ===
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # === 1. Self-attention (MDA) ===
        # When MDA is enabled, uses per-PBR projections
        # When disabled, uses standard self-attention (albedo path only)
        self.attn1 = MLXSelfAttnProcessor(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            pbr_settings=self.pbr_settings if use_mda else ["albedo"],
        )

        # === 2. Text cross-attention (attn2) ===
        self.attn2 = MLXCrossAttnProcessor(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_heads,
            dim_head=dim_head,
        )

        # === 3. Multiview attention (MA) with RoPE ===
        if use_ma:
            self.attn_multiview = MLXPoseRoPEAttnProcessor(
                query_dim=dim,
                heads=num_heads,
                dim_head=dim_head,
            )

        # === 4. Reference attention (RA) ===
        if use_ra:
            self.attn_refview = MLXRefAttnProcessor(
                query_dim=dim,
                heads=num_heads,
                dim_head=dim_head,
                pbr_settings=self.pbr_settings,
            )

        # === 5. DINO cross-attention ===
        if use_dino:
            self.attn_dino = MLXCrossAttnProcessor(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_heads,
                dim_head=dim_head,
            )

        # === Feed-forward (GeGLU) ===
        # linear1 = gate, linear2 = proj (from split GeGLU), linear3 = output
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, dim)

    @staticmethod
    def _broadcast_scale(scale, batch_size, num_in_batch, n_pbr, ndim):
        """Broadcast a scale factor to match hidden_states shape for elementwise multiply.

        Args:
            scale: float or mx.array [B] per-batch scale factors
            batch_size: B * N_pbr * num_in_batch (flat batch dim)
            num_in_batch: number of views
            n_pbr: number of PBR materials
            ndim: number of dims in attn_output (typically 3: [batch, seq, channels])

        Returns:
            float or mx.array broadcastable to attn_output shape
        """
        if isinstance(scale, (int, float)):
            return scale
        # scale is mx.array [B] — expand to [B*N_pbr*N, 1, 1]
        # Repeat for N_pbr and num_in_batch
        scale = mx.expand_dims(scale, axis=1)  # [B, 1]
        scale = mx.repeat(scale, num_in_batch * n_pbr, axis=1)  # [B, N_pbr*N]
        scale = scale.reshape(-1)  # [B*N_pbr*N]
        for _ in range(ndim - 1):
            scale = mx.expand_dims(scale, axis=-1)  # [B*N_pbr*N, 1, 1]
        return scale

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        num_in_batch: int = 1,
        mode: str = "r",
        mva_scale: Union[float, mx.array] = 1.0,
        ref_scale: Union[float, mx.array] = 1.0,
        condition_embed_dict: Optional[Dict] = None,
        dino_hidden_states: Optional[mx.array] = None,
        position_voxel_indices: Optional[Dict] = None,
        layer_name: str = "",
        n_pbr: int = 2,
    ) -> mx.array:
        """Forward pass with multi-mechanism attention.

        Args:
            hidden_states: [B*N_pbr*N_views, L, C]
            encoder_hidden_states: [B, seq_len, cross_dim] text embeddings
            num_in_batch: number of views per batch
            mode: "w" to write condition embeddings, "r" to read them, "wr" for both
            mva_scale: scaling for multiview attention residual
            ref_scale: scaling for reference attention residual
            condition_embed_dict: dict for storing/retrieving ref attention embeddings
            dino_hidden_states: [B, n_patches, dino_dim] DINO features
            position_voxel_indices: dict with voxel indices for RoPE
            layer_name: identifier for condition_embed_dict key
            n_pbr: number of PBR materials
        """
        batch_size = hidden_states.shape[0]

        # ── 1. Normalize + Self-attention (MDA) ──
        norm_hidden_states = self.norm1(hidden_states)

        if self.use_mda:
            # Reshape for PBR: [B*N_pbr*N, L, C] → [B, N_pbr, N, L, C]
            B = batch_size // (n_pbr * num_in_batch)
            mda_input = norm_hidden_states.reshape(B, n_pbr, num_in_batch, *norm_hidden_states.shape[1:])
            attn_output = self.attn1(mda_input)
            attn_output = attn_output.reshape(batch_size, *attn_output.shape[3:])
        else:
            # Simple self-attention: wrap in [B, 1, 1, L, C] format
            mda_input = mx.expand_dims(mx.expand_dims(norm_hidden_states, axis=1), axis=1)
            attn_output = self.attn1(mda_input)
            attn_output = attn_output.reshape(batch_size, *attn_output.shape[3:])

        hidden_states = attn_output + hidden_states

        # ── 2. Reference Attention (RA) ──
        # Write mode: store features for reference
        if "w" in mode and condition_embed_dict is not None:
            # [B*N_pbr*N, L, C] → [B, (N*L), C]  (only need one copy per batch)
            B = batch_size // (n_pbr * num_in_batch)
            ref_features = norm_hidden_states.reshape(B, n_pbr * num_in_batch, *norm_hidden_states.shape[1:])
            # Take first N_pbr views and concatenate spatial dims
            ref_features = ref_features[:, :num_in_batch]  # [B, N, L, C]
            ref_features = ref_features.reshape(B, -1, ref_features.shape[-1])  # [B, N*L, C]
            condition_embed_dict[layer_name] = ref_features

        # Read mode: condition on stored reference features
        if "r" in mode and self.use_ra and condition_embed_dict is not None:
            condition_embed = condition_embed_dict.get(layer_name)
            if condition_embed is not None:
                B = batch_size // (n_pbr * num_in_batch)

                # Use albedo features only for reference attention
                ref_input = norm_hidden_states.reshape(B, n_pbr, num_in_batch, *norm_hidden_states.shape[1:])
                ref_input = ref_input[:, 0]  # [B, N, L, C] — albedo only
                ref_input = ref_input.reshape(B, -1, ref_input.shape[-1])  # [B, N*L, C]

                attn_output = self.attn_refview(ref_input, condition_embed)
                # [B, N_pbr, N*L, C] → broadcast to all PBR channels
                if attn_output.ndim == 3:
                    attn_output = mx.expand_dims(attn_output, axis=1)
                    attn_output = mx.repeat(attn_output, n_pbr, axis=1)
                elif attn_output.shape[1] != n_pbr:
                    attn_output = mx.repeat(attn_output[:, :1], n_pbr, axis=1)

                # [B, N_pbr, N*L, C] → [B*N_pbr*N, L, C]
                seq_len = hidden_states.shape[1]
                attn_output = attn_output.reshape(B, n_pbr, num_in_batch, seq_len, -1)
                attn_output = attn_output.reshape(batch_size, seq_len, -1)

                ref_scale_b = self._broadcast_scale(ref_scale, batch_size, num_in_batch, n_pbr, attn_output.ndim)
                hidden_states = ref_scale_b * attn_output + hidden_states

        # ── 3. Multiview Attention (MA) with RoPE ──
        if num_in_batch > 1 and self.use_ma:
            B = batch_size // (n_pbr * num_in_batch)
            # [B*N_pbr*N, L, C] → [B*N_pbr, N*L, C]
            mv_input = norm_hidden_states.reshape(B * n_pbr, num_in_batch, *norm_hidden_states.shape[1:])
            mv_input = mv_input.reshape(B * n_pbr, -1, mv_input.shape[-1])

            # Get position indices for this sequence length
            position_indices = None
            if position_voxel_indices is not None:
                seq_key = mv_input.shape[1]
                if seq_key in position_voxel_indices:
                    position_indices = position_voxel_indices[seq_key]

            attn_output = self.attn_multiview(
                mv_input,
                encoder_hidden_states=mv_input,
                position_indices=position_indices,
                n_pbrs=n_pbr,
            )

            # [B*N_pbr, N*L, C] → [B*N_pbr*N, L, C]
            seq_len = hidden_states.shape[1]
            attn_output = attn_output.reshape(B * n_pbr, num_in_batch, seq_len, -1)
            attn_output = attn_output.reshape(batch_size, seq_len, -1)

            mva_scale_b = self._broadcast_scale(mva_scale, batch_size, num_in_batch, n_pbr, attn_output.ndim)
            hidden_states = mva_scale_b * attn_output + hidden_states

        # ── 4. Text Cross-Attention ──
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # ── 5. DINO Cross-Attention ──
        if self.use_dino and dino_hidden_states is not None:
            # Expand DINO features: [B, n_patches, C] → [B*N_pbr*N, n_patches, C]
            dino_expanded = mx.expand_dims(dino_hidden_states, axis=1)
            dino_expanded = mx.repeat(dino_expanded, n_pbr * num_in_batch, axis=1)
            dino_expanded = dino_expanded.reshape(-1, *dino_expanded.shape[2:])

            attn_output = self.attn_dino(norm_hidden_states, dino_expanded)
            hidden_states = attn_output + hidden_states

        # ── 6. Feed-Forward (GeGLU) ──
        norm_hidden_states = self.norm3(hidden_states)

        # GeGLU: gate * proj
        gate = self.linear1(norm_hidden_states)
        proj = self.linear2(norm_hidden_states)
        ff_output = gate * nn.gelu(proj)
        ff_output = self.linear3(ff_output)

        hidden_states = ff_output + hidden_states

        return hidden_states
