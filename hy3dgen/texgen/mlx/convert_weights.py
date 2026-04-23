"""Convert Hunyuan paint checkpoints from PyTorch -> MLX npz weights.

Supports two profiles:
- paint-2.0      (legacy non-PBR runtime in this repo)
- paint-pbr-2.1  (PBR paint checkpoint)

Example:
    python -m hy3dgen.texgen.mlx.convert_weights \
      --model-path ~/.cache/hy3dgen/tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0 \
      --profile paint-2.0
"""

import argparse
import os
import re

import numpy as np

PROFILE_PAINT_20 = "paint-2.0"
PROFILE_PAINT_PBR_21 = "paint-pbr-2.1"

# Lazy imports so module can be imported without torch/mlx installed.
_torch = None
_mx = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_mx():
    global _mx
    if _mx is None:
        import mlx.core
        _mx = mlx.core
    return _mx


def _infer_profile(model_path: str, profile: str | None) -> str:
    if profile in {PROFILE_PAINT_20, PROFILE_PAINT_PBR_21}:
        return profile
    name = os.path.basename(model_path).lower()
    if "paintpbr" in name or "v2-1" in name:
        return PROFILE_PAINT_PBR_21
    return PROFILE_PAINT_20


def _default_output_dir(model_path: str, profile: str) -> str:
    parent = os.path.dirname(model_path)
    if profile == PROFILE_PAINT_20:
        return os.path.join(parent, "hunyuan3d-2.0-mlx")
    return os.path.join(parent, "hunyuan3d-2.1-mlx")


def _resolve_model_subfolder(model_path: str, profile: str) -> str:
    """Accept either subfolder path (contains unet/vae) or repo root path."""
    if os.path.isdir(os.path.join(model_path, "unet")) and os.path.isdir(os.path.join(model_path, "vae")):
        return model_path

    if profile == PROFILE_PAINT_20:
        candidates = [
            os.path.join(model_path, "hunyuan3d-paint-v2-0"),
            os.path.join(model_path, "hunyuan3d-paint-v2-0-turbo"),
        ]
    else:
        candidates = [
            os.path.join(model_path, "hunyuan3d-paintpbr-v2-1"),
            os.path.join(model_path, "hunyuan3d-paint-v2-1"),
        ]

    for c in candidates:
        if os.path.isdir(os.path.join(c, "unet")) and os.path.isdir(os.path.join(c, "vae")):
            return c

    raise FileNotFoundError(
        f"Could not locate a paint model subfolder with unet/ + vae/ under: {model_path}"
    )


def _transpose_conv_weight(arr):
    if arr.ndim == 4:
        # PyTorch conv: (out, in, kh, kw) -> MLX conv: (out, kh, kw, in)
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _split_geglu_weight(weight):
    h = weight.shape[0] // 2
    return weight[:h], weight[h:]


def _split_geglu_bias(bias):
    h = bias.shape[0] // 2
    return bias[:h], bias[h:]


def _remap_transformer_block_keys(old_key, prefix):
    if not old_key.startswith(prefix + "."):
        return None, None

    local = old_key[len(prefix) + 1:]

    if local == "transformer.ff.net.0.proj.weight":
        return prefix + ".transformer.linear1.weight", "geglu_split_1"
    if local == "transformer.ff.net.0.proj.bias":
        return prefix + ".transformer.linear1.bias", "geglu_split_1"
    if local == "transformer.ff.net.2.weight":
        return prefix + ".transformer.linear3.weight", None
    if local == "transformer.ff.net.2.bias":
        return prefix + ".transformer.linear3.bias", None

    return old_key, None


def _remap_key_for_mlx_model(key: str, profile: str):
    k = key

    # Top-level projections
    k = k.replace("unet.image_proj_model_dino.", "image_proj_model_dino.")

    # Learned text embeddings differ between legacy and pbr checkpoints.
    if profile == PROFILE_PAINT_20:
        # Legacy uses learned_text_clip_gen/ref; MLX model stores albedo/ref.
        k = k.replace("unet.learned_text_clip_gen", "learned_text_clip_albedo")
        k = k.replace("unet.learned_text_clip_ref", "learned_text_clip_ref")
    else:
        k = k.replace("unet.learned_text_clip_", "learned_text_clip_")

    # Mid block flattening (diffusers -> MLX structure)
    k = k.replace("unet.mid_block.resnets.0.", "unet.mid_resnets_0.")
    k = k.replace("unet.mid_block.attentions.0.", "unet.mid_attentions_0.")
    k = k.replace("unet.mid_block.resnets.1.", "unet.mid_resnets_1.")

    k = k.replace("unet_dual.mid_block.resnets.0.", "unet_dual.mid_resnets_0.")
    k = k.replace("unet_dual.mid_block.attentions.0.", "unet_dual.mid_attentions_0.")
    k = k.replace("unet_dual.mid_block.resnets.1.", "unet_dual.mid_resnets_1.")

    # Inside transformer blocks: drop extra .transformer. nesting
    k = re.sub(
        r"(transformer_blocks\.\d+)\.transformer\.(attn1|attn2|norm1|norm2|norm3|linear1|linear2|linear3)\.",
        r"\1.\2.",
        k,
    )

    # Flatten optional PBR processor submodule
    k = re.sub(r"\.processor\.to_", ".to_", k)

    # ModuleList index normalization
    k = k.replace(".to_out.0.", ".to_out_0.")
    k = k.replace(".to_out_mr.0.", ".to_out_mr_0.")

    # Down/upsampler naming
    k = k.replace(".downsamplers.0.conv.", ".downsample.")
    k = k.replace(".downsamplers.0.", ".downsample.")
    k = k.replace(".upsamplers.0.conv.", ".upsample.")
    k = k.replace(".upsamplers.0.", ".upsample.")

    return k


def convert_unet_weights(pytorch_state_dict, profile: str):
    mlx_weights = {}

    for key, tensor in pytorch_state_dict.items():
        arr = tensor.cpu().numpy()
        new_key = key
        transform = None

        parts = key.split(".")
        for i in range(len(parts)):
            if parts[i] == "transformer_blocks" and i + 1 < len(parts):
                block_prefix = ".".join(parts[: i + 2])
                new_key, transform = _remap_transformer_block_keys(key, block_prefix)
                if new_key is None:
                    new_key = key
                break

        if transform == "geglu_split_1":
            gate, proj = _split_geglu_weight(arr) if arr.ndim == 2 else _split_geglu_bias(arr)
            mlx_weights[new_key] = gate
            mlx_weights[new_key.replace("linear1", "linear2")] = proj
            continue

        if "conv_shortcut" in new_key and arr.ndim == 4:
            arr = arr.squeeze(-1).squeeze(-1)
            mlx_weights[new_key] = arr
            continue

        if arr.ndim == 4:
            arr = _transpose_conv_weight(arr)

        mlx_weights[new_key] = arr

    remapped = {}
    for k, v in mlx_weights.items():
        remapped[_remap_key_for_mlx_model(k, profile)] = v

    # Legacy checkpoints may carry keys that are not used by our MLX graph.
    # Keep them only if they're likely relevant.
    if profile == PROFILE_PAINT_20:
        remapped = {
            k: v
            for k, v in remapped.items()
            if not k.startswith("learned_text_clip_gen")
        }

    return remapped


def convert_vae_weights(pytorch_state_dict):
    mlx_weights = {}

    for key, tensor in pytorch_state_dict.items():
        arr = tensor.cpu().numpy()
        new_key = key

        if key in ("quant_conv.weight", "post_quant_conv.weight"):
            new_key = key.replace("_conv", "_proj")
            if arr.ndim == 4:
                arr = arr.squeeze(-1).squeeze(-1)
            mlx_weights[new_key] = arr
            continue

        if key in ("quant_conv.bias", "post_quant_conv.bias"):
            new_key = key.replace("_conv", "_proj")
            mlx_weights[new_key] = arr
            continue

        if "mid_block." in key:
            new_key = key.replace("mid_block.resnets.0.", "mid_blocks.0.")
            new_key = new_key.replace("mid_block.attentions.0.", "mid_blocks.1.")
            new_key = new_key.replace("mid_block.resnets.1.", "mid_blocks.2.")

            if "mid_blocks.1." in new_key:
                new_key = new_key.replace(".to_q.", ".query_proj.")
                new_key = new_key.replace(".to_k.", ".key_proj.")
                new_key = new_key.replace(".to_v.", ".value_proj.")
                new_key = new_key.replace(".to_out.0.", ".out_proj.")
                new_key = new_key.replace(".query.", ".query_proj.")
                new_key = new_key.replace(".key.", ".key_proj.")
                new_key = new_key.replace(".value.", ".value_proj.")
                new_key = new_key.replace(".proj_attn.", ".out_proj.")

        if "downsamplers.0." in new_key:
            new_key = new_key.replace("downsamplers.0.conv.", "downsample.")
            new_key = new_key.replace("downsamplers.0.", "downsample.")
        if "upsamplers.0." in new_key:
            new_key = new_key.replace("upsamplers.0.conv.", "upsample.")
            new_key = new_key.replace("upsamplers.0.", "upsample.")

        if "conv_shortcut" in new_key and arr.ndim == 4:
            arr = arr.squeeze(-1).squeeze(-1)
            mlx_weights[new_key] = arr
            continue

        if arr.ndim == 4:
            arr = _transpose_conv_weight(arr)

        mlx_weights[new_key] = arr

    return mlx_weights


def convert_and_save(model_path: str, output_dir: str | None = None, profile: str | None = None):
    torch = _get_torch()
    _get_mx()  # ensure mlx is installed early for clearer errors

    profile = _infer_profile(model_path, profile)
    model_path = _resolve_model_subfolder(model_path, profile)
    if output_dir is None:
        output_dir = _default_output_dir(model_path, profile)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[MLX] profile={profile}")
    print(f"[MLX] model_path={model_path}")
    print(f"[MLX] output_dir={output_dir}")

    unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.bin")
    vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.bin")

    print("\nLoading UNet weights...")
    unet_sd = torch.load(unet_path, map_location="cpu", weights_only=True)
    print(f"  {len(unet_sd)} PyTorch keys loaded")

    print("Converting UNet weights...")
    unet_mlx = convert_unet_weights(unet_sd, profile)
    print(f"  {len(unet_mlx)} MLX keys produced")

    conv_count = sum(1 for _, v in unet_mlx.items() if getattr(v, "ndim", 0) == 4)
    print(f"  {conv_count} conv weights transposed to NHWC")

    unet_out = os.path.join(output_dir, "unet.npz")
    np.savez(unet_out, **unet_mlx)
    print(f"  Saved {unet_out} ({os.path.getsize(unet_out) / 1024 ** 2:.1f} MB)")

    del unet_sd, unet_mlx

    print("\nLoading VAE weights...")
    vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
    print(f"  {len(vae_sd)} PyTorch keys loaded")

    print("Converting VAE weights...")
    vae_mlx = convert_vae_weights(vae_sd)
    print(f"  {len(vae_mlx)} MLX keys produced")

    vae_out = os.path.join(output_dir, "vae.npz")
    np.savez(vae_out, **vae_mlx)
    print(f"  Saved {vae_out} ({os.path.getsize(vae_out) / 1024 ** 2:.1f} MB)")

    print("\nDone.")
    return output_dir, profile


def validate_conversion(model_path: str, output_dir: str | None = None, profile: str | None = None):
    torch = _get_torch()

    profile = _infer_profile(model_path, profile)
    model_path = _resolve_model_subfolder(model_path, profile)
    if output_dir is None:
        output_dir = _default_output_dir(model_path, profile)

    print("=== Validating UNet conversion ===")
    unet_pt = torch.load(
        os.path.join(model_path, "unet", "diffusion_pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    unet_mlx = dict(np.load(os.path.join(output_dir, "unet.npz"), allow_pickle=True))

    print(f"  PyTorch keys: {len(unet_pt)}")
    print(f"  MLX keys: {len(unet_mlx)}")

    for k, v in unet_mlx.items():
        if v.ndim == 4:
            assert v.shape[1] == v.shape[2], f"Conv weight looks wrong: {k} {v.shape}"

    print("=== Validating VAE conversion ===")
    vae_mlx = dict(np.load(os.path.join(output_dir, "vae.npz"), allow_pickle=True))
    for k, v in vae_mlx.items():
        if v.ndim == 4:
            assert v.shape[1] == v.shape[2], f"Conv weight looks wrong: {k} {v.shape}"

    print("Validation passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hunyuan paint weights to MLX format")
    parser.add_argument("--model-path", required=True, help="Path to paint model subfolder (contains unet/ and vae/)")
    parser.add_argument(
        "--profile",
        choices=[PROFILE_PAINT_20, PROFILE_PAINT_PBR_21],
        default=None,
        help="Conversion profile (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: sibling hunyuan3d-2.0-mlx or hunyuan3d-2.1-mlx)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate after conversion")
    args = parser.parse_args()

    output_dir, profile = convert_and_save(args.model_path, args.output_dir, args.profile)
    if args.validate:
        validate_conversion(args.model_path, output_dir=output_dir, profile=profile)
