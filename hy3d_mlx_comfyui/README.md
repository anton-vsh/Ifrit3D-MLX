# HY3D MLX ComfyUI Nodes

This folder is meant to be copied into `ComfyUI/custom_nodes/` as a custom node package.

Included nodes:

- `Hunyuan Shape`
- `Hunyuan Paint`
- `Hunyuan Load Mesh`
- `Hunyuan Save Mesh`

Supported presets:

- Shape: `mini`, `mini-turbo`, `2.0`, `2.0-turbo`, `2.1`, `mv`, `mv-turbo`
- Paint: `2.0`, `2.0-turbo`

## Install

Recommended install flow:

1. Clone this repo somewhere on the same machine as ComfyUI.
2. Install the Hunyuan runtime dependencies into the same Python environment ComfyUI uses.
3. Copy or symlink `hy3d_mlx_comfyui/` into `ComfyUI/custom_nodes/`.
4. Restart ComfyUI.

If your ComfyUI environment uses `uv`, install from the repo root:

```bash
cd /path/to/Hunyuan3D-MLX
uv sync
```

If your ComfyUI environment uses `pip`, install from the repo root into that same environment:

```bash
cd /path/to/Hunyuan3D-MLX
pip install -r hy3d_mlx_comfyui/requirements.txt
pip install -e libraries/mtldiffrast
pip install -e libraries/mtlbvh
pip install -e libraries/mtlmesh
pip install -e libraries/mtlgemm
```

Then install the custom node itself:

```bash
cd /path/to/ComfyUI/custom_nodes
ln -s /path/to/Hunyuan3D-MLX/hy3d_mlx_comfyui ./hy3d_mlx_comfyui
```

Or just copy the folder:

```bash
cp -R /path/to/Hunyuan3D-MLX/hy3d_mlx_comfyui /path/to/ComfyUI/custom_nodes/
```

## Notes

- The vendored Python runtime lives under `hy3d_mlx_comfyui/vendor/hy3dgen`.
- This package vendors the Hunyuan Python code, but it still relies on native/runtime dependencies such as `torch`, `diffusers`, `rembg`, `mtldiffrast`, `mtlbvh`, `cumesh`, and `flex-gemm`.
- The save node writes `.glb` files into ComfyUI's output directory when available, and falls back to `hy3d_mlx_comfyui/outputs/` otherwise.
- Paint intentionally does not expose the 2.1 paint models in this package.