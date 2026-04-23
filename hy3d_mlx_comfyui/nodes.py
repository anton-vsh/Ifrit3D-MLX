from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


SHAPE_PRESETS: Dict[str, Tuple[str, str]] = {
    "mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"),
    "mini-turbo": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"),
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1"),
    "mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv"),
    "mv-turbo": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"),
}

PAINT_PRESETS: Dict[str, Tuple[str, str]] = {
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"),
}


@dataclass
class HunyuanMesh:
    mesh: Any
    source: str
    textured: bool = False


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_output_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        out_dir = PACKAGE_DIR / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


def _next_output_path(prefix: str, extension: str = ".glb") -> Path:
    base_dir = _get_output_dir()
    stem = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in prefix).strip("_") or "hy3d"
    candidate = base_dir / f"{stem}{extension}"
    index = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{index}{extension}"
        index += 1
    return candidate


def _tensor_to_pil(image: Any):
    import numpy as np
    from PIL import Image

    if image is None:
        return None

    if hasattr(image, "detach"):
        image = image.detach().cpu()

    if getattr(image, "ndim", None) == 4:
        if image.shape[0] < 1:
            raise ValueError("Expected at least one image in the IMAGE batch.")
        image = image[0]

    arr = image.numpy()
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255.0).round().astype("uint8")

    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    if arr.shape[-1] == 3:
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported IMAGE channel count: {arr.shape[-1]}")


def _maybe_remove_bg(image, enabled: bool, remover=None):
    if image is None:
        return None
    if not enabled:
        return image.convert("RGBA")
    if remover is None:
        from hy3dgen.rembg import BackgroundRemover

        remover = BackgroundRemover()
    if image.mode == "RGB":
        image = remover(image)
    return image.convert("RGBA")


def _prepare_images(front_image, left_image=None, back_image=None, remove_bg: bool = True):
    front = _tensor_to_pil(front_image)
    left = _tensor_to_pil(left_image) if left_image is not None else None
    back = _tensor_to_pil(back_image) if back_image is not None else None

    if (left is None) ^ (back is None):
        raise ValueError("Provide both left and back images for multiview inputs, or leave both disconnected.")

    remover = None
    if remove_bg:
        from hy3dgen.rembg import BackgroundRemover

        remover = BackgroundRemover()

    if left is None and back is None:
        return "sv", [_maybe_remove_bg(front, remove_bg, remover=remover)]

    return "mv", [
        _maybe_remove_bg(front, remove_bg, remover=remover),
        _maybe_remove_bg(left, remove_bg, remover=remover),
        _maybe_remove_bg(back, remove_bg, remover=remover),
    ]


def _load_shape_pipeline(model_repo: str, subfolder: str, variant: str, use_safetensors: bool):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    kwargs = {
        "model_path": model_repo,
        "subfolder": subfolder,
        "device": _pick_device(),
        "use_safetensors": use_safetensors,
    }
    if variant:
        kwargs["variant"] = variant
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(**kwargs)


def _load_paint_pipeline(model_repo: str, subfolder: str, diffusion_backend: str, mlx_weights_path: str):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return Hunyuan3DPaintPipeline.from_pretrained(
        model_repo,
        subfolder=subfolder,
        diffusion_backend=diffusion_backend,
        mlx_weights_path=mlx_weights_path or None,
    )


class Hunyuan3DShapeNode:
    CATEGORY = "Hunyuan3D"
    FUNCTION = "generate"
    RETURN_TYPES = ("HY3D_MESH",)
    RETURN_NAMES = ("mesh",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_image": ("IMAGE",),
                "model": (tuple(SHAPE_PRESETS.keys()), {"default": "mini-turbo"}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0x7FFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "num_chunks": ("INT", {"default": 12000, "min": 1000, "max": 50000, "step": 500}),
                "shape_variant": ("STRING", {"default": "fp16", "multiline": False}),
                "use_safetensors": ("BOOLEAN", {"default": True}),
                "remove_background": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "left_image": ("IMAGE",),
                "back_image": ("IMAGE",),
            },
        }

    def generate(
        self,
        front_image,
        model,
        seed,
        steps,
        octree_resolution,
        num_chunks,
        shape_variant,
        use_safetensors,
        remove_background,
        left_image=None,
        back_image=None,
    ):
        import torch

        mode, images = _prepare_images(
            front_image,
            left_image=left_image,
            back_image=back_image,
            remove_bg=remove_background,
        )

        if mode == "mv" and model not in {"mv", "mv-turbo"}:
            raise ValueError("Multiview shape generation only supports the mv and mv-turbo shape presets.")
        if mode == "sv" and model in {"mv", "mv-turbo"}:
            raise ValueError("The mv and mv-turbo shape presets require front, left, and back images.")

        model_repo, subfolder = SHAPE_PRESETS[model]
        pipeline = _load_shape_pipeline(
            model_repo=model_repo,
            subfolder=subfolder,
            variant=shape_variant,
            use_safetensors=use_safetensors,
        )

        generator = torch.manual_seed(seed)
        image_input = images[0] if mode == "sv" else {"front": images[0], "left": images[1], "back": images[2]}
        mesh = pipeline(
            image=image_input,
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=generator,
            output_type="trimesh",
        )[0]
        return (HunyuanMesh(mesh=mesh, source=f"{model_repo}/{subfolder}", textured=False),)


class Hunyuan3DPaintNode:
    CATEGORY = "Hunyuan3D"
    FUNCTION = "paint"
    RETURN_TYPES = ("HY3D_MESH",)
    RETURN_NAMES = ("textured_mesh",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("HY3D_MESH",),
                "front_image": ("IMAGE",),
                "model": (tuple(PAINT_PRESETS.keys()), {"default": "2.0-turbo"}),
                "render_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "texture_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "diffusion_backend": (("pytorch", "mlx"), {"default": "mlx"}),
                "mlx_weights_path": ("STRING", {"default": "", "multiline": False}),
                "remove_background": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "left_image": ("IMAGE",),
                "back_image": ("IMAGE",),
            },
        }

    def paint(
        self,
        mesh,
        front_image,
        model,
        render_size,
        texture_size,
        diffusion_backend,
        mlx_weights_path,
        remove_background,
        left_image=None,
        back_image=None,
    ):
        mode, images = _prepare_images(
            front_image,
            left_image=left_image,
            back_image=back_image,
            remove_bg=remove_background,
        )

        model_repo, subfolder = PAINT_PRESETS[model]
        painter = _load_paint_pipeline(
            model_repo=model_repo,
            subfolder=subfolder,
            diffusion_backend=diffusion_backend,
            mlx_weights_path=mlx_weights_path,
        )
        painter.config.render_size = render_size
        painter.config.texture_size = texture_size
        painter.render.set_default_render_resolution(render_size)
        painter.render.set_default_texture_resolution(texture_size)

        image_arg = images if mode == "mv" else images[0]
        textured = painter(mesh.mesh, image=image_arg)
        return (HunyuanMesh(mesh=textured, source=f"{model_repo}/{subfolder}", textured=True),)


class Hunyuan3DLoadMeshNode:
    CATEGORY = "Hunyuan3D"
    FUNCTION = "load_mesh"
    RETURN_TYPES = ("HY3D_MESH",)
    RETURN_NAMES = ("mesh",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def load_mesh(self, mesh_path):
        import trimesh

        path = Path(mesh_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Mesh path does not exist: {path}")
        mesh = trimesh.load(path, force="mesh")
        return (HunyuanMesh(mesh=mesh, source=str(path), textured=False),)


class Hunyuan3DSaveMeshNode:
    CATEGORY = "Hunyuan3D"
    FUNCTION = "save_mesh"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("HY3D_MESH",),
                "filename_prefix": ("STRING", {"default": "hunyuan3d", "multiline": False}),
            }
        }

    def save_mesh(self, mesh, filename_prefix):
        out_path = _next_output_path(filename_prefix, extension=".glb")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.mesh.export(out_path)
        return (str(out_path),)


NODE_CLASS_MAPPINGS = {
    "Hunyuan3DShape": Hunyuan3DShapeNode,
    "Hunyuan3DPaint": Hunyuan3DPaintNode,
    "Hunyuan3DLoadMesh": Hunyuan3DLoadMeshNode,
    "Hunyuan3DSaveMesh": Hunyuan3DSaveMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DShape": "Hunyuan Shape",
    "Hunyuan3DPaint": "Hunyuan Paint",
    "Hunyuan3DLoadMesh": "Hunyuan Load Mesh",
    "Hunyuan3DSaveMesh": "Hunyuan Save Mesh",
}
