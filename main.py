from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
SV_DIR = IMAGES_DIR / "sv"
MV_DIR = IMAGES_DIR / "mv"
PENGUIN_IMAGE = IMAGES_DIR / "penguin.png"
OUTPUTS_DIR = ROOT / "outputs"

from shape.runner import SHAPE_PRESETS, run_shape_command, run_shape_pipeline

PAINT_PRESETS = {
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"),
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-paintpbr-v2-1"),
}


def parse_mode_and_index(selector: Optional[str], index: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if selector in {"sv", "mv"}:
        return selector, index
    if selector is None:
        return None, None
    if index is not None:
        raise SystemExit("If the first positional argument is an index, do not pass a second positional argument.")
    return None, selector


def _find_sv_by_index(index: str) -> Path:
    exact = sorted(SV_DIR.glob(f"{index}.*"))
    if exact:
        return exact[0]
    if index.isdigit():
        wanted = int(index)
        candidates = [p for p in SV_DIR.iterdir() if p.is_file() and p.stem.isdigit() and int(p.stem) == wanted]
        if candidates:
            return sorted(candidates)[0]
    raise FileNotFoundError(f"No single-view image found for index '{index}' in {SV_DIR}")


def resolve_demo_inputs(mode: Optional[str], index: Optional[str]) -> Tuple[List[Path], str, str]:
    if mode == "mv":
        mv_index = "1" if not index else str(index)
        mv_dir = MV_DIR / mv_index
        if not mv_dir.exists() and mv_index.isdigit():
            for cand in MV_DIR.iterdir():
                if cand.is_dir() and cand.name.isdigit() and int(cand.name) == int(mv_index):
                    mv_dir = cand
                    break
        if not mv_dir.exists():
            raise FileNotFoundError(f"No multiview directory found for index '{mv_index}' in {MV_DIR}")

        paths = [mv_dir / "front.png", mv_dir / "left.png", mv_dir / "back.png"]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing multiview images: {missing}")
        return paths, "mv", mv_dir.name

    if index:
        sv = _find_sv_by_index(str(index))
        return [sv], "sv", sv.stem

    if not PENGUIN_IMAGE.exists():
        raise FileNotFoundError(f"Default penguin image not found at {PENGUIN_IMAGE}")
    return [PENGUIN_IMAGE], "sv", "penguin"


def resolve_inputs(mode: Optional[str], index: Optional[str], manual_images: Optional[List[str]]) -> Tuple[List[Path], str, str, bool]:
    if manual_images:
        image_paths = [Path(p) for p in manual_images]
        missing = [str(p) for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing manual image paths: {missing}")
        inferred_mode = mode or ("mv" if len(image_paths) > 1 else "sv")
        return image_paths, inferred_mode, "manual", False

    image_paths, demo_mode, label = resolve_demo_inputs(mode, index)
    return image_paths, demo_mode, label, True


def maybe_remove_bg(image: Any, use_rembg: bool, remover: Optional[Any]) -> Any:
    if not use_rembg:
        return image.convert("RGBA")
    if image.mode == "RGB" and remover is not None:
        image = remover(image)
    return image.convert("RGBA")


def load_pil_images(paths: List[Path], use_rembg: bool) -> List[Any]:
    from PIL import Image

    remover = None
    if use_rembg:
        from hy3dgen.rembg import get_background_remover

        remover = get_background_remover()

    out: List[Any] = []
    for p in paths:
        out.append(maybe_remove_bg(Image.open(p), use_rembg, remover))
    return out


def next_custom_dir() -> Path:
    custom_root = OUTPUTS_DIR / "custom"
    custom_root.mkdir(parents=True, exist_ok=True)
    used = {int(p.name) for p in custom_root.iterdir() if p.is_dir() and p.name.isdigit()}
    n = 1
    while n in used:
        n += 1
    out = custom_root / str(n)
    out.mkdir(parents=True, exist_ok=False)
    return out


def default_demo_dir(mode: str, label: str) -> Path:
    out = OUTPUTS_DIR / mode / label
    out.mkdir(parents=True, exist_ok=True)
    return out


def choose_paint_model(args) -> Tuple[str, str]:
    if args.paint_model_repo or args.paint_subfolder:
        if not (args.paint_model_repo and args.paint_subfolder):
            raise SystemExit("If overriding paint model, pass both --paint-model-repo and --paint-subfolder")
        return args.paint_model_repo, args.paint_subfolder

    if args.paint_preset:
        return PAINT_PRESETS[args.paint_preset]

    return PAINT_PRESETS["2.0-turbo"]


# Single-slot cache: process-lifetime reuse of the loaded paint pipeline across
# calls with the same (model_repo, subfolder, diffusion_backend, mlx_weights_path).
# A fresh process (CLI usage) gets an empty cache and behaves exactly as before.
_PAINT_CACHE_KEY = None
_PAINT_CACHE_PIPELINE = None
_PAINT_CACHE_LOCK = threading.Lock()


def get_or_load_paint_pipeline(model_repo, subfolder, diffusion_backend, mlx_weights_path, pbr_albedo_only=False, progress_callback=None):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hf_progress import report_hf_downloads

    global _PAINT_CACHE_KEY, _PAINT_CACHE_PIPELINE

    # pbr_albedo_only must be part of the cache key: a full-PBR and an
    # albedo-only pipeline are structurally different loaded UNets, not just
    # a runtime flag on an otherwise-identical pipeline.
    key = (model_repo, subfolder, diffusion_backend, mlx_weights_path, pbr_albedo_only)
    with _PAINT_CACHE_LOCK:
        if _PAINT_CACHE_KEY != key:
            if _PAINT_CACHE_PIPELINE is not None:
                import gc
                import torch
                # Hunyuan3DPaintPipeline has no unified .to() to move it to CPU;
                # dropping the reference is what already happened implicitly
                # between calls before this cache existed, not a regression.
                _PAINT_CACHE_PIPELINE = None
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

            with report_hf_downloads(progress_callback, f"Downloading paint model {model_repo}/{subfolder} (first run only)"):
                _PAINT_CACHE_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(
                    model_repo,
                    subfolder=subfolder,
                    diffusion_backend=diffusion_backend,
                    mlx_weights_path=mlx_weights_path,
                    pbr_albedo_only=pbr_albedo_only,
                )
            _PAINT_CACHE_KEY = key

        return _PAINT_CACHE_PIPELINE


def run_paint_pipeline(mesh, image_paths: List[Path], args, progress_callback=None):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_repo, subfolder = choose_paint_model(args)
    images = load_pil_images(image_paths, use_rembg=not args.no_rembg)
    image_input = images if len(images) > 1 else images[0]

    painter = get_or_load_paint_pipeline(
        model_repo, subfolder, args.paint_diffusion_backend, args.paint_mlx_weights,
        pbr_albedo_only=getattr(args, 'paint_basic_texture', False),
        progress_callback=progress_callback,
    )

    t0 = time.time()
    textured = painter(mesh, image=image_input, seed=getattr(args, 'seed', 0), progress_callback=progress_callback)
    print(f"paint_model={model_repo}/{subfolder}")
    print(f"paint_backend={painter.config.device}/{painter.render.raster_mode}")
    print(f"paint_diffusion_backend={args.paint_diffusion_backend}")
    print(f"paint_time={time.time() - t0:.1f}s")
    return textured


def add_input_args(p):
    p.add_argument("selector", nargs="?", help="sv, mv, or an index. If omitted, defaults to penguin single-view input.")
    p.add_argument("index", nargs="?", help="Index when selector is sv or mv")
    p.add_argument("--image", action="append", help="Manual input image path. Repeat for multiple images.")
    p.add_argument("--no-rembg", action="store_true", help="Disable background removal")


def add_shape_model_args(p):
    p.add_argument("--shape-preset", choices=sorted(SHAPE_PRESETS), help="Shape model preset")
    p.add_argument("--shape-model-repo", help="Override shape model repo")
    p.add_argument("--shape-subfolder", help="Override shape model subfolder")
    p.add_argument("--shape-variant", default="fp16", help="Shape model variant (set empty string to disable)")
    p.add_argument("--no-shape-safetensors", action="store_true")
    p.add_argument("--shape-steps", type=int, default=30)
    p.add_argument("--shape-octree-resolution", type=int, default=256)
    p.add_argument("--shape-num-chunks", type=int, default=12000)
    p.add_argument("--shape-backend", choices=["pytorch", "swift"], default="pytorch",
                    help="swift uses the native MLX-Swift hy3d binary (2.0-turbo preset only, single-image only)")


def add_paint_model_args(p):
    p.add_argument("--paint-preset", choices=sorted(PAINT_PRESETS), help="Paint model preset")
    p.add_argument("--paint-model-repo", help="Override paint model repo")
    p.add_argument("--paint-subfolder", help="Override paint model subfolder")
    p.add_argument(
        "--paint-diffusion-backend",
        choices=["pytorch", "mlx"],
        default="pytorch",
        help="Diffusion UNet backend for paint: pytorch (default) or mlx (Apple Silicon)",
    )
    p.add_argument(
        "--paint-mlx-weights",
        help="Optional directory containing converted MLX weights (unet.npz / vae.npz)",
    )
    p.add_argument(
        "--paint-basic-texture",
        action="store_true",
        help="For the PBR (2.1) preset: skip metallic-roughness generation, roughly halving "
        "multiview diffusion time. No effect on other presets (they never generate mr).",
    )


def cmd_shape(args):
    run_shape_command(args)


def cmd_paint(args):
    mode_hint, index = parse_mode_and_index(args.selector, args.index)
    image_paths, mode, label, is_demo = resolve_inputs(mode_hint, index, args.image)

    import trimesh

    mesh = trimesh.load(args.mesh, force="mesh")
    textured = run_paint_pipeline(mesh, image_paths, args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = default_demo_dir(mode, label) if is_demo else next_custom_dir()
        out_path = out_dir / "paint.glb"

    textured.export(out_path)
    print(f"mode={mode}")
    print(f"inputs={[str(p) for p in image_paths]}")
    print(f"saved={out_path}")


def cmd_full(args):
    mode_hint, index = parse_mode_and_index(args.selector, args.index)
    image_paths, mode, label, is_demo = resolve_inputs(mode_hint, index, args.image)

    if mode == "mv" and len(image_paths) != 3:
        raise SystemExit("Full multiview run expects exactly 3 images (front/left/back)")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = default_demo_dir(mode, label) if is_demo else next_custom_dir()

    mesh = run_shape_pipeline(image_paths, mode, args)
    shape_path = out_dir / "shape.glb"
    mesh.export(shape_path)

    textured = run_paint_pipeline(mesh, image_paths, args)
    paint_path = out_dir / "paint.glb"
    textured.export(paint_path)

    print(f"mode={mode}")
    print(f"inputs={[str(p) for p in image_paths]}")
    print(f"shape_saved={shape_path}")
    print(f"paint_saved={paint_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for shape, paint, and full runs")
    p.add_argument("--seed", type=int, default=12345)

    sp = p.add_subparsers(dest="cmd", required=True)

    shape = sp.add_parser("shape", help="Generate shape only")
    add_input_args(shape)
    add_shape_model_args(shape)
    shape.add_argument("--output", help="Output mesh path")
    shape.set_defaults(func=cmd_shape)

    paint = sp.add_parser("paint", help="Texture an existing mesh")
    add_input_args(paint)
    add_paint_model_args(paint)
    paint.add_argument("--mesh", required=True, help="Input mesh path")
    paint.add_argument("--output", help="Output textured mesh path")
    paint.set_defaults(func=cmd_paint)

    full = sp.add_parser("full", help="Run shape then paint")
    add_input_args(full)
    add_shape_model_args(full)
    add_paint_model_args(full)
    full.add_argument("--output-dir", help="Output directory. If omitted, uses outputs/sv|mv/... or outputs/custom/<n>")
    full.set_defaults(func=cmd_full)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
