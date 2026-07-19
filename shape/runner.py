from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
SV_DIR = IMAGES_DIR / "sv"
MV_DIR = IMAGES_DIR / "mv"
PENGUIN_IMAGE = IMAGES_DIR / "penguin.png"
OUTPUTS_DIR = ROOT / "outputs"


SHAPE_PRESETS = {
    "mini": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"),
    "mini-turbo": ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo"),
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1"),
    "mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv"),
    "mv-turbo": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-turbo"),
}


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def choose_shape_model(mode: str, args, forced_preset: Optional[str] = None) -> Tuple[str, str]:
    if args.shape_model_repo or args.shape_subfolder:
        if not (args.shape_model_repo and args.shape_subfolder):
            raise SystemExit("If overriding shape model, pass both --shape-model-repo and --shape-subfolder")
        return args.shape_model_repo, args.shape_subfolder

    if forced_preset:
        return SHAPE_PRESETS[forced_preset]

    if getattr(args, "shape_preset", None):
        return SHAPE_PRESETS[args.shape_preset]

    default_preset = "mv-turbo" if mode == "mv" else "mini-turbo"
    return SHAPE_PRESETS[default_preset]


# Single-slot cache: process-lifetime reuse of the loaded shape pipeline across
# calls with the same (model_repo, subfolder, variant, use_safetensors, device).
# A fresh process (CLI usage) gets an empty cache and behaves exactly as before.
_SHAPE_CACHE_KEY = None
_SHAPE_CACHE_PIPELINE = None
_SHAPE_CACHE_LOCK = threading.Lock()


def get_or_load_shape_pipeline(model_repo, subfolder, variant, use_safetensors, device, progress_callback=None):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hf_progress import report_hf_downloads

    global _SHAPE_CACHE_KEY, _SHAPE_CACHE_PIPELINE

    key = (model_repo, subfolder, variant, use_safetensors, device)
    with _SHAPE_CACHE_LOCK:
        if _SHAPE_CACHE_KEY != key:
            if _SHAPE_CACHE_PIPELINE is not None:
                import gc
                import torch
                _SHAPE_CACHE_PIPELINE.to("cpu")
                _SHAPE_CACHE_PIPELINE = None
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

            load_kwargs = {
                "model_path": model_repo,
                "subfolder": subfolder,
                "device": device,
                "use_safetensors": use_safetensors,
            }
            if variant:
                load_kwargs["variant"] = variant

            with report_hf_downloads(progress_callback, f"Downloading shape model {model_repo}/{subfolder} (first run only)"):
                _SHAPE_CACHE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(**load_kwargs)
            _SHAPE_CACHE_KEY = key

        pipeline = _SHAPE_CACHE_PIPELINE

    # enable_flashvdm() reloads a VAE checkpoint in both directions (enable and
    # disable), so only call it when the desired state actually differs from
    # what's already applied — otherwise every cache hit would pay that cost.
    want_flashvdm = os.environ.get('HY3D_USE_FLASHVDM', '0') == '1'
    if getattr(pipeline, '_flashvdm_enabled', False) != want_flashvdm:
        pipeline.enable_flashvdm(enabled=want_flashvdm)
        pipeline._flashvdm_enabled = want_flashvdm

    return pipeline


def _filter_mesh_fragments(mesh, min_faces=None):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if min_faces is None:
        min_faces = int(os.environ.get('HY3D_FRAGMENT_MIN_FACES', '20'))

    total_faces = len(mesh.faces)
    if total_faces == 0:
        return mesh

    # scipy csgraph instead of trimesh.split(): split() takes minutes on
    # exactly the dust-fragmented meshes this filter exists to clean
    # (observed: 20K+ components on an ambiguous white-on-white input).
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return mesh
    graph = coo_matrix(
        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
        shape=(total_faces, total_faces),
    )
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp <= 1:
        return mesh

    counts = np.bincount(labels)
    threshold = max(min_faces, int(total_faces * 0.001))
    keep_comp = counts >= threshold
    if not keep_comp.any():
        return mesh

    face_mask = keep_comp[labels]
    print(f"[shape] fragment filter: kept {int(keep_comp.sum())}/{ncomp} components "
          f"({int(face_mask.sum())}/{total_faces} faces, threshold={threshold})")
    return mesh.submesh([np.nonzero(face_mask)[0]], append=True)


def run_shape_pipeline(image_paths: List[Path], mode: str, args, forced_preset: Optional[str] = None, progress_callback=None):
    if getattr(args, "shape_backend", "pytorch") == "swift":
        effective_preset = forced_preset or getattr(args, "shape_preset", None)
        if mode != "sv":
            raise SystemExit("Swift shape backend does not support multi-view input.")
        if effective_preset != "2.0-turbo":
            raise SystemExit("Swift shape backend only supports the 2.0-turbo preset.")

        import tempfile
        from shape.swift_runner import run_swift_shape, SWIFT_SHAPE_STEPS

        if args.shape_steps != SWIFT_SHAPE_STEPS:
            print(
                f"[shape] swift backend ignores --shape-steps={args.shape_steps}: "
                f"this checkpoint is consistency-distilled for exactly {SWIFT_SHAPE_STEPS} steps "
                f"(more steps degrades output — verified: 566 mesh fragments at 30 steps vs. 27 at 8)."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # The binary does its own conditioning-image prep but no background
            # removal — feeding it a raw photo makes the model reconstruct the
            # background as geometry (planes behind/under the subject). Run the
            # same rembg preprocessing the PyTorch path gets and hand the Swift
            # binary the RGBA cutout instead.
            input_path = image_paths[0]
            if not args.no_rembg:
                cutout = load_pil_images([input_path], use_rembg=True)[0]
                input_path = Path(tmpdir) / "input_rgba.png"
                cutout.save(input_path)

            out_glb = Path(tmpdir) / "swift_shape.glb"
            mesh = run_swift_shape(
                input_path,
                out_glb_path=out_glb,
                octree_resolution=args.shape_octree_resolution,
                seed=args.seed,
                progress_callback=progress_callback,
            )
            return _filter_mesh_fragments(mesh)

    import torch

    device = pick_device()
    model_repo, subfolder = choose_shape_model(mode, args, forced_preset=forced_preset)

    if mode == "mv" and subfolder == "hunyuan3d-dit-v2-1":
        raise SystemExit("2.1 does not provide an official multiview shape checkpoint in this repo. Use mv/mv-turbo instead.")

    images = load_pil_images(image_paths, use_rembg=not args.no_rembg)
    image_input = images[0] if len(images) == 1 else {"front": images[0], "left": images[1], "back": images[2]}

    pipeline = get_or_load_shape_pipeline(
        model_repo, subfolder, args.shape_variant, not args.no_shape_safetensors, device,
        progress_callback=progress_callback,
    )

    gen = torch.manual_seed(args.seed)
    t0 = time.time()
    print(f"[shape] pipeline call: steps={args.shape_steps}, octree={args.shape_octree_resolution}, chunks={args.shape_num_chunks}")

    step_kwargs = {}
    if progress_callback is not None:
        total_steps = args.shape_steps

        def _on_step(step_idx, t, outputs):
            progress_callback((step_idx + 1) / total_steps, f"Shape diffusion — step {step_idx + 1}/{total_steps}")

        step_kwargs["callback"] = _on_step
        step_kwargs["callback_steps"] = 1

    mesh = pipeline(
        image=image_input,
        num_inference_steps=args.shape_steps,
        octree_resolution=args.shape_octree_resolution,
        num_chunks=args.shape_num_chunks,
        generator=gen,
        output_type="trimesh",
        **step_kwargs,
    )[0]
    print(f"shape_device={device}")
    print(f"shape_model={model_repo}/{subfolder}")
    print(f"shape_time={time.time() - t0:.1f}s")

    # Not just a FlashVDM problem: any decoder produces iso-surface noise
    # dust when the SDF is ambiguous (e.g. a white object on a white
    # background) — verified 20K+ micro-fragments from the vanilla decoder,
    # which then bake into speckle artifacts in the texture atlas.
    mesh = _filter_mesh_fragments(mesh)

    return mesh


def run_shape_command(args, forced_preset: Optional[str] = None):
    mode_hint, index = parse_mode_and_index(args.selector, args.index)
    image_paths, mode, label, is_demo = resolve_inputs(mode_hint, index, args.image)

    if mode == "mv" and len(image_paths) != 3:
        raise SystemExit("Multiview shape expects exactly 3 images (front/left/back)")

    mesh = run_shape_pipeline(image_paths, mode, args, forced_preset=forced_preset)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = default_demo_dir(mode, label) if is_demo else next_custom_dir()
        out_path = out_dir / "shape.glb"

    mesh.export(out_path)
    print(f"mode={mode}")
    print(f"inputs={[str(p) for p in image_paths]}")
    print(f"saved={out_path}")
    return out_path


def build_shape_parser(description: str = "Run shape generation") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("selector", nargs="?", help="sv, mv, or an index. If omitted, defaults to penguin single-view input.")
    p.add_argument("index", nargs="?", help="Index when selector is sv or mv")
    p.add_argument("--image", action="append", help="Manual input image path. Repeat for multiple images.")
    p.add_argument("--no-rembg", action="store_true", help="Disable background removal")

    p.add_argument("--shape-model-repo", help="Override shape model repo")
    p.add_argument("--shape-subfolder", help="Override shape model subfolder")
    p.add_argument("--shape-variant", default="fp16", help="Shape model variant (set empty string to disable)")
    p.add_argument("--no-shape-safetensors", action="store_true")
    p.add_argument("--shape-steps", type=int, default=30)
    p.add_argument("--shape-octree-resolution", type=int, default=256)
    p.add_argument("--shape-num-chunks", type=int, default=12000)
    p.add_argument("--shape-backend", choices=["pytorch", "swift"], default="pytorch",
                    help="swift uses the native MLX-Swift hy3d binary (2.0-turbo preset only, single-image only)")

    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", help="Output mesh path")
    return p


def run_shape_preset_cli(preset: str) -> None:
    parser = build_shape_parser(description=f"Run shape preset: {preset}")
    args = parser.parse_args()
    run_shape_command(args, forced_preset=preset)


def run_shape_cli() -> None:
    parser = build_shape_parser(description="Run shape generation")
    parser.add_argument("--shape-preset", choices=sorted(SHAPE_PRESETS), help="Shape model preset")
    args = parser.parse_args()
    run_shape_command(args)


if __name__ == "__main__":
    run_shape_cli()
