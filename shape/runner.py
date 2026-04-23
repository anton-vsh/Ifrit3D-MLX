from __future__ import annotations

import argparse
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
        from hy3dgen.rembg import BackgroundRemover

        remover = BackgroundRemover()

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


def run_shape_pipeline(image_paths: List[Path], mode: str, args, forced_preset: Optional[str] = None):
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    device = pick_device()
    model_repo, subfolder = choose_shape_model(mode, args, forced_preset=forced_preset)

    if mode == "mv" and subfolder == "hunyuan3d-dit-v2-1":
        raise SystemExit("2.1 does not provide an official multiview shape checkpoint in this repo. Use mv/mv-turbo instead.")

    images = load_pil_images(image_paths, use_rembg=not args.no_rembg)
    image_input = images[0] if len(images) == 1 else {"front": images[0], "left": images[1], "back": images[2]}

    load_kwargs = {
        "model_path": model_repo,
        "subfolder": subfolder,
        "device": device,
        "use_safetensors": not args.no_shape_safetensors,
    }
    if args.shape_variant:
        load_kwargs["variant"] = args.shape_variant

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(**load_kwargs)

    gen = torch.manual_seed(args.seed)
    t0 = time.time()
    mesh = pipeline(
        image=image_input,
        num_inference_steps=args.shape_steps,
        octree_resolution=args.shape_octree_resolution,
        num_chunks=args.shape_num_chunks,
        generator=gen,
        output_type="trimesh",
    )[0]
    print(f"shape_device={device}")
    print(f"shape_model={model_repo}/{subfolder}")
    print(f"shape_time={time.time() - t0:.1f}s")
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
