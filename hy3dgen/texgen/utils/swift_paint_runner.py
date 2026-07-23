"""Runs the vendored native Swift/MLX `hy3d paint` binary as a subprocess for
the "2.0" (RGB, non-PBR) profile — an alternative to this repo's own hybrid
PyTorch+MLX pipeline (hy3dgen/texgen/mlx/hybrid_unet.py).

Why this exists: the hybrid pipeline keeps the whole diffusers PyTorch
pipeline (VAE, scheduler, tensor bookkeeping) running on MPS and only swaps
the UNet's forward pass to MLX, paying PyTorch<->numpy<->MLX conversion
overhead on every denoising step. The native Swift binary runs the entire
paint pipeline (UV unwrap, multiview render, diffusion, baking) in MLX with
zero cross-framework copying, end to end. See swift/README.md for the full
investigation (checkpoint parity, RoPE bug, speed/quality A/Bs).

Scope: RGB ("2.0"/paint-v2-0) only, matching swift/swift_runner.py's
shape-side scope note. PBR ("2.1") is not wired up here.

SD-Turbo detail pass: Swift has no SD-Turbo of its own (only ESRGAN via
--no-superres). To get a per-view generative detail pass without the
z-fighting/ghosting that comes from baking Swift-rendered views with
Python's renderer (see swift/README.md), the flow instead round-trips
through Swift twice: dump raw per-view PNGs (--dump-views), run SD-Turbo on
each view in Python (this repo's existing SDTurboUpscaler/SubjectClassifier
— the same code path the PyTorch pipeline already uses successfully), then
hand the processed views back to Swift's own renderer to bake
(--bake-views) — so the same renderer that determined visibility for the
diffusion conditioning also does the final bake.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
SWIFT_BIN = ROOT / "swift" / "bin" / "hy3d"

_PROGRESS_RE = re.compile(r"^\s*\[\s*(\d+)%\]\s*(.+)$")

_ENV = {**os.environ, "LLVM_PROFILE_FILE": "/dev/null"}


def swift_paint_available() -> bool:
    return SWIFT_BIN.exists()


def _run_swift(cmd, stage_label: str, progress_callback=None, progress_range=(0.0, 1.0)):
    """Runs an `hy3d` subprocess, forwarding its `[ NN%] stage` progress lines
    (rescaled into `progress_range`) through `progress_callback`."""
    lo, hi = progress_range
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(SWIFT_BIN.parent), env=_ENV,
    )
    last_line = ""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        last_line = line
        m = _PROGRESS_RE.match(line)
        if m and progress_callback is not None:
            pct, stage = m.groups()
            frac = lo + (int(pct) / 100.0) * (hi - lo)
            progress_callback(frac, f"{stage_label} — {stage}")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Swift hy3d {stage_label} failed (exit {ret}): {last_line}")
    return time.time() - t0


def run_swift_paint(
    mesh,
    image,
    weights_root: str,
    model: str = "rgb",
    seed: int = 0,
    res: int = 512,
    steps: int = 15,
    guidance: float = 2.0,
    tex: int = 2048,
    superres: bool = False,
    sd_detail: bool = False,
    sd_strength: float = 0.3,
    sd_res: int = 768,
    progress_callback=None,
):
    """Paints `mesh` (a trimesh Trimesh, untextured) using `image` (a single
    PIL reference image — the Swift binary handles UV unwrap, multiview
    render, diffusion and baking itself) via the native Swift binary.

    `weights_root` must be a directory containing `hunyuan3d-paint-v2-0/`
    in Tencent's own on-disk layout (config.json/unet/vae/...) — the
    already-downloaded HF snapshot dir works directly, no separate MLX
    conversion needed (the Swift binary loads the raw safetensors itself).

    `sd_detail=True` runs a per-view SD-Turbo pass (see module docstring)
    instead of Swift's own single-shot diffuse-then-bake — sd_strength/
    sd_res control that pass; `superres`/`tex` still apply to the final
    Swift bake either way (superres is a no-op on the dump-views leg,
    matching Swift's own dump-views mode, which never super-resolves).

    Returns a textured trimesh Trimesh (matching Hunyuan3DPaintPipeline's
    return type — same .export()/.visual.material.baseColorTexture
    interface used elsewhere in this repo).
    """
    import trimesh

    if not swift_paint_available():
        raise RuntimeError(
            f"Swift hy3d binary not found at {SWIFT_BIN}. See swift/README.md to build it."
        )

    # Gradio sliders hand back floats (e.g. 384.0), but the Swift CLI parses
    # integer flags with Swift's Int(String), which returns nil for any decimal
    # string ("384.0") and then silently falls back to its own default. Coerce
    # the int-typed knobs here so a UI value of 384.0 actually reaches Swift as
    # "384" rather than being dropped. (guidance/sd_strength are genuinely
    # float and parse fine via Swift's Float(String) / Python's float().)
    res = int(res)
    steps = int(steps)
    tex = int(tex)
    sd_res = int(sd_res)
    seed = int(seed)
    guidance = float(guidance)
    sd_strength = float(sd_strength)

    with tempfile.TemporaryDirectory(prefix="hy3d_swift_paint_") as tmp:
        tmp_path = Path(tmp)
        mesh_path = tmp_path / "mesh.glb"
        image_path = tmp_path / "reference.png"
        out_path = tmp_path / "painted.glb"

        mesh.export(str(mesh_path))
        image = image.convert("RGB")
        image.save(str(image_path))

        if not sd_detail:
            cmd = [
                str(SWIFT_BIN), "paint", str(mesh_path), str(image_path),
                "-o", str(out_path),
                "--weights", str(weights_root),
                "--model", model,
                "--res", str(res), "--steps", str(steps),
                "--guidance", str(guidance), "--tex", str(tex),
                "--seed", str(seed),
            ]
            if not superres:
                cmd.append("--no-superres")

            elapsed = _run_swift(cmd, "Swift paint", progress_callback)
            print("paint_backend_impl=swift")
            print(f"paint_time_swift={elapsed:.1f}s")
            return trimesh.load(str(out_path), force="mesh")

        # --- sd_detail: dump raw views, SD-Turbo each in Python, bake in Swift ---
        dump_dir = tmp_path / "dump"
        sd_dir = tmp_path / "sd"

        t_dump = _run_swift(
            [
                str(SWIFT_BIN), "paint", str(mesh_path), str(image_path),
                "-o", str(tmp_path / "unused.glb"),
                "--weights", str(weights_root), "--model", model,
                "--res", str(res), "--steps", str(steps), "--guidance", str(guidance),
                "--seed", str(seed), "--dump-views", str(dump_dir),
            ],
            "Swift dump-views", progress_callback, progress_range=(0.0, 0.4),
        )

        if progress_callback is not None:
            progress_callback(0.4, "SD-Turbo detail pass — loading models")
        from PIL import Image
        from .subject_classifier import SubjectClassifier
        from .sdturbo_upscale_utils import SDTurboUpscaler

        classifier = SubjectClassifier(device="cpu")
        subject = classifier(image)

        class _SDConfig:
            device = "mps"
            texture_size = sd_res

        os.environ["HY3D_SUPER_RES_STRENGTH"] = str(sd_strength)
        upscaler = SDTurboUpscaler(_SDConfig())

        sd_dir.mkdir(parents=True, exist_ok=True)
        t_sd0 = time.time()
        for i in range(6):
            view = Image.open(dump_dir / f"view_{i}.png").convert("RGB")
            processed = upscaler(view, subject=subject)
            processed.save(sd_dir / f"view_{i}.png")
            if progress_callback is not None:
                progress_callback(0.4 + 0.4 * (i + 1) / 6, f"SD-Turbo detail pass — view {i + 1}/6")
        t_sd = time.time() - t_sd0

        t_bake = _run_swift(
            [
                str(SWIFT_BIN), "paint", str(mesh_path), str(image_path),
                "-o", str(out_path),
                "--weights", str(weights_root), "--model", model,
                "--tex", str(tex),
                "--bake-views", str(sd_dir), "--view-size", str(sd_res),
            ],
            "Swift bake-views", progress_callback, progress_range=(0.8, 1.0),
        )

        print("paint_backend_impl=swift+sdturbo")
        print(f"paint_time_swift_dump={t_dump:.1f}s paint_time_sdturbo={t_sd:.1f}s paint_time_swift_bake={t_bake:.1f}s")
        return trimesh.load(str(out_path), force="mesh")
