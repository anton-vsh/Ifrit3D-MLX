from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SWIFT_BIN = ROOT / "swift" / "bin" / "hy3d"

# Inside the packaged .app, this file lives under Contents/Resources/app/,
# which is read-only once code-signed — writing weights there fails and
# would silently discard the signature even if it didn't. HY3DGEN_MODELS is
# the app's existing writable-directory convention (set by the launcher to
# ~/Library/Application Support/Ifrit3D-MLX/models, sibling to hf_home/logs;
# app.py falls back to ROOT/models/hy3dgen in dev). Mirror that fallback so
# swift's weights land in the *same* writable location either way, one
# level up (sibling of "hy3dgen"/"models", matching the launcher's own
# per-subsystem directory layout).
_hy3dgen_models = os.environ.get("HY3DGEN_MODELS", str(ROOT / "models" / "hy3dgen"))
SWIFT_SHAPE_WEIGHTS = Path(_hy3dgen_models).parent / "hy3d-swift" / "shape-large"
SWIFT_SHAPE_REPO = "zimengxiong/hunyuan3d-mlx-shape-large"

# This checkpoint is consistency-distilled for exactly 8 steps ("8-step
# turbo" per the upstream README's own benchmark) — it is NOT a tunable
# quality/speed knob like a normal diffusion model. Verified: running it at
# 30 steps instead of 8 produced 566 mesh fragments (vs. 27 at 8 steps) on
# the same input, i.e. visibly broken output, not merely lower quality.
SWIFT_SHAPE_STEPS = 8

_PROGRESS_RE = re.compile(r"^\s*\[\s*(\d+)%\]\s*(.+)$")
_DONE_RE = re.compile(r"^shape:\s*(\d+)\s*verts,\s*(\d+)\s*faces in ([\d.]+)s")


def swift_shape_available() -> bool:
    return SWIFT_BIN.exists()


def ensure_swift_shape_weights(progress_callback=None):
    if (SWIFT_SHAPE_WEIGHTS / "model.fp16.safetensors").exists():
        return
    from huggingface_hub import snapshot_download
    from hf_progress import report_hf_downloads

    SWIFT_SHAPE_WEIGHTS.mkdir(parents=True, exist_ok=True)
    with report_hf_downloads(progress_callback, "Downloading Swift shape weights (~4.9GB, first run only)"):
        snapshot_download(repo_id=SWIFT_SHAPE_REPO, local_dir=str(SWIFT_SHAPE_WEIGHTS))


def run_swift_shape(
    image_path: Path,
    out_glb_path: Path,
    octree_resolution: int = 256,
    seed: int = 0,
    progress_callback=None,
):
    """Runs the vendored Swift `hy3d shape` binary as a subprocess and loads
    the resulting mesh. Only supports single-image, non-PBR shape generation
    (the checkpoint zimengxiong ported to MLX-Swift covers 2.0-turbo only).

    Step count is intentionally not a parameter here — see SWIFT_SHAPE_STEPS."""
    import trimesh

    if not swift_shape_available():
        raise RuntimeError(
            f"Swift hy3d binary not found at {SWIFT_BIN}. See swift/README.md to build it."
        )
    ensure_swift_shape_weights(progress_callback)

    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SWIFT_BIN), "shape", str(image_path),
        "-o", str(out_glb_path),
        "--weights", str(SWIFT_SHAPE_WEIGHTS),
        "--octree", str(octree_resolution),
        "--steps", str(SWIFT_SHAPE_STEPS),
        "--seed", str(seed),
    ]

    t0 = time.time()
    # The vendored binary carries LLVM profiling instrumentation and writes a
    # default.profraw into its cwd on every run; send that to /dev/null.
    env = {**os.environ, "LLVM_PROFILE_FILE": "/dev/null"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(SWIFT_BIN.parent), env=env,
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
            progress_callback(int(pct) / 100.0, f"Swift shape — {stage}")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Swift hy3d shape failed (exit {ret}): {last_line}")

    print(f"shape_backend=swift")
    print(f"shape_time={time.time() - t0:.1f}s")
    print(f"shape_swift_summary={last_line}")

    mesh = trimesh.load(str(out_glb_path), force="mesh")
    return mesh
