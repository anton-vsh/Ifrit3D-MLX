from __future__ import annotations

import os
from pathlib import Path

from hf_progress import report_hf_downloads

SD_TURBO_REPO = "stabilityai/sd-turbo"

# Mirror the convention in shape/swift_runner.py: SD turbo is a large
# checkpoint that cannot be shipped inside the signed .app bundle (build_app.sh
# never copies models/ — the bundle would be multi-GB and, once signed,
# read-only anyway). Resolve it relative to HY3DGEN_MODELS so it lands in the
# launcher's writable app-support location on packaged installs, and next to
# the repo's own models/ dir in dev. Same directory layout as the bundled
# "models/sd-turbo" the code already references.
def sd_turbo_path() -> Path:
    hy3dgen_models = os.environ.get(
        "HY3DGEN_MODELS",
        str(Path(__file__).resolve().parent / "models" / "hy3dgen"),
    )
    return Path(hy3dgen_models).parent / "sd-turbo"


def ensure_sd_turbo(progress_callback=None) -> Path:
    """Downloads the SD Turbo diffusers checkpoint (fp16 variant only, ~2.4GB)
    into the writable location from sd_turbo_path() on first use. Both the
    Text-to-3D text2img pipeline and the SD-Turbo texture upscale pass load it
    with local_files_only=True, so a missing local dir is an instant error —
    not a network fallback — hence the explicit download here.

    allow_patterns keeps this to the fp16 files diffusers needs with
    variant="fp16" (model_index.json + component configs + tokenizer + the
    three .fp16.safetensors), skipping the ~4GB fp32 single-file weights."""
    target = sd_turbo_path()
    if (target / "model_index.json").exists() and (
        target / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    ).exists():
        return target

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    with report_hf_downloads(
        progress_callback, "Downloading SD Turbo (~2.4GB, first run only)"
    ):
        snapshot_download(
            repo_id=SD_TURBO_REPO,
            local_dir=str(target),
            allow_patterns=[
                "*.json",
                "tokenizer/*",
                "*.fp16.safetensors",
                "README.md",
                "LICENSE.md",
            ],
        )
    return target
