# Ifrit3D-MLX (mac Os only)

Spiritual successor to Luma Genie (RIP). Now you can generate those «ugly» (but incredibly cozy) 3D models once again (texture-detail passes included), with shape generation running on a native Swift/MLX backend by default — roughly 4x faster than the PyTorch fallback. As a cherry on the cake, you can also generate robust lowpoly models and «normal» high poly aswell.

<img width="1551" height="892" alt="Снимок экрана — 2026-07-16 в 13 48 36" src="https://github.com/user-attachments/assets/e8829d46-c764-40a5-bbe5-abf3171621a9" />



Huge thank you to the authors of the original models and MLX port.

Full reference implementation of Hunyuan3D inference on native Apple Silicon, including MLX texturing, plus a full Gradio UI and a standalone macOS app.

Maintained by [Anton Shlyonkin](https://www.shlyonk.in).

<img width="1427" height="474" alt="Снимок экрана — 2026-07-18 в 12 23 50" src="https://github.com/user-attachments/assets/444e3bb8-8ac1-4121-826f-34f0dd7c7fa9" />


---

## Latest release: [v0.3.0](../../releases/tag/v0.3.0)

- **New native Swift/MLX paint backend** — the whole paint pipeline (UV unwrap, multiview render, diffusion, baking) now runs end-to-end in MLX for the RGB/"2.0" profile, avoiding the PyTorch↔MLX conversion overhead of the existing hybrid backend.
- **Fixed real classifier-free-guidance and a 3D-RoPE multiview attention bug** that caused texture corruption on complex subjects (surfaced by an A/B against a sibling Swift-native app sharing the same checkpoint weights).
- **New per-view "Upscale texture" pass** — an optional SD Turbo detail touch-up that round-trips through Swift's own renderer for the bake, avoiding cross-renderer depth/occlusion artifacts.
- **ESRGAN sharpening removed entirely** — measured to reduce detail versus the SD Turbo pass alone.
- **Redesigned presets** — Lowpoly / Draft / Normal / High now each tune their own paint resolution, steps, texture size, and CFG for a real quality ladder.
- New paint controls (resolution, steps, texture size) exposed in the UI for direct testing.

See the [full release notes](../../releases/tag/v0.3.0) for details.

---

## What's new in Ifrit3D-MLX

Based on [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) (CLI-only) and has grown into a full application on top of it:

- **Zero manual model setup** — shape, paint, delight, SD Turbo, and the CLIP subject classifier all download automatically from Hugging Face on first use and are cached locally — no manual checkpoint placement, no config editing, whether you're running the packaged app or from source.
- **Standalone macOS app** — the same UI packaged as a double-clickable `.app`/`.dmg` with a menu bar helper (no Terminal window, no Dock icon). See [Releases](../../releases) for a prebuilt build, or `scripts/build_app.sh` to build your own.
- **Gradio UI** (`app.py`) — Image-to-3D and Text-to-3D tabs, covering shape generation, texturing, polygon reduction, and upscaling without touching a terminal.
- **Polygon reduction** — Inserts a remesh step inside the main pipeline, resulting in a cleaner mesh and correct lowpoly UV.
- **Text to 3D** — image generation as the starting step. Instrumental in getting that Luma Genie look.
- **Re-texture with seed** — re-run just the texturing pass on an existing mesh with a new (or fixed) seed, without regenerating the shape.
- **Swift/MLX shape backend** — shape generation defaults to a native Swift binary (~4x faster than PyTorch at the same settings), with an in-process cache keeping it loaded across generations; falls back to PyTorch automatically if not built locally.
- **Swift/MLX paint backend** — paint can also run end-to-end (UV unwrap through baking) on the same native Swift binary instead of the PyTorch/hybrid-MLX pipeline, avoiding per-step PyTorch↔MLX conversion overhead. Each generation currently runs as its own subprocess, so unlike the shape backend it reloads weights from disk every run rather than staying warm in memory.
- **Upscale texture pass** — an optional latent generative touch-up applied per-view before baking.
- **Lowpoly / Draft / Normal / High presets** — one-click combinations tuning geometry (reduction target, octree resolution) together with paint settings (resolution, steps, texture size, CFG), calibrated from measured face counts and A/B-tested settings rather than arbitrary numbers.
- **Granular progress reporting** — per-diffusion-step progress in the UI instead of a single stalled bar for the whole shape or texture pass.

---

## Setup / install

1. Use «Releases» section to download .dmg and install as a regular .app
2. Open the .dmg, drag Ifrit3D-MLX.app into Applications.
3. First launch only: right-click (or Control-click) the app → Open → Open in the confirmation dialog. This build is ad-hoc signed, not notarized (no Apple Developer Program), so Gatekeeper shows one "unidentified developer" warning on first launch. After that one approval, double-click works normally from then on.
4. On first use, model weights download automatically (takes time) into ~/Library/Application Support/Ifrit3D-MLX/ — no manual setup needed.


## Credits

This project builds upon the work of:

- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) — the original CLI this project forked from, and also the source of the vendored Swift/MLX shape backend (their newer Swift `main` branch — see [`swift/README.md`](swift/README.md))
- [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [TRELLIS](https://github.com/microsoft/TRELLIS) (Lane et al., 2024)
- [pedronaugusto](https://github.com/pedronaugusto) — MLX implementation and related contributions
- [Stability AI SD Turbo](https://huggingface.co/stabilityai/sd-turbo)
- [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) / [VCGLib](https://github.com/cnr-isti-vclab/vcglib) — mesh simplification
- [xatlas](https://github.com/jpcy/xatlas) — UV atlas generation
- Garland & Heckbert (1997), *Surface Simplification Using Quadric Error Metrics* — https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf

## Third-Party Components

This project also includes or depends on the following third-party software:

| Component | License |
|----------|---------|
| Hunyuan3D-2 | Tencent Hunyuan 3D 2.0 Community License |
| SD Turbo | Stability AI Community License |
| diffusers | Apache 2.0 |
| transformers | Apache 2.0 |
| Gradio | Apache 2.0 |
| OpenCV | Apache 2.0 |
| PyTorch | BSD-3-Clause |
| MLX | MIT |
| trimesh | MIT |
| rembg | MIT |
| einops | MIT |
| OmegaConf | BSD-3-Clause |
| PyMeshLab | MIT |
| VCGLib | BSD-2-Clause |
| xatlas | MIT |
| mlx-swift | MIT |
| swift-numerics | Apache 2.0 |
| mtldiffrast | See `libraries/mtldiffrast/LICENSE.txt` |
| mtlbvh | See `libraries/mtlbvh/LICENSE.txt` |
| mtlmesh | See `libraries/mtlmesh/LICENSE` |
| mtlgemm | See `libraries/mtlgemm/LICENSE` |

## Licensing

Models based on Hunyuan3D are subject to the **TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT**. See the [legal/hunyuan](legal/hunyuan/) directory.

SD Turbo models are subject to the Stability AI Community License.

Unless otherwise noted, all original code and modifications in this repository are licensed under the [MIT License](LICENSE).
