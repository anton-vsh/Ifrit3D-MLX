<img width="1920" height="879" alt="m3dium-head" src="https://github.com/user-attachments/assets/269785ab-6a6b-4785-8c89-89fa1516d2a5" />
Fast 3d generation for graphic designers. Includes image to 3d and text to 3d, built in postprocessing and png export — no need for additional 3d software. Apple native architecture.


# m3dium (Apple Silicon)

Started as a spiritual successor to Luma Genie (RIP) ended up as an ultimate poster asset machine. Not only you can generate those «ugly» (but incredibly cozy) 3D models once again using built-in Stable diffusion turbo (texture-detail passes included), with shape generation running on a native Swift/MLX backend by default — roughly 4x faster than the PyTorch pipeline. As a cherry on top, you can also generate robust lowpoly models and «normal» high poly as well.

<img width="1560" height="773" alt="Снимок экрана — 2026-08-10 в 21 50 48" src="https://github.com/user-attachments/assets/b4d42277-7b39-4c44-aca8-78a17f01a591" />

Benchmark: ~4 min on m1, ~2 min on m4
Maintained by [Anton Shlyonkin](https://www.shlyonk.in).




## What's new in m3dium

- **Zero manual model setup** — shape, paint, delight, SD Turbo, and the CLIP subject classifier all download automatically from Hugging Face on first use and are cached locally — no manual checkpoint placement, no config editing, whether you're running the packaged app or from source.
- **Lowpoly / Draft / Normal / High presets** — one-click combinations tuning geometry (reduction target, octree resolution) together with paint settings (resolution, steps, texture size, CFG), calibrated from measured face counts and A/B-tested settings rather than arbitrary numbers.
- **Built-in postprocessing filters** — Riso, Dither, Stipple, 3d Mosh, Halftone, Haring, Fresnel (heuristic), Checkerboard.
- **Built-in PNG export** — high resolution, no need for external 3d software just to get an image.
- **Standalone macOS app** — the same UI packaged as a double-clickable `.app`/`.dmg` with a menu bar helper (no Terminal window, no Dock icon). See [Releases](../../releases) for a prebuilt build, or `scripts/build_app.sh` to build your own.
- **Gradio UI** (`app.py`) — Image-to-3D and Text-to-3D tabs, covering shape generation, texturing, polygon reduction, and upscaling without touching a terminal.
- **Polygon reduction** — Inserts a remesh step inside the main pipeline, resulting in a cleaner mesh and correct lowpoly UV.
- **Text to 3D** — image generation as the starting step. Instrumental in getting that Luma Genie look.
- **Re-texture with seed** — re-run just the texturing pass on an existing mesh with a new (or fixed) seed, without regenerating the shape.
- **Swift/MLX shape backend** — shape generation defaults to a native Swift binary (~4x faster than PyTorch at the same settings), with an in-process cache keeping it loaded across generations; falls back to PyTorch automatically if not built locally.
- **Swift/MLX paint backend** — paint can also run end-to-end (UV unwrap through baking) on the same native Swift binary instead of the PyTorch/hybrid-MLX pipeline, avoiding per-step PyTorch↔MLX conversion overhead. Each generation currently runs as its own subprocess, so unlike the shape backend it reloads weights from disk every run rather than staying warm in memory.
- **Upscale texture pass** — an optional latent generative touch-up applied per-view before baking.
- **Granular progress reporting** — per-diffusion-step progress in the UI instead of a single stalled bar for the whole shape or texture pass.




## Setup / install

1. Use «Releases» section to download .dmg and install as a regular .app
2. Open the .dmg, drag m3dium.app into Applications.
3. First launch only: right-click (or Control-click) the app → Open → Open in the confirmation dialog. This build is ad-hoc signed, not notarized (no Apple Developer Program), so Gatekeeper shows one "unidentified developer" warning on first launch. After that one approval, double-click works normally from then on.
4. On first use, model weights download automatically (takes time) into ~/Library/Application Support/m3dium/ — no manual setup needed.

### Alternative: Pinokio

If you use [Pinokio](https://pinokio.co), paste this repo's URL (`https://github.com/anton-vsh/m3dium`) into its "Install" field instead. The launcher (`install.js`/`start.js`/`pinokio.js`) downloads the same signed `.dmg` from this repo's Releases and extracts the prebuilt interpreter + Metal extensions — no compilation, no Xcode Command Line Tools needed. Model weights still download on first use, same as above.

Once running, the app exposes a standard Gradio API at `<url>/?view=api` (see the URL in Pinokio's "Open Web UI" tab) with an auto-generated schema — usable from cURL, Python (`gradio_client`), or JavaScript (`@gradio/client`).


## Credits

This project builds upon the work of:

- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) — the original CLI this project forked from, and also the source of the vendored Swift/MLX shape and paint backends (their newer Swift `main` branch — see [`swift/README.md`](swift/README.md))
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
