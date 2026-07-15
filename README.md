# Ifrit3D-MLX (mac Os only)

Spiritual successor to Luma Genie (RIP). Now you can generate whose «ugly» (but incredibly cozy) 3D models once again (upscale option included). As a cherry on the cake, you can also generate robust lowpoly models and «normal» high poly aswell.

Huge thank you to the authors of the original models and MLX port.

Full reference implementation of Hunyuan3D inference on native Apple Silicon, including MLX texturing, plus a full Gradio UI and a standalone macOS app.

Maintained by [Anton Shlyonkin](https://www.shlyonk.in).

---

## What's new in Ifrit3D-MLX compared to originam Hunyuan3d-MLX

This started as a clone of [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) (CLI-only) and has grown into a full application on top of it:

- **Gradio UI** (`app.py`) — Image-to-3D and Text-to-3D tabs, covering shape generation, texturing, polygon reduction, and upscaling without touching a terminal.
- **Standalone macOS app** — the same UI packaged as a double-clickable `.app`/`.dmg` with a menu bar helper (no Terminal window, no Dock icon), model weights downloaded on first use into `~/Library/Application Support/`. See [Releases](../../releases) for a prebuilt build, or `scripts/build_app.sh` to build your own.
- **Pipeline caching** — shape and paint pipelines stay loaded across generations instead of reloading from disk every run.
- **Polygon reduction** — main feature. Inserts remesh step inside of the main pipeline resulting in cleaner mesh and correct lowpoly UV.
- **Text to 3D** — image generation as the starting step. Instrumental in getting that Luma Genie look.
- **FlashVDM volume decoder** — an optional faster decode path, with fragment filtering to clean up the mesh artifacts it can introduce.
- **Multi-view shape input** — reconstruct from front + left + back images instead of a single photo, for shape models that support it.
- **Re-texture with seed** — re-run just the texturing pass on an existing mesh with a new (or fixed) seed, without regenerating the shape.
- **Upscale** — regenerate a result at higher resolution/step counts, with optional SD-based input image refinement first.
- **Granular progress reporting** — per-diffusion-step progress in the UI instead of a single stalled bar for the whole shape or texture pass.

---

## Supported Models (same as original Hunyuan3D-MLX)

| Model | Type | MPS | MLX | MLX HF |
| - | - | - | - | - |
| hunyuan3d-dit-v2-mini | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-mini-turbo | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-0 | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-0-turbo | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-1 | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-mv | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-dit-v2-mv-turbo | 🧱 Shape | ✅ | 🏗️ | |
| hunyuan3d-paint-v2-0 | 🎨 Paint | ✅ | ✅ | [zimengxiong/Hunyuan3D-2.0-Paint-MLX](https://huggingface.co/zimengxiong/Hunyuan3D-2.0-Paint-MLX) |
| hunyuan3d-paint-v2-0-turbo | 🎨 Paint | ✅ | ✅ | [zimengxiong/Hunyuan3D-2.0-Paint-MLX](https://huggingface.co/zimengxiong/Hunyuan3D-2.0-Paint-MLX) |
| hunyuan3d-paintpbr-v2-1 | 🎨 Paint | ✅ | 🏗️ | [zimengxiong/Hunyuan3D-2.1-Paint-MLX](https://huggingface.co/zimengxiong/Hunyuan3D-2.1-Paint-MLX) |

---

## Setup / install

Use «Releases» section to download .dmg and install as a regular .app

## Credits
Built on top of [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX), which is itself a derivative work of [Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [Lane et. al](https://arxiv.org/abs/2011.03277), and pedronaugusto.

Model and derivative models respect the `TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT`, see [legal](legal/hunyuan/)

All other work is licensed under [MIT](LICENSE).
