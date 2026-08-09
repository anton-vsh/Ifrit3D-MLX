# AGENTS.md — context for coding agents working on m3dium

This file is for whichever coding agent picks up work on this repo next. It's not
user-facing documentation (see `README.md` for that) — it's a dump of architecture,
conventions, and hard-won lessons from recent work, so you don't have to
re-derive them or re-make mistakes that already got made and fixed once.

## What this project is

m3dium (formerly "Ifrit3D-MLX", renamed — see git history) is a native Apple Silicon
port of Hunyuan3D: image/text → 3D mesh generation with MLX/Metal-accelerated shape
and paint backends, a Gradio UI (`app.py`), and a set of stylized post-process
filters. Distributed two ways: a signed `.dmg` (GitHub Releases) and a Pinokio
launcher (`install.js` etc. at repo root) that downloads the same `.dmg` and extracts
prebuilt binaries — see "Distribution" below.

## Architecture map

- **`app.py`** — the Gradio UI and orchestration. Two tabs (Image-to-3D, Text-to-3D),
  each with its own `generate()`/`_run_retexture()`-style entry points that call into
  `shape/runner.py` (shape) and `main.py` (paint), then dispatch the selected filter
  (see "Filters" below), then export `.glb`/`.obj`.
- **`main.py`** — paint pipeline orchestration (`run_paint_pipeline`), plus a
  process-lifetime pipeline cache (`get_or_load_paint_pipeline`) keyed on
  (model_repo, subfolder, diffusion_backend, ...) so repeated generations in one
  session don't reload the paint model every time.
- **`shape/runner.py`** — shape pipeline orchestration (`run_shape_pipeline`). Picks
  Swift-native (`shape/swift_runner.py`, calls the vendored `swift/bin/hy3d` binary,
  ~4x faster) vs PyTorch fallback (`hy3dgen/shapegen/`) depending on availability.
  Has its own `load_pil_images`/rembg glue, **duplicated** from `main.py`'s (known,
  not (yet) worth deduplicating — see "Known duplication" below).
- **`hy3dgen/`** — the core library (forked/adapted from Tencent's Hunyuan3D +
  ZimengXiong's MLX port): `shapegen/` (shape diffusion), `texgen/` (paint diffusion,
  UV baking, differentiable renderer, and all the post-process filters under
  `texgen/utils/`).
- **Native Metal extensions** — `mtldiffrast`, `mtlbvh`, `cumesh` (in
  `libraries/mtlmesh`), `flex-gemm` (in `libraries/mtlgemm`). Custom Metal-compiled
  Python C-extensions, no PyPI wheels. **Shaders are compiled to `.metallib` at build
  time** (each library's own `setup.py`), not JIT-compiled at runtime — so there is no
  compile step on an end user's machine, by design (see "Distribution" for why this
  matters).
- **`swift/bin/hy3d`** — vendored prebuilt Swift/MLX binary (accelerated shape *and*
  paint backend). Gitignored; must exist locally to build a release (see
  `swift/README.md`); the release `.dmg` bundles it.
- **`scripts/build_app.sh` / `sign_adhoc.sh` / `make_dmg.sh`** — the release pipeline.
  `build_app.sh` assembles `dist/m3dium.app` by copying the uv-managed standalone
  Python interpreter + resolved `site-packages` (with the 4 Metal extensions already
  built) + app source + the Swift binary — **not** a from-source install. Ad-hoc
  signed (no Apple Developer account), not notarized.

## Filters — read this before touching any of them

All filters live in `hy3dgen/texgen/utils/*_filter.py` and follow one shared
convention: `apply_X_filter(mesh: trimesh.Trimesh, **creative_kwargs) -> trimesh.Trimesh`,
`DEFAULT_*` module-level constants for every creative kwarg (used both as the
kwarg default *and* the UI's "Reset to Defaults" value). Wired into `app.py` via:
- `_build_filter_panels()` — builds one `gr.Column(visible=False)` settings group per
  filter, returns `{"groups": (...), "<name>_params": [...], ...}`.
- One `Filter` dropdown per tab; `_FILTER_NAMES` tuple order **must exactly match**
  `_build_filter_panels()`'s returned `"groups"` tuple order — they're zipped
  positionally, nothing checks this at runtime if you get it wrong.
- Two dispatch call sites (`generate()` and `_run_retexture()`, each duplicated
  again for the Text-to-3D tab's wrapper) — an `if/elif` chain on `filter_style`
  calling the matching `apply_X_filter`, right after paint and before export.

### The 8 filters

| Filter | Space | What it does |
|---|---|---|
| Dither | texture | 1-bit Bayer dither, driven by real AO + baked creases + Canny edges on the painted albedo |
| Stipple | texture | Same signal, dart-thrown ink dots instead of ordered dither |
| Riso | texture | Same signal, two-color risograph print simulation |
| Haring | texture | Black/white maze pattern grown from painted luminance |
| **3D Mosh** | **geometry** | Datamosh-style vertex glitch — block copy/displace directly on the vertex position buffer. **Ported literally from a reference three.js prototype; do not "improve" the algorithm without direct evidence it's wrong** — see "Lessons learned" below, this one has a whole saga. |
| Halftone | texture | Classic CMYK color halftone, angled per-channel dot screens (C/M/Y/K at 15°/75°/0°/45° to avoid moiré), color comes from the painted albedo itself (not AO-driven like the others) |
| Checkerboard | texture | UV-space checker, independent horizontal/vertical cell counts, AO-shaded (darkens in occlusion) |
| **Fresnel** | **material only** | No texture bake at all — sets a flat PBRMaterial (color/metallic/roughness). The "fresnel" look comes for free from the Model3D viewer's own real-time PBR/IBL rendering as the camera orbits. **Do not try to bake a glow/rim-light texture for this one** — that would only be correct from one fixed camera angle, defeating the point (this was explicitly discussed and rejected). |

Two filter-classification sets in `app.py` matter for correctness, not just cleanliness:
- `_TEXTURE_FILTERS` — filters that actually read the painted albedo's color/detail;
  used to raise the `Paint Texture Size` floor (a low-res atlas starves their signal).
  3D Mosh/Checkerboard/Fresnel are excluded (they don't read texture content).
- `_MATERIAL_REPLACING_FILTERS` — filters that fully replace `mesh.visual.material`;
  used to skip wasted PBR-map generation (`postfactum_pbr.py`) when it would just get
  thrown away. All 7 texture/material filters are in this set; 3D Mosh isn't (it
  never touches material).

### Filter dropdown panel-switch race (already fixed, don't reintroduce)

`app.py`'s `_filter_style_show` / `_filter_style_hide_others` are **two sequential
`.then()`-chained steps**, not one atomic function, and not two independent
`.change()` listeners. Both of those were tried and both broke:
- Two independent `.change()` listeners (even sharing a `concurrency_id`) can settle
  on different dropdown values under rapid switching — the "settings panel doesn't
  appear" bug.
- One atomic function that shows the new panel *and* hides the old one in the same
  update batch drops the reveal of any panel being shown for the first time ever in
  that session, if it's paired with hiding an already-visible sibling in the same
  batch (verified directly, looks like a Gradio/Svelte mount-timing bug, not
  something fixable from the Python side). Splitting into "show new" then (separate
  tick) "hide others" avoids the collision.

## Distribution

**Two channels, same artifact.** The `.dmg` published on GitHub Releases is the
source of truth; nothing gets compiled on an end user's machine either way.

1. **Direct `.dmg`**: `build_app.sh` → `sign_adhoc.sh` → `make_dmg.sh` →
   `gh release create vX.Y.Z dist/m3dium.dmg --repo anton-vsh/m3dium`. Bump
   `pyproject.toml`'s `version` first, tag matches (`git tag vX.Y.Z`).
2. **Pinokio** (`install.js`/`start.js`/`pinokio.js`/`pinokio.json`/`reset.js`/
   `update.js`/`icon.png` at repo root): downloads
   `releases/latest/download/m3dium.dmg`, mounts it, `cp -RL`s out the prebuilt
   Python interpreter + `site-packages` + app source into its own `app/` folder, runs
   the extracted interpreter directly (**no venv/conda** — the copied `site-packages`
   is ABI-tied to the copied interpreter, nothing else would work). Verified
   end-to-end via `pterm` against a real Pinokio instance.
   - Repo has GitHub topics `pinokio`, `3d`, `3dgen`, `image-to-3d`, `apple-silicon`,
     `mlx` — the `pinokio` topic is what makes it show up in Pinokio's Community
     Scripts / Discover feed automatically (no approval needed, but indexing has a
     real lag — new topics took longer than a few minutes to show up under a
     tag filter in testing). Getting onto the curated/**Verified** Discover
     placement is a separate, manual process: contact the Pinokio admin
     (@cocktailpeanut on X) for publisher verification, then the repo gets
     transferred into the Pinokio Factory GitHub org. Not done — just documented in
     case it's wanted later.

## Known gotchas (all cost real debugging time — don't re-learn these)

- **AO raycasting**: `hy3dgen/texgen/utils/mesh_ao.py`'s `raycast_ao_raw` uses
  `mtlbvh` (Metal-accelerated BVH, already an unconditional project dependency), NOT
  trimesh's `mesh.ray`. This project has no Embree/pyembree binding — `mesh.ray`
  silently falls back to trimesh's pure-Python `RayMeshIntersector`, which does not
  scale: confirmed directly that it never returns (hours, near-zero CPU) on a real
  generated mesh with ~25k unique vertices. This bug affected *every* AO-driven
  filter (Dither/Stipple/Riso/Haring too), not just new ones — if AO-related code
  starts hanging again, check this hasn't regressed back to `mesh.ray`.
- **Pinokio's `fs.copy` API hangs** on the standalone Python distribution's internal
  symlinks (confirmed: 4+ hours, zero files copied). Use `shell.run` + `cp -RL`
  instead (same reason `build_app.sh` itself needs `-L` — see its own comment).
- **`hdiutil attach -mountpoint <path>`** fails with "Access denied" when the target
  path is inside a folder on an external volume (i.e. `PINOKIO_HOME` on an external
  drive). Let `hdiutil` pick its own default `/Volumes/<name>` location and capture
  the actual mount path via the shell `on:`-event regex mechanism (same pattern
  `start.js` uses to capture the server URL) rather than assuming/hardcoding it.
- **Model download progress**: only genuinely-network-hitting `huggingface_hub`
  calls get progress via `hf_progress.report_hf_downloads` (a context manager that
  monkey-patches `huggingface_hub.utils.tqdm` globally for its duration — scope it
  around the *outer* call, not necessarily the exact `from_pretrained` line, so
  anything triggered underneath is covered too). Before adding a new one, check
  whether the target actually hits network (`local_files_only=True` or an
  existence-gated local ckpt path means it doesn't) and whether it's already covered
  by an existing outer wrap before adding a redundant one. `rembg`'s background
  remover downloads via its own `pooch`-based downloader, entirely outside
  `huggingface_hub` — no real byte-level progress is available for it, only a
  before-the-call announcement.
- **Don't guess-fix reference algorithm ports.** The 3D Mosh filter went through
  several wrong "fixes" (BFS-based patch selection, unwelding the mesh) based on
  plausible-sounding theories that turned out wrong once actually tested against
  real output — the original literal port was correct all along. If a ported
  algorithm "looks wrong," get a concrete repro/comparison before rewriting it, not
  just a plausible mechanism.

## Known duplication (not yet cleaned up, intentional to leave alone for now)

`load_pil_images`/`maybe_remove_bg` exist near-identically in both `main.py` and
`shape/runner.py`. Both got the same `progress_callback` fix applied in parallel when
this was found — if you fix one, check the other.

## Where things stand as of this writing

- All 8 filters implemented, tested, shipped.
- Pinokio launcher implemented, tested end-to-end via `pterm`, shipped, listed in
  Community Scripts (topic-based).
- Latest release: v0.5.10 (first-run model download progress fixes — several
  `from_pretrained`/`snapshot_download` call sites had zero progress feedback,
  looked like a hang; audited all of them, fixed the ones that were real gaps).
- No open known bugs at time of writing. If continuing Pinokio work: Verified/Discover
  listing is the only unstarted item (see "Distribution" above).
