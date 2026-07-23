## Vendored Swift `hy3d` binary

`swift/bin/` (gitignored — not committed) holds a built copy of the `hy3d`
CLI from [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX)
(the current, Swift-based `main` branch of the repo this project originally
forked from). It's used for:

- **Shape generation** (`shape/swift_runner.py`), `2.0-turbo` preset only.
- **Paint generation** (`hy3dgen/texgen/utils/swift_paint_runner.py` /
  `hy3dgen/texgen/pipelines.py`'s `SwiftPaintPipeline`), RGB/"2.0"
  (non-turbo, non-PBR) profile only — select it with
  `--paint-diffusion-backend swift --paint-preset 2.0`.

Both run the whole respective stage (UV unwrap, multiview render,
diffusion, baking for paint) natively in MLX, with no PyTorch involved at
all — unlike this repo's own hybrid pipeline
(`hy3dgen/texgen/mlx/hybrid_unet.py`), which keeps the full diffusers
PyTorch pipeline running and only swaps the UNet's forward pass to MLX,
paying PyTorch↔numpy↔MLX conversion overhead on every denoising step.

**A/B tested** (same reference photo, same seed): the hybrid "2.0"+UniPC
path took ~93-101s paint time. The native Swift path, run through
[ZimengXiong/Modelr](https://github.com/ZimengXiong/Modelr) (a sibling
macOS app by the same author, confirmed to load byte-identical checkpoint
weights via sha256) completed the equivalent generation in ~36.5s with
clean, uncorrupted output across all views — the hybrid path was also
found to have a real bug in its multiview 3D-RoPE attention (unrelated to
CFG) that native Swift's implementation doesn't share.

### Memory footprint — corrected

An earlier note here claimed "Swift's own paint path peaks around
38-39GB" as the reason paint stayed Python-only. That figure was never
re-verified against the RGB/"2.0" profile specifically — it may have been
about the heavier PBR/"2.1" profile (not wired up here), or from a stale
build. Directly measured on this repo's own 24GB machine: the RGB/"2.0"
profile ran to completion with no memory pressure via Modelr's own app.
Don't assume the 38-39GB figure applies to what's wired up here without
re-measuring — but treat it as unverified rather than as a current
blocker.

### Why it's not committed

It's a compiled arm64 binary + a Metal shader resource bundle (~90MB
together), and building a working one requires more than plain
`swift build`/`swift run` — see below. That's a heavier requirement than
the rest of this project's build, so it's kept as a local, gitignored
artifact rather than forcing every clone to build Swift from source.

### Rebuilding it

`swift build`/`swift run` alone cannot compile the Metal shaders
mlx-swift needs (confirmed against mlx-swift's own README: "SwiftPM
(command line) cannot build the Metal shaders"). Two working paths:

**Path A — `xcodebuild` (produces a working metallib, but watch for code
coverage):**

```bash
git clone --branch main https://github.com/ZimengXiong/Hunyuan3D-MLX.git /tmp/hy3d-swift-src
cd /tmp/hy3d-swift-src
xcodebuild -scheme hy3d -configuration Release -destination 'platform=macOS' build

DERIVED=$(xcodebuild -scheme hy3d -showBuildSettings -configuration Release 2>/dev/null | awk -F'= ' '/ BUILT_PRODUCTS_DIR /{print $2; exit}')
mkdir -p /Users/user/Hunyuan3D-MLX/swift/bin
cp "$DERIVED/hy3d" /Users/user/Hunyuan3D-MLX/swift/bin/
cp -R "$DERIVED/mlx-swift_Cmlx.bundle" /Users/user/Hunyuan3D-MLX/swift/bin/
```

**Gotcha found this session:** building a bare `Package.swift` via
`xcodebuild -scheme <name>` (no persisted `.xcscheme`, no real
`.xcodeproj`) silently enables LLVM code-coverage instrumentation
(`-fprofile-instr-generate -fcoverage-mapping`) on every target, including
the compute-heavy ones (`Cmlx`, `hy3d`, `CXatlas`) — even for a plain
`build` action, not just `test`. This is real per-branch counter-increment
overhead, not just extra binary size — harmless for GPU-dispatched MLX
compute, but a large, real slowdown for CPU-bound C++ code (e.g. the
`xatlas` UV-unwrap library). Verified: `nm <binary> | grep -c __llvm_prof`
is nonzero on such a build. Neither `-enableCodeCoverage NO` (test-only
flag, rejected for `build`), nor `CODE_COVERAGE_ENABLED=NO`/
`CLANG_ENABLE_CODE_COVERAGE=NO` build-setting overrides, nor a
hand-written persisted `.xcscheme` fixed this in testing — the instrumented
flags persisted regardless. If a build has this problem, use Path B
instead of continuing to fight the Xcode scheme.

**Path B — plain `swift build` + borrow a working metallib (verified
clean, no instrumentation):**

```bash
git clone --branch main https://github.com/ZimengXiong/Hunyuan3D-MLX.git /tmp/hy3d-swift-src
cd /tmp/hy3d-swift-src
swift build -c release   # compiles fine, but the resulting binary can't run yet —
                          # it has no metallib (Metal shaders aren't compiled by plain swift build)

mkdir -p /Users/user/Hunyuan3D-MLX/swift/bin
cp .build/release/hy3d /Users/user/Hunyuan3D-MLX/swift/bin/

# Borrow the metallib from any working mlx-swift-based app/build (e.g. a
# Modelr.app release download, or a Path A build) — resource lookup is
# relative to the executable, so it must sit right next to the binary:
cp -R "<source>/mlx-swift_Cmlx.bundle" /Users/user/Hunyuan3D-MLX/swift/bin/
```

Verify either path produced a clean binary:

```bash
nm /Users/user/Hunyuan3D-MLX/swift/bin/hy3d | grep -c __llvm_prof   # want: 0
```

### Weights

**Shape**: `shape/swift_runner.py` auto-downloads the MLX-format
shape-large checkpoint (`zimengxiong/hunyuan3d-mlx-shape-large` on HF Hub,
~4.9GB) into `models/hy3d-swift/shape-large/` on first use if not already
present — same lazy-download convention as everything else under
`models/`.

**Paint**: no separate download or conversion needed — the Swift binary
loads the raw Tencent PyTorch safetensors checkpoint directly (`mx.load`
reads `.safetensors` natively), so `hy3dgen/texgen/utils/swift_paint_runner.py`
just points `--weights` at the same HF snapshot directory this repo
already downloads for the `pytorch`/`mlx` paint backends
(`tencent/Hunyuan3D-2` snapshot root, containing `hunyuan3d-paint-v2-0/`).

### Coverage

- Shape: only the `2.0-turbo` checkpoint (no `2.1`, no multiview).
  `shape/runner.py`'s `run_shape_pipeline` enforces this —
  `shape_backend="swift"` raises if the preset isn't `2.0-turbo` or the
  mode isn't single-image.
- Paint: only the RGB/"2.0" (non-turbo, non-PBR) checkpoint.
  `SwiftPaintPipeline.from_pretrained` raises if the subfolder looks like
  turbo or PBR/2.1. PBR is not wired up in this repo.

### Third-party Swift dependencies

Per upstream's `Package.swift`/`Package.resolved`, the compiled `hy3d`
binary links two third-party Swift packages, both permissively licensed
(also credited in the top-level README's Third-Party Components table):

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.31.4) — MIT
- [swift-numerics](https://github.com/apple/swift-numerics) (1.1.1) — Apache 2.0

The `hy3d` binary itself and the rest of upstream's Swift code is MIT
licensed under the same `ZimengXiong/Hunyuan3D-MLX` copyright already
covered by this repo's `LICENSE` file.
