## Vendored Swift `hy3d` binary

`swift/bin/` (gitignored — not committed) holds a built copy of the `hy3d`
CLI from [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX)
(the current, Swift-based `main` branch of the repo this project originally
forked from), used to accelerate shape generation for the `2.0-turbo` preset
via `shape/swift_runner.py`. Only shape generation is used from it — paint
stays on this repo's own Python/MLX pipeline (see the plan notes on memory
footprint: Swift's own paint path peaks around 38-39GB, vs. well under 10GB
for the pipeline already in this repo).

### Why it's not committed

It's a compiled arm64 binary + a Metal shader resource bundle (~90MB
together), and building it requires Xcode (not just command-line tools —
`swift build`/`swift run` cannot compile the Metal shaders mlx-swift needs;
only `xcodebuild` can). That's a heavier requirement than the rest of this
project's build, so it's kept as a local, gitignored artifact rather than
forcing every clone to have Xcode installed.

### Rebuilding it

```bash
git clone --branch main https://github.com/ZimengXiong/Hunyuan3D-MLX.git /tmp/hy3d-swift-src
cd /tmp/hy3d-swift-src
xcodebuild -scheme hy3d -configuration Release -destination 'platform=macOS' build

# Copy the binary + its Metal resource bundle together (resource lookup is
# relative to the executable, so they must stay side by side):
DERIVED=$(xcodebuild -scheme hy3d -showBuildSettings -configuration Release 2>/dev/null | awk -F'= ' '/ BUILT_PRODUCTS_DIR /{print $2; exit}')
mkdir -p /Users/user/Hunyuan3D-MLX/swift/bin
cp "$DERIVED/hy3d" /Users/user/Hunyuan3D-MLX/swift/bin/
cp -R "$DERIVED/mlx-swift_Cmlx.bundle" /Users/user/Hunyuan3D-MLX/swift/bin/
```

### Weights

`shape/swift_runner.py` auto-downloads the MLX-format shape-large checkpoint
(`zimengxiong/hunyuan3d-mlx-shape-large` on HF Hub, ~4.9GB) into
`models/hy3d-swift/shape-large/` on first use if not already present —
same lazy-download convention as everything else under `models/`.

### Coverage

Swift only ships an MLX port of the `2.0-turbo` shape checkpoint (no `2.1`,
no multiview). `shape/runner.py`'s `run_shape_pipeline` enforces this —
`shape_backend="swift"` raises if the preset isn't `2.0-turbo` or the mode
isn't single-image.

### Third-party Swift dependencies

Per upstream's `Package.swift`/`Package.resolved`, the compiled `hy3d`
binary links two third-party Swift packages, both permissively licensed
(also credited in the top-level README's Third-Party Components table):

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.31.4) — MIT
- [swift-numerics](https://github.com/apple/swift-numerics) (1.1.1) — Apache 2.0

The `hy3d` binary itself and the rest of upstream's Swift code is MIT
licensed under the same `ZimengXiong/Hunyuan3D-MLX` copyright already
covered by this repo's `LICENSE` file.
