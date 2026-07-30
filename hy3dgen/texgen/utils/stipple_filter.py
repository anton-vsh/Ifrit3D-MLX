# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# Ink stipple filter: same AO + crease + thresholded-albedo-detail "darkness" signal as
# dither_filter.py (see that module's docstring for what each component contributes and
# why), rendered as weighted dart-thrown dots instead of Atkinson error-diffusion. Reads
# softer and more "hand-drawn ink" than the Atkinson dither; darker regions get denser,
# slightly larger dots (tighter minimum spacing), bright/open regions stay sparse.
#
# This is a simple grid-accelerated dart-throwing (rejection sampling) stippler, not full
# weighted Voronoi relaxation -- dot placement is a bit more randomly clustered than the
# "proper" academic method, but reads fine at this scale and is far cheaper to compute.
#
# Dots are drawn on a supersampled canvas then downsampled with Lanczos for clean
# anti-aliased circles -- deliberately NOT the dither filter's blocky NEAREST upscale,
# since stippling's dots are meant to look like ink, not print-grid facets.

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .mesh_ao import raycast_ao_raw
from .mesh_crease import bake_crease_map

# Shared darkness-signal tuning -- kept identical to dither_filter.py's locked recipe,
# since this is the same underlying AO/crease/albedo-detail signal, just rendered
# differently.
_BLUR_SIGMA = 6.0
_GAMMA = 0.6
_AO_MULTIPLIER = 4.0
_CREASE_ANGLE_DEG = 15.0
_CREASE_STRENGTH = 1.0
_ALBEDO_DETAIL_STRENGTH = 0.75
_ALBEDO_HP_SIGMA = 2.5
_ALBEDO_THRESHOLD_PERCENTILE = 75.0
# 1024, not 512: same blockiness bug as dither_filter.py (see its _DITHER_RES comment) --
# a coarser darkness-map resolution makes the albedo-detail layer's genuine fine paint
# texture look chunky once resampled up to the dart-throwing canvas.
_WORK_RES = 1024

# Stipple-specific: locked "max density" recipe (see conversation/commit history) --
# tested against three very different meshes and approved as the production density,
# even though it's dense enough to nearly solid-fill very simple/smooth geometry.
_CANVAS_SIZE = 1536
_SUPERSAMPLE = 2
_R_MIN = 0.5
_R_MAX = 1.6
_SPACING_MIN = 0.9
_SPACING_MAX = 3.5
_MAX_ATTEMPTS = 1_500_000


def _build_darkness_map(mesh: trimesh.Trimesh, render, albedo: Image.Image) -> np.ndarray:
    """AO + crease + thresholded-albedo-detail, combined into one darkness map at
    _WORK_RES. See dither_filter.py's module docstring for what each signal is for."""
    ao_raw = raycast_ao_raw(mesh, render)
    ao_raw = gaussian_filter(ao_raw, sigma=_BLUR_SIGMA)
    ao_contrast = np.clip(ao_raw, 0, 1) ** _GAMMA
    darkness_ao = np.clip((1 - ao_contrast) * _AO_MULTIPLIER, 0, 1)

    crease = bake_crease_map(mesh, render, angle_deg=_CREASE_ANGLE_DEG)
    darkness_crease = crease * _CREASE_STRENGTH

    combined_hires = np.maximum(darkness_ao, darkness_crease)
    combined_small = np.array(Image.fromarray((combined_hires * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0

    small_albedo = albedo.convert("L").resize((_WORK_RES, _WORK_RES), Image.LANCZOS)
    gray = np.asarray(small_albedo, dtype=np.float32) / 255.0
    hp = np.abs(gray - gaussian_filter(gray, sigma=_ALBEDO_HP_SIGMA))
    cutoff = np.percentile(hp, _ALBEDO_THRESHOLD_PERCENTILE)
    detail_mask = (hp > cutoff).astype(np.float32)

    return np.clip(combined_small + _ALBEDO_DETAIL_STRENGTH * detail_mask, 0, 1)


def _stipple_render(darkness: np.ndarray, seed: int = 0) -> Image.Image:
    """Weighted dart-throwing stippling: darker regions get denser, slightly larger dots
    (closer minimum spacing); brighter regions stay sparse/empty."""
    rng = np.random.default_rng(seed)
    hi = _CANVAS_SIZE * _SUPERSAMPLE
    h, w = darkness.shape

    cell = _SPACING_MIN
    grid = {}

    def grid_key(x, y):
        return (int(x // cell), int(y // cell))

    def too_close(x, y, min_d):
        gx, gy = int(x // cell), int(y // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py) in grid.get((gx + dx, gy + dy), []):
                    if (px - x) ** 2 + (py - y) ** 2 < min_d ** 2:
                        return True
        return False

    points = []
    radii = []
    for _ in range(_MAX_ATTEMPTS):
        x = rng.uniform(0, hi)
        y = rng.uniform(0, hi)
        sx = min(w - 1, int(x / hi * w))
        sy = min(h - 1, int(y / hi * h))
        d = float(darkness[sy, sx])
        if d < 0.03:
            continue
        if rng.random() > d:
            continue
        min_d = (_SPACING_MAX - (_SPACING_MAX - _SPACING_MIN) * d) * _SUPERSAMPLE
        if too_close(x, y, min_d):
            continue
        points.append((x, y))
        radii.append((_R_MIN + (_R_MAX - _R_MIN) * d) * _SUPERSAMPLE)
        grid.setdefault(grid_key(x, y), []).append((x, y))

    canvas = Image.new("L", (hi, hi), 255)
    draw = ImageDraw.Draw(canvas)
    for (x, y), r in zip(points, radii):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    return canvas.resize((_CANVAS_SIZE, _CANVAS_SIZE), Image.LANCZOS)


def apply_stipple_filter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo entirely with a stylized ink-stipple texture,
    driven by real AO + baked creases + thresholded albedo detail (see module
    docstring). `mesh.visual.material` must already be a trimesh material with
    `baseColorTexture` set (i.e. already painted). Returns `mesh` for convenience; also
    mutates it. Purely visual/stylized replacement, so the material is reset to
    flat/matte (no metallic/roughness/AO/normal maps -- those describe a real surface's
    light response, which doesn't apply to a flat black/white graphic)."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    darkness = _build_darkness_map(mesh, render, albedo_tex)
    stipple = _stipple_render(darkness)
    stipple_tex = stipple.resize((texture_size, texture_size), Image.LANCZOS).convert("RGB")

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=stipple_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
