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

# Classic offset-print color halftone (the Roy Lichtenstein / Ben-Day-dot look):
# CMYK-separates the mesh's own painted albedo, then renders each ink as a regular grid
# of dots whose *radius* (not placement -- unlike stipple_filter.py's dart-throwing)
# encodes local ink coverage, each channel's grid rotated to a different classic screen
# angle (C=15, M=75, Y=0, K=45) so the four grids don't visually beat against each
# other (moire) once overprinted. Composited with a real multiply ink blend, same as
# riso_filter.py, so channel overlaps produce genuine mixed colors instead of one ink
# flatly replacing another.
#
# Unlike the black/white filters in this package, color comes straight from the
# painted albedo (that's the point of a *color* halftone) -- but the same AO + baked
# crease shading signal the others use is blended into the K (black) channel only, so
# shadowed/creased areas still get denser black dots on top of their own CMY color
# instead of reading as flat as a paint-by-numbers comic panel.

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .mesh_ao import raycast_ao_raw
from .mesh_crease import bake_crease_map

_WORK_RES = 1024
_CANVAS_SIZE = 1536
_BASE_CELL_PX = 16.0
_AO_SIGMA = 5.0
_GAMMA = 0.6

_ANGLE_C = 15.0
_ANGLE_M = 75.0
_ANGLE_Y = 0.0
_ANGLE_K = 45.0

# Approximate process-ink primaries (not pure neon CMY) -- reads as print, not neon pop art.
_COLOR_C = np.array([0, 174, 239], dtype=np.float32)
_COLOR_M = np.array([236, 0, 140], dtype=np.float32)
_COLOR_Y = np.array([255, 242, 0], dtype=np.float32)
_COLOR_K = np.array([35, 31, 32], dtype=np.float32)
_PAPER = np.array([255, 255, 255], dtype=np.float32)

# User-facing defaults -- the four creative knobs exposed in app.py's Halftone panel.
DEFAULT_AO_STRENGTH = 3.0
DEFAULT_CREASE_SENSITIVITY = 15.0
DEFAULT_DOT_SIZE = 0.4
DEFAULT_COLOR_BOOST = 1.2


def _angled_dot_screen(coverage: np.ndarray, angle_deg: float, cell_px: float) -> np.ndarray:
    """Renders `coverage` (0-1 ink amount per pixel) as a regular grid of dots whose
    radius encodes the local average coverage, rotated to `angle_deg`. Padded to 1.5x
    before rotating (and cropped back after) so the grid has no gaps at the canvas edges
    once rotated back to axis-aligned."""
    size = coverage.shape[0]
    diag = int(size * 1.5)
    off = (diag - size) // 2

    padded = np.zeros((diag, diag), dtype=np.float32)
    padded[off:off + size, off:off + size] = coverage
    img = Image.fromarray((np.clip(padded, 0, 1) * 255).astype(np.uint8), mode="L")
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=0)
    arr = np.asarray(rotated, dtype=np.float32) / 255.0

    n_cells = max(1, int(diag / cell_px))
    dot_canvas = Image.new("L", (diag, diag), 0)
    draw = ImageDraw.Draw(dot_canvas)
    for iy in range(n_cells):
        y0, y1 = int(iy * cell_px), int(min(diag, (iy + 1) * cell_px))
        if y1 <= y0:
            continue
        for ix in range(n_cells):
            x0, x1 = int(ix * cell_px), int(min(diag, (ix + 1) * cell_px))
            if x1 <= x0:
                continue
            amount = float(arr[y0:y1, x0:x1].mean())
            r = (cell_px * 0.5) * np.sqrt(amount)
            if r < 0.4:
                continue
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    rotated_back = dot_canvas.rotate(-angle_deg, resample=Image.BICUBIC, fillcolor=0)
    cropped = rotated_back.crop((off, off, off + size, off + size))
    return np.asarray(cropped, dtype=np.float32) / 255.0


def _composite_cmyk(c: np.ndarray, m: np.ndarray, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    canvas = np.tile(_PAPER, (*c.shape, 1))
    for coverage, color in ((c, _COLOR_C), (m, _COLOR_M), (y, _COLOR_Y), (k, _COLOR_K)):
        blend = 1.0 - coverage[..., None] * (1.0 - color[None, None, :] / 255.0)
        canvas = canvas * blend
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_halftone_filter(mesh: trimesh.Trimesh,
                           ao_strength: float = DEFAULT_AO_STRENGTH,
                           crease_sensitivity: float = DEFAULT_CREASE_SENSITIVITY,
                           dot_size: float = DEFAULT_DOT_SIZE,
                           color_boost: float = DEFAULT_COLOR_BOOST) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo with a stylized CMYK color halftone print (see
    module docstring). `mesh.visual.material` must already be a trimesh material with
    `baseColorTexture` set (i.e. already painted). Returns `mesh` for convenience; also
    mutates it. Purely visual/stylized replacement, so the material is reset to
    flat/matte (no metallic/roughness/AO/normal maps).

    The four keyword args are the user-facing "creative" knobs (see app.py's Halftone
    panel); everything else in this module is a fixed implementation detail."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    small_albedo = albedo_tex.resize((_WORK_RES, _WORK_RES), Image.LANCZOS)
    rgb = np.asarray(small_albedo, dtype=np.float32) / 255.0
    rgb = np.clip(0.5 + (rgb - 0.5) * color_boost, 0, 1)

    k = 1.0 - rgb.max(axis=-1)
    denom = np.clip(1.0 - k, 1e-4, None)
    c = np.clip((1 - rgb[..., 0] - k) / denom, 0, 1)
    m = np.clip((1 - rgb[..., 1] - k) / denom, 0, 1)
    y = np.clip((1 - rgb[..., 2] - k) / denom, 0, 1)

    ao_raw = raycast_ao_raw(mesh, render)
    ao_blurred = gaussian_filter(ao_raw, sigma=_AO_SIGMA)
    shading_darkness = np.clip((1 - np.clip(ao_blurred, 0, 1) ** _GAMMA) * ao_strength, 0, 1)
    crease = bake_crease_map(mesh, render, angle_deg=crease_sensitivity)
    extra_k_hires = np.maximum(shading_darkness, crease)
    extra_k = np.array(Image.fromarray((extra_k_hires * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0
    k = np.clip(k + extra_k, 0, 1)

    def _upsample(chan):
        return np.array(Image.fromarray((chan * 255).astype(np.uint8)).resize(
            (_CANVAS_SIZE, _CANVAS_SIZE), Image.LANCZOS)) / 255.0

    cell_px = _BASE_CELL_PX * dot_size
    dots_c = _angled_dot_screen(_upsample(c), _ANGLE_C, cell_px)
    dots_m = _angled_dot_screen(_upsample(m), _ANGLE_M, cell_px)
    dots_y = _angled_dot_screen(_upsample(y), _ANGLE_Y, cell_px)
    dots_k = _angled_dot_screen(_upsample(k), _ANGLE_K, cell_px)

    canvas = _composite_cmyk(dots_c, dots_m, dots_y, dots_k)
    halftone_tex = Image.fromarray(cv2.resize(canvas, (texture_size, texture_size), interpolation=cv2.INTER_LANCZOS4))

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=halftone_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
