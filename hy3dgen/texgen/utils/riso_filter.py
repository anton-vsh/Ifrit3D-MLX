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

# Risograph-style two-color print filter: splits the same AO/crease/edge signal used
# by dither_filter.py into two separate spot-color "ink" layers instead of one
# black/white one -- a broad AO-driven "shadow shape" layer (form: deep occlusion,
# long soft gradients) and a fine crease+Canny-edge "linework" layer (detail: eyes,
# embroidery, carved texture, hard mesh creases), each independently Atkinson-dithered
# and printed in its own spot color on cream paper, composited with a real multiply
# ink blend (so overlaps between the two colors produce a third, darker mixed tone,
# like real overlaid transparent riso ink) and a small pixel offset between layers to
# mimic authentic riso misregistration.

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as nd_shift

from .mesh_ao import raycast_ao_raw
from .mesh_crease import bake_crease_map
from .dither_filter import _atkinson_dither

_AO_BROAD_SIGMA = 6.0
_AO_FINE_SIGMA = 1.0
_AO_BROAD_MULTIPLIER = 4.0
_AO_FINE_MULTIPLIER = 3.0
_GAMMA = 0.6
_WHITE_CAP = 0.9
_CREASE_ANGLE_DEG = 15.0
_CREASE_STRENGTH = 1.0
_CANNY_BLUR = 1.0
_CANNY_LOW = 40
_CANNY_HIGH = 120
_WORK_RES = 1024
_OFFSET_SHADOW = (0, 0)
_OFFSET_DETAIL = (3, -2)

# Locked default palette (see conversation/commit history): a classic riso blue for
# the broad shadow-shape layer, a fluorescent pink for the fine linework layer, on
# warm cream paper.
_PAPER = np.array([246, 242, 232], dtype=np.float32)
_COLOR_SHADOW = np.array([0, 120, 191], dtype=np.float32)
_COLOR_DETAIL = np.array([255, 72, 176], dtype=np.float32)

# User-facing defaults -- the locked recipe, exposed as "Reset" values for the UI's
# creative knobs (see app.py). Everything else here (offsets, canny blur, work_res) is
# a fixed implementation detail.
DEFAULT_AO_STRENGTH = _AO_BROAD_MULTIPLIER
DEFAULT_AO_GRADIENT_LENGTH = _AO_BROAD_SIGMA
DEFAULT_FINE_DETAIL = _AO_FINE_MULTIPLIER
DEFAULT_CREASE_SENSITIVITY = _CREASE_ANGLE_DEG
DEFAULT_EDGE_SENSITIVITY = float(_CANNY_LOW)


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _build_layers(mesh: trimesh.Trimesh, render, albedo: Image.Image,
                   ao_strength: float, ao_gradient_length: float, fine_detail: float,
                   crease_sensitivity: float, edge_sensitivity: float):
    """Same underlying AO/crease/edge signal as dither_filter.py, but split into two
    darkness maps instead of merged into one -- shadow (AO form) and detail
    (crease + Canny albedo edges), each becomes its own ink layer."""
    ao_raw = raycast_ao_raw(mesh, render)
    broad = gaussian_filter(ao_raw, sigma=ao_gradient_length)
    fine = gaussian_filter(ao_raw, sigma=_AO_FINE_SIGMA)
    darkness_broad = np.clip((1 - np.clip(broad, 0, 1) ** _GAMMA) * ao_strength, 0, 1)
    darkness_fine = np.clip((1 - np.clip(fine, 0, 1) ** _GAMMA) * fine_detail, 0, 1)
    shadow_darkness = np.maximum(darkness_broad, darkness_fine)
    shadow_small = np.array(Image.fromarray((shadow_darkness * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0

    crease = bake_crease_map(mesh, render, angle_deg=crease_sensitivity)
    crease_small = np.array(Image.fromarray((crease * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0 * _CREASE_STRENGTH

    small_albedo = albedo.convert("L").resize((_WORK_RES, _WORK_RES), Image.LANCZOS)
    gray_u8 = np.asarray(small_albedo, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), _CANNY_BLUR)
    edges = cv2.Canny(blurred, edge_sensitivity, edge_sensitivity * 3)
    edge_mask = (edges > 0).astype(np.float32)

    detail_darkness = np.clip(np.maximum(crease_small, edge_mask), 0, 1)
    return shadow_small, detail_darkness


def _riso_composite(shadow_darkness: np.ndarray, detail_darkness: np.ndarray,
                     color_shadow: np.ndarray, color_detail: np.ndarray,
                     paper: np.ndarray) -> np.ndarray:
    brightness_shadow = np.clip(1 - shadow_darkness, 0, _WHITE_CAP)
    brightness_detail = np.clip(1 - detail_darkness, 0, _WHITE_CAP)

    ink_shadow = 1.0 - _atkinson_dither(brightness_shadow)
    ink_detail = 1.0 - _atkinson_dither(brightness_detail)

    if _OFFSET_SHADOW != (0, 0):
        ink_shadow = nd_shift(ink_shadow, _OFFSET_SHADOW, order=0, mode="constant", cval=0)
    if _OFFSET_DETAIL != (0, 0):
        ink_detail = nd_shift(ink_detail, _OFFSET_DETAIL, order=0, mode="constant", cval=0)

    canvas = np.tile(paper, (*ink_shadow.shape, 1))
    # Multiply blend, like real overlaid transparent riso ink layers -- overlaps
    # produce a third, darker mixed color instead of one ink just replacing the other.
    for ink, color in [(ink_shadow, color_shadow), (ink_detail, color_detail)]:
        blend = 1.0 - ink[..., None] * (1.0 - color[None, None, :] / 255.0)
        canvas = canvas * blend
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_riso_filter(mesh: trimesh.Trimesh,
                       ao_strength: float = DEFAULT_AO_STRENGTH,
                       ao_gradient_length: float = DEFAULT_AO_GRADIENT_LENGTH,
                       fine_detail: float = DEFAULT_FINE_DETAIL,
                       crease_sensitivity: float = DEFAULT_CREASE_SENSITIVITY,
                       edge_sensitivity: float = DEFAULT_EDGE_SENSITIVITY,
                       color_shadow: str = None, color_detail: str = None,
                       paper_color: str = None) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo entirely with a stylized two-color risograph
    print texture (see module docstring). `mesh.visual.material` must already be a
    trimesh material with `baseColorTexture` set (i.e. already painted). Returns `mesh`
    for convenience; also mutates it. Purely visual/stylized replacement, so the
    material is reset to flat/matte (no metallic/roughness/AO/normal maps).

    The keyword args are the user-facing "creative" knobs (see app.py's Riso panel);
    `color_shadow`/`color_detail`/`paper_color` accept "#rrggbb" hex strings and fall
    back to the locked default palette when None. Everything else in this module is a
    fixed implementation detail."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    shadow_darkness, detail_darkness = _build_layers(
        mesh, render, albedo_tex, ao_strength, ao_gradient_length, fine_detail,
        crease_sensitivity, edge_sensitivity)

    rgb_shadow = _hex_to_rgb(color_shadow) if color_shadow else _COLOR_SHADOW
    rgb_detail = _hex_to_rgb(color_detail) if color_detail else _COLOR_DETAIL
    rgb_paper = _hex_to_rgb(paper_color) if paper_color else _PAPER

    canvas = _riso_composite(shadow_darkness, detail_darkness, rgb_shadow, rgb_detail, rgb_paper)
    riso_tex = Image.fromarray(cv2.resize(canvas, (texture_size, texture_size),
                                           interpolation=cv2.INTER_NEAREST))

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=riso_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
