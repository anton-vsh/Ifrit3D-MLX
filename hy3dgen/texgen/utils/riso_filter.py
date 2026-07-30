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
# warm cream paper -- not user-selectable yet, just the single "Riso" filter option.
_PAPER = np.array([246, 242, 232], dtype=np.float32)
_COLOR_SHADOW = np.array([0, 120, 191], dtype=np.float32)
_COLOR_DETAIL = np.array([255, 72, 176], dtype=np.float32)


def _build_layers(mesh: trimesh.Trimesh, render, albedo: Image.Image):
    """Same underlying AO/crease/edge signal as dither_filter.py, but split into two
    darkness maps instead of merged into one -- shadow (AO form) and detail
    (crease + Canny albedo edges), each becomes its own ink layer."""
    ao_raw = raycast_ao_raw(mesh, render)
    broad = gaussian_filter(ao_raw, sigma=_AO_BROAD_SIGMA)
    fine = gaussian_filter(ao_raw, sigma=_AO_FINE_SIGMA)
    darkness_broad = np.clip((1 - np.clip(broad, 0, 1) ** _GAMMA) * _AO_BROAD_MULTIPLIER, 0, 1)
    darkness_fine = np.clip((1 - np.clip(fine, 0, 1) ** _GAMMA) * _AO_FINE_MULTIPLIER, 0, 1)
    shadow_darkness = np.maximum(darkness_broad, darkness_fine)
    shadow_small = np.array(Image.fromarray((shadow_darkness * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0

    crease = bake_crease_map(mesh, render, angle_deg=_CREASE_ANGLE_DEG)
    crease_small = np.array(Image.fromarray((crease * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0 * _CREASE_STRENGTH

    small_albedo = albedo.convert("L").resize((_WORK_RES, _WORK_RES), Image.LANCZOS)
    gray_u8 = np.asarray(small_albedo, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), _CANNY_BLUR)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
    edge_mask = (edges > 0).astype(np.float32)

    detail_darkness = np.clip(np.maximum(crease_small, edge_mask), 0, 1)
    return shadow_small, detail_darkness


def _riso_composite(shadow_darkness: np.ndarray, detail_darkness: np.ndarray) -> np.ndarray:
    brightness_shadow = np.clip(1 - shadow_darkness, 0, _WHITE_CAP)
    brightness_detail = np.clip(1 - detail_darkness, 0, _WHITE_CAP)

    ink_shadow = 1.0 - _atkinson_dither(brightness_shadow)
    ink_detail = 1.0 - _atkinson_dither(brightness_detail)

    if _OFFSET_SHADOW != (0, 0):
        ink_shadow = nd_shift(ink_shadow, _OFFSET_SHADOW, order=0, mode="constant", cval=0)
    if _OFFSET_DETAIL != (0, 0):
        ink_detail = nd_shift(ink_detail, _OFFSET_DETAIL, order=0, mode="constant", cval=0)

    canvas = np.tile(_PAPER, (*ink_shadow.shape, 1))
    # Multiply blend, like real overlaid transparent riso ink layers -- overlaps
    # produce a third, darker mixed color instead of one ink just replacing the other.
    for ink, color in [(ink_shadow, _COLOR_SHADOW), (ink_detail, _COLOR_DETAIL)]:
        blend = 1.0 - ink[..., None] * (1.0 - color[None, None, :] / 255.0)
        canvas = canvas * blend
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_riso_filter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo entirely with a stylized two-color risograph
    print texture (see module docstring). `mesh.visual.material` must already be a
    trimesh material with `baseColorTexture` set (i.e. already painted). Returns `mesh`
    for convenience; also mutates it. Purely visual/stylized replacement, so the
    material is reset to flat/matte (no metallic/roughness/AO/normal maps)."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    shadow_darkness, detail_darkness = _build_layers(mesh, render, albedo_tex)
    canvas = _riso_composite(shadow_darkness, detail_darkness)
    riso_tex = Image.fromarray(cv2.resize(canvas, (texture_size, texture_size),
                                           interpolation=cv2.INTER_NEAREST))

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=riso_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
