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

# 1-bit "old Mac" dither filter: replaces the mesh's painted albedo entirely with a
# stylized black/white Bayer (ordered) dither, driven by real geometric AO rather than
# the albedo's own colors. An ordered/Bayer dither of the raw painted texture was tried
# first, early in this feature's development, and rejected: it produced severe moire
# under 3D perspective (the fixed periodic grid beats against the viewing angle) and
# reacted to arbitrary albedo color noise rather than the mesh's actual form. Atkinson
# (1984 MacPaint) error-diffusion was adopted instead, since its non-periodic pattern
# doesn't moire the same way. Once the AO/crease/Canny-edge signal below matured,
# though, Bayer was revisited on top of it (see conversation/commit history) and
# explicitly chosen over Atkinson for the shipped "Dither" look despite the residual
# moire risk on smooth/simple geometry -- Atkinson is still used by the Riso filter
# (riso_filter.py imports `_atkinson_dither` from here), so both remain available.
#
# Three signals are combined into one "darkness" map before dithering:
#   - AO (mesh_ao.raycast_ao_raw): the dominant shading signal, real hemisphere-raycast
#     occlusion, combined at two blur scales -- a wide/broad pass so shadow transitions
#     read as long, soft gradients rather than sharp cutoffs, PLUS a narrow/fine pass
#     (much less blurred) so small real geometric detail (eye sockets, brow furrows,
#     ear rims) survives instead of being smoothed away by the broad pass alone. Each
#     scale's darkness is multiplied (not gamma'd) so partially-occluded areas push
#     toward full black without flattening open/convex areas, which stay genuinely
#     white; the two scales are combined by max so neither washes out the other.
#   - crease (mesh_crease.bake_crease_map): a static substitute for a view-dependent
#     silhouette outline (which can't be baked -- see conversation/commit history) --
#     fixed hard mesh creases (dihedral angle above a threshold) get a permanent edge
#     line regardless of camera angle.
#   - albedo edge detail: Canny edge detection (not a plain high-pass threshold) on the
#     *painted* albedo's own luminance, so real painted detail (eyes, embroidery,
#     marbling, carved texture) shows up as clean, continuous ink linework -- a plain
#     percentile-threshold high-pass was tried first and rejected: it produced isolated
#     speckled noise (many scattered single-pixel-ish blobs) rather than connected lines,
#     since a per-pixel contrast threshold has no notion of an edge as a continuous
#     structure. Canny's non-max-suppression + hysteresis linking gives clean traced
#     edges instead.
#
# Settings below are the result of an extended interactive tuning session (see
# conversation/commit history) across three very different test meshes (smooth glazed
# porcelain, an ornate mixed-material figurine, a sculpted marble bust) -- this is not
# exposed as user-facing parameters (yet), just the single "Dither" filter option.

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

from .mesh_ao import raycast_ao_raw
from .mesh_crease import bake_crease_map

_AO_BROAD_SIGMA = 6.0
_AO_FINE_SIGMA = 1.0
_AO_BROAD_MULTIPLIER = 4.0
_AO_FINE_MULTIPLIER = 3.0
_GAMMA = 0.6
_WHITE_CAP = 0.85
_CREASE_ANGLE_DEG = 15.0
_CREASE_STRENGTH = 1.0
_ALBEDO_DETAIL_STRENGTH = 0.75
# 1.0 (the original value) barely suppressed fine paint/skin-texture noise before
# Canny ran, so on any albedo with real micro-texture (fur, hair, skin pores) that
# noise had comparable gradient magnitude to genuine structural edges (eyes, nose,
# jaw line) and often out-competed them -- real contours got lost in noise rather than
# traced cleanly. 4.0 was picked by direct visual sweep against a real generated mesh
# (see conversation/commit history): strong enough to let real structure dominate, not
# so strong it erases small real features (e.g. eyes) entirely.
_ALBEDO_CANNY_BLUR = 4.0
# 40 (the original value), read at 4.0 blur, was actually a bit too permissive again
# (slightly more detail than the cleanest result in that same sweep) -- 35 is the
# picked balance between "detail visible" and "not noisy."
_ALBEDO_CANNY_LOW = 35
_ALBEDO_CANNY_HIGH = 120
# Canny's raw output is a 1px-wide line, which reads as too faint once resized up to
# the final texture resolution. Dilated with a small 2x2 kernel for a touch more
# weight -- a 3x3 kernel was tried first and looked too heavy/bold at this scale.
_ALBEDO_LINE_DILATE = 2
# 1024, not 512: the final NEAREST upscale to texture_size (2048) tiles each
# dither-resolution pixel into a flat block -- invisible where the darkness signal is
# smooth (AO/crease), but wherever the albedo-detail layer has genuine fine paint
# texture (e.g. marbling), a coarser dither_res made those blocks visible as blocky/
# jagged "triangular" artifacts under perspective on a curved surface. 1024 halves the
# upscale factor (4x -> 2x) and the blockiness disappears; higher still may look even
# cleaner but costs more (the ordered-dither tiling and Atkinson's error-diffusion
# loop, used by the Riso filter, both scale with dither_res^2).
_DITHER_RES = 1024


# Classic 4x4 Bayer ordered-dither threshold matrix, values normalized to [0, 1).
_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32) / 16.0


def _bayer_dither(brightness: np.ndarray, matrix: np.ndarray = _BAYER4) -> np.ndarray:
    """Ordered dither: tiles `matrix` across the image and thresholds each pixel
    against its corresponding matrix cell, giving the classic regular halftone-grid
    look (as opposed to Atkinson's non-periodic error-diffusion scatter)."""
    h, w = brightness.shape
    mh, mw = matrix.shape
    tile = np.tile(matrix, (h // mh + 1, w // mw + 1))[:h, :w]
    return (brightness > tile).astype(np.float32)


def _atkinson_dither(gray: np.ndarray) -> np.ndarray:
    """Classic 1984 Apple/MacPaint error-diffusion dither. Only 6/8 of the quantization
    error is distributed (vs Floyd-Steinberg's full 8/8) -- the discarded 2/8 is what
    gives Atkinson its higher-contrast, slightly "cleaner" look, and its non-periodic
    (image-dependent) error pattern avoids the fixed-grid moire that ordered/Bayer
    dithering produces under 3D perspective."""
    img = gray.astype(np.float32).copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 1.0 if old > 0.5 else 0.0
            err = (old - new) / 8.0
            img[y, x] = new
            if x + 1 < w:
                img[y, x + 1] += err
            if x + 2 < w:
                img[y, x + 2] += err
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += err
                img[y + 1, x] += err
                if x + 1 < w:
                    img[y + 1, x + 1] += err
            if y + 2 < h:
                img[y + 2, x] += err
    return img


# User-facing defaults -- the locked recipe from the tuning session, exposed as the
# "Reset" values for the six creative knobs the UI adjusts (see app.py). Everything
# else in this module (dither_res, white_cap, canny blur, gamma) is an implementation
# detail, not a look decision, and stays a fixed constant below.
DEFAULT_AO_STRENGTH = _AO_BROAD_MULTIPLIER
DEFAULT_AO_GRADIENT_LENGTH = _AO_BROAD_SIGMA
DEFAULT_FINE_DETAIL = _AO_FINE_MULTIPLIER
DEFAULT_CREASE_SENSITIVITY = _CREASE_ANGLE_DEG
DEFAULT_EDGE_STRENGTH = _ALBEDO_DETAIL_STRENGTH
DEFAULT_EDGE_SENSITIVITY = float(_ALBEDO_CANNY_LOW)


def _albedo_edge_mask(albedo: Image.Image, size: int, canny_low: float) -> np.ndarray:
    """Canny edges of the painted albedo's own luminance -- clean, continuous ink
    linework tracing real paint/material boundaries (eyes, embroidery, marbling),
    instead of a plain contrast-threshold's isolated speckled noise. `canny_low` is
    the single user-facing "edge sensitivity" knob; the high threshold is kept at the
    tuned 3x ratio to the low one rather than exposed separately."""
    small = albedo.convert("L").resize((size, size), Image.LANCZOS)
    gray_u8 = np.asarray(small, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), _ALBEDO_CANNY_BLUR)
    edges = cv2.Canny(blurred, canny_low, canny_low * 3)
    edges = cv2.dilate(edges, np.ones((_ALBEDO_LINE_DILATE, _ALBEDO_LINE_DILATE), np.uint8), iterations=1)
    return (edges > 0).astype(np.float32)


def apply_dither_filter(mesh: trimesh.Trimesh,
                         ao_strength: float = DEFAULT_AO_STRENGTH,
                         ao_gradient_length: float = DEFAULT_AO_GRADIENT_LENGTH,
                         fine_detail: float = DEFAULT_FINE_DETAIL,
                         crease_sensitivity: float = DEFAULT_CREASE_SENSITIVITY,
                         edge_strength: float = DEFAULT_EDGE_STRENGTH,
                         edge_sensitivity: float = DEFAULT_EDGE_SENSITIVITY) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo entirely with a stylized 1-bit Bayer (ordered)
    dither texture, driven by real AO + baked creases + Canny-detected albedo detail
    (see module docstring). `mesh.visual.material` must already be a trimesh material with
    `baseColorTexture` set (i.e. already painted). Returns `mesh` for convenience; also
    mutates it. This is a purely visual/stylized replacement, so the material is reset
    to flat/matte (no metallic/roughness/AO/normal maps -- those describe a real
    surface's light response, which doesn't apply to a flat black/white graphic).

    The six keyword args are the user-facing "creative" knobs (see app.py's Dither
    panel); everything else in this module is a fixed implementation detail."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    ao_raw = raycast_ao_raw(mesh, render)
    ao_broad = gaussian_filter(ao_raw, sigma=ao_gradient_length)
    ao_fine = gaussian_filter(ao_raw, sigma=_AO_FINE_SIGMA)
    darkness_broad = np.clip((1 - np.clip(ao_broad, 0, 1) ** _GAMMA) * ao_strength, 0, 1)
    darkness_fine = np.clip((1 - np.clip(ao_fine, 0, 1) ** _GAMMA) * fine_detail, 0, 1)
    darkness_ao = np.maximum(darkness_broad, darkness_fine)

    crease = bake_crease_map(mesh, render, angle_deg=crease_sensitivity)
    darkness_crease = crease * _CREASE_STRENGTH

    combined_hires = np.maximum(darkness_ao, darkness_crease)
    combined_small = np.array(Image.fromarray((combined_hires * 255).astype(np.uint8)).resize(
        (_DITHER_RES, _DITHER_RES), Image.LANCZOS)) / 255.0

    detail_mask = _albedo_edge_mask(albedo_tex, _DITHER_RES, edge_sensitivity)
    darkness_small = np.clip(combined_small + edge_strength * detail_mask, 0, 1)
    brightness_small = np.clip(1 - darkness_small, 0, _WHITE_CAP)

    dithered_small = _bayer_dither(brightness_small)
    dithered = np.array(Image.fromarray((dithered_small * 255).astype(np.uint8)).resize(
        (texture_size, texture_size), Image.NEAREST))
    dither_tex = Image.fromarray(np.repeat(dithered[..., None], 3, axis=-1), mode="RGB")

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=dither_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
