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

# UV-space checkerboard: replaces the painted albedo with a flat two-color grid, cell
# count independently adjustable per UV axis (`cells_u`/`cells_v`), then shaded by the
# same real AO signal the other filters use (raycast_ao_raw, see mesh_ao.py) so the
# flat grid still reads the mesh's actual form instead of looking like a texture debug
# overlay pasted on top. AO here multiplies brightness (darkens occluded cells)
# rather than adding "ink" the way the black/white filters' darkness maps do, since
# there's no white paper to composite onto -- both checker colors need to darken
# in shadow, not just get an extra dark layer stacked on top.

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

from .mesh_ao import raycast_ao_raw

_WORK_RES = 1024
_AO_SIGMA = 5.0
_GAMMA = 0.6
# How dark full occlusion can push a cell -- 0 would crush shadowed cells to pure
# black regardless of their own color; 1 would disable AO shading entirely.
_AO_SHADOW_FLOOR = 0.25

DEFAULT_CELLS_U = 8
DEFAULT_CELLS_V = 8
DEFAULT_AO_STRENGTH = 1.5
DEFAULT_COLOR_A = "#1A1A1A"
DEFAULT_COLOR_B = "#F2F2F2"


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def apply_checkerboard_filter(mesh: trimesh.Trimesh,
                               cells_u: int = DEFAULT_CELLS_U,
                               cells_v: int = DEFAULT_CELLS_V,
                               ao_strength: float = DEFAULT_AO_STRENGTH,
                               color_a: str = None,
                               color_b: str = None) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo with an AO-shaded UV checkerboard (see module
    docstring). `mesh.visual.material` must already be a trimesh material with
    `baseColorTexture` set (i.e. already painted) -- only its resolution is used, not
    its content. Returns `mesh` for convenience; also mutates it. Purely visual/
    stylized replacement, so the material is reset to flat/matte (no metallic/
    roughness/AO/normal maps).

    `cells_u`/`cells_v` are independent horizontal/vertical cell counts across the full
    0-1 UV range. `color_a`/`color_b` accept "#rrggbb" hex strings and fall back to the
    locked default palette when None."""
    material = mesh.visual.material
    texture_size = material.baseColorTexture.size[0]

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    rgb_a = _hex_to_rgb(color_a) if color_a else _hex_to_rgb(DEFAULT_COLOR_A)
    rgb_b = _hex_to_rgb(color_b) if color_b else _hex_to_rgb(DEFAULT_COLOR_B)

    u = (np.arange(_WORK_RES, dtype=np.float32) + 0.5) / _WORK_RES
    v = (np.arange(_WORK_RES, dtype=np.float32) + 0.5) / _WORK_RES
    cell_u = np.floor(u * cells_u).astype(np.int64)
    cell_v = np.floor(v * cells_v).astype(np.int64)
    parity = (cell_u[None, :] + cell_v[:, None]) % 2  # rows = v (image y), cols = u (image x)

    checker = np.where(parity[..., None] == 0, rgb_a[None, None, :], rgb_b[None, None, :])

    ao_raw = raycast_ao_raw(mesh, render)
    ao_blurred = gaussian_filter(ao_raw, sigma=_AO_SIGMA)
    shading = np.clip(np.clip(ao_blurred, 0, 1) ** _GAMMA, 0, 1)
    shading = _AO_SHADOW_FLOOR + (1 - _AO_SHADOW_FLOOR) * np.clip(shading * ao_strength, 0, 1) \
        if ao_strength > 0 else np.ones_like(ao_blurred)
    shading_small = np.array(Image.fromarray((np.clip(shading, 0, 1) * 255).astype(np.uint8)).resize(
        (_WORK_RES, _WORK_RES), Image.LANCZOS)) / 255.0

    canvas = np.clip(checker * shading_small[..., None], 0, 255).astype(np.uint8)
    checker_tex = Image.fromarray(cv2.resize(canvas, (texture_size, texture_size), interpolation=cv2.INTER_NEAREST))

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=checker_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
