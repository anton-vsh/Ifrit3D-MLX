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

# Unlike every other filter in this package, this one bakes nothing. The painted
# texture is dropped entirely in favor of a flat glTF PBR material (a single solid
# color, high metalness, low roughness). The "fresnel" look -- edges brightening at
# grazing viewing angles -- isn't baked as a static image because it can't be: it's a
# function of view direction, which changes continuously as the viewer orbits the
# model. Instead it comes for free from the *real-time* PBR shader every glTF viewer
# (including this app's Model3D/Babylon.js viewer, which ships a default HDRI
# environment for image-based lighting) already runs: a low-roughness, high-metalness
# surface reflects its environment more strongly at grazing angles per the physical
# Schlick-fresnel term baked into the glTF metallic-roughness BRDF itself. So this
# genuinely updates in real time as the model is orbited, unlike a baked texture --
# there is deliberately no glow/emissive rim option here, since that *would* need to be
# baked for one fixed viewing angle and would look wrong from every other angle.

import trimesh

DEFAULT_COLOR = "#000000"
DEFAULT_METALLIC = 1.0
DEFAULT_ROUGHNESS = 0.0


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def apply_fresnel_filter(mesh: trimesh.Trimesh,
                          color: str = None,
                          metallic: float = DEFAULT_METALLIC,
                          roughness: float = DEFAULT_ROUGHNESS) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo with a flat, high-metalness/low-roughness PBR
    material in `color` (see module docstring) -- a real-time fresnel/reflectivity
    response, not a baked texture. Returns `mesh` for convenience; also mutates it.

    `color` accepts a "#rrggbb" hex string and falls back to the locked default when
    None. `metallic`/`roughness` are 0-1 PBR factors (see app.py's Fresnel panel)."""
    rgb = _hex_to_rgb01(color) if color else _hex_to_rgb01(DEFAULT_COLOR)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
    )
    return mesh
