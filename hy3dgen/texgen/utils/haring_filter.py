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

# "Haring" filter: replaces the mesh's painted albedo with a bold black/white organic
# maze/stripe pattern (reminiscent of Keith Haring's linework, or naturally-occurring
# Turing/reaction-diffusion patterns -- zebra stripes, coral, fingerprints), generated
# by repeatedly blurring the painted texture's luminance at two scales and feeding the
# difference back into the image ("Difference of Gaussians" reaction-diffusion): the
# small-sigma blur acts as a local "activator" pushing pixels apart from their
# immediate surroundings, the large-sigma blur acts as a broader "inhibitor" pulling
# them back -- iterated, this self-organizes flat/smooth regions into maze-like bands
# while real image structure (a face's eyes/nose/mouth, a horse's mane) seeds and
# steers the pattern rather than being erased by it.
#
# Unlike Dither/Stipple/Riso, this is NOT driven by real AO/geometry data -- it's a
# pure 2D image process on the existing painted texture's luminance. Three approaches
# were tried and rejected before this one (see conversation/commit history):
#   - Running the reaction-diffusion independently per RGB channel: each channel
#     traces a genuinely different maze (different starting gradients), so they never
#     agree pixel-for-pixel -- reads as garish rainbow-fringed speckle, not a clean
#     black/white pattern, no matter how many iterations.
#   - Re-coloring the converged pattern by keeping the original hue/saturation and
#     only replacing value/lightness: coherent color, but explicitly not what was
#     wanted here -- this filter is monochrome by design.
#   - Scaling the blur radii with working resolution to keep pattern density visually
#     constant across texture sizes: technically consistent, but the fixed-radius
#     look (pattern gets finer as resolution increases) was preferred instead, so
#     resolution is normalized (always resized to _WORK_RES) rather than scaled.

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

# Locked recipe from an interactive tuning session on real generated meshes (see
# conversation/commit history) -- picked by direct visual sweep, not derived from
# first principles.
_SIGMA_SMALL = 1.0
_SIGMA_RATIO = 3.0  # large sigma = density * this ratio, keeps the two scales locked together
_REACTION_STRENGTH = 1.0
# Always resized to this working resolution before iterating, regardless of the
# source texture's native size -- pattern density/stripe thickness stays visually
# consistent across meshes with different texture resolutions (a 256px lowpoly atlas
# and a 4096px one produce the same look), only sharpness at the final upscale
# differs. A fixed-pixel sigma at native resolution was tried and rejected: pattern
# density then depended on texture size, which read as an unrelated side effect
# rather than a creative choice.
_WORK_RES = 1024

# User-facing defaults -- the locked recipe, exposed as "Reset" values for the UI's
# creative knobs (see app.py). Everything else here (sigma ratio, work_res) is a
# fixed implementation detail.
DEFAULT_ITERATIONS = 30
DEFAULT_DENSITY = _SIGMA_SMALL * _SIGMA_RATIO
DEFAULT_CONTRAST = _REACTION_STRENGTH


def _reaction_diffusion(channel: np.ndarray, iterations: int, sigma_small: float,
                         sigma_large: float, k: float) -> np.ndarray:
    """Difference-of-Gaussians reaction-diffusion: repeatedly blurs `channel` at two
    scales and feeds `k` times their difference back in, self-organizing flat regions
    into maze/stripe bands. Converges within a few dozen iterations -- running far
    past that mostly just holds the pattern steady rather than changing it further."""
    img = channel.copy()
    for _ in range(iterations):
        blur_small = gaussian_filter(img, sigma_small)
        blur_large = gaussian_filter(img, sigma_large)
        img = np.clip(img + k * (blur_small - blur_large), 0, 1)
    return img


def apply_haring_filter(mesh: trimesh.Trimesh,
                         iterations: int = DEFAULT_ITERATIONS,
                         density: float = DEFAULT_DENSITY,
                         contrast: float = DEFAULT_CONTRAST) -> trimesh.Trimesh:
    """Replaces `mesh`'s painted albedo entirely with a bold black/white organic
    maze pattern grown from the existing painted texture's luminance (see module
    docstring). `mesh.visual.material` must already be a trimesh material with
    `baseColorTexture` set (i.e. already painted). Returns `mesh` for convenience; also
    mutates it. Purely visual/stylized replacement, so the material is reset to
    flat/matte (no metallic/roughness/AO/normal maps).

    `iterations`: how many blur-diffuse cycles to run -- the pattern converges within
    a few dozen, so this mostly matters at the low end (too few leaves it looking like
    a blurred photo rather than a committed maze pattern).
    `density`: the large-scale blur radius (small radius is locked at a fixed 1:3
    ratio to it) -- larger values give wider-spaced, chunkier bands.
    `contrast`: how strongly each iteration's difference gets fed back in -- higher
    pushes the pattern toward pure black/white faster and more starkly."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    small = albedo_tex.convert("L").resize((_WORK_RES, _WORK_RES), Image.LANCZOS)
    gray = np.asarray(small, dtype=np.float32) / 255.0

    sigma_small = density / _SIGMA_RATIO
    pattern = _reaction_diffusion(gray, iterations, sigma_small, density, contrast)

    pattern_u8 = (pattern * 255).clip(0, 255).astype(np.uint8)
    haring_tex = Image.fromarray(np.repeat(pattern_u8[..., None], 3, axis=-1), mode="RGB").resize(
        (texture_size, texture_size), Image.LANCZOS)

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=haring_tex,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh
