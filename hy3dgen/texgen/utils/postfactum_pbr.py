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

# Postfactum PBR: upgrades an already-baked albedo-only material (baseColorTexture,
# metallicFactor=0, roughnessFactor=1 -- mesh_utils.py's save_mesh default) into a real
# PBR material, without a second (slow) PBR diffusion pass:
#   - metallic/roughness: zero-shot CLIP material classification on the reference photo
#     (material_classifier.py) -- a semantic judgment, one flat value for the whole mesh,
#     set as glTF material factors (not baked textures -- they're genuinely uniform, no
#     per-texel signal exists to bake). A per-region (grid-classified, multiview-baked)
#     version of this was tried and reverted: CLIP scored isolated patches (e.g. a cat's
#     smooth, cool-toned ear crop with no face/body context) as confidently metallic
#     regardless of how much surrounding context was given, producing a hard wrong seam,
#     while widening the crop enough to fix that diluted genuine cases (a red-lacquer/
#     gold-trim horse) back to a uniform whole-photo bias, losing the split entirely. No
#     margin threaded both needles, so this reverted to the flat whole-photo estimate.
#   - ambient occlusion: real geometry -- hemisphere raycasting against the mesh itself
#     (Embree-accelerated via trimesh), baked to UV space, filled with the *actual* Tencent
#     reference inpaint algorithm (meshVerticeInpaint + cv2 Navier-Stokes -- see mesh_utils.py/
#     mesh_processor.py), then lightly blurred (AO has no fine detail worth protecting, unlike
#     albedo, so a blanket blur is safe here).
#   - normal (surface detail): Sobel gradients of the albedo's own luminance, baked in UV/
#     tangent space (not view/screen space) so it behaves like a normal normal map under any
#     lighting. This one is a heuristic, not a measurement or a judgment -- there is no real
#     higher-frequency surface data beyond the mesh itself, so this is a guess that "this
#     photo contrast implies this relief," which is only sometimes true. Strength is scaled
#     by the roughness estimate: glossy/glazed objects (low roughness) are far more likely to
#     have flat painted detail with no real bump, so their effective strength is small;
#     rough/matte objects get more, where a correlation between contrast and relief is more
#     plausible. Verified end-to-end (see conversation/commit history): CLIP scores correctly
#     separate ceramic/stone/metal-hull test cases; AO shows genuine crevice shading after the
#     real inpaint fill; the normal map's visibility is confirmed lighting-direction-dependent
#     like any real normal map (near-front lighting suppresses it, grazing light reveals it --
#     this is expected of all normal maps, not a defect).
#
# All four numbers/maps together cost roughly 6s after the CLIP model is warm (~12s cold,
# one-time) on a 100k-face/2048px-texture mesh -- negligible against paint generation time.

import threading

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel

from .material_classifier import MaterialClassifier
from .mesh_ao import raycast_ao_raw

_CLASSIFIER_LOCK = threading.Lock()
_classifier = None


def _get_material_classifier(progress_callback=None):
    global _classifier
    with _CLASSIFIER_LOCK:
        if _classifier is None:
            from hf_progress import report_hf_downloads
            with report_hf_downloads(progress_callback, "Downloading material classifier (first run only)"):
                _classifier = MaterialClassifier()
        return _classifier


def _bake_ao(mesh: trimesh.Trimesh, render, n_rays: int = 32, seed: int = 0) -> np.ndarray:
    """Realistic-lighting AO: the shared raycast (see mesh_ao.raycast_ao_raw), blurred to
    smooth per-vertex raycast sampling noise (sigma=3 wasn't enough on meshes with a lot of
    small-scale surface variation -- read as patchy/confetti-like rather than a clean
    gradient; sigma=8 fixes that), then floored. AO here can reach literal 0 on genuinely
    deep concave geometry (e.g. the inside of a bowl -- that's correct occlusion, not a
    bug), but letting it multiply lighting all the way to black looks harsh; flooring at
    0.35 is standard practice in production AO (real engines almost always clamp AO's
    influence rather than allow full black). Returns HxW float32 in [0.35, 1.0]."""
    ao_tex = raycast_ao_raw(mesh, render, n_rays=n_rays, seed=seed)
    smoothed = gaussian_filter(ao_tex, sigma=8)
    return 0.35 + 0.65 * smoothed


def _bake_normal_from_luminance(albedo_tex: Image.Image, strength: float) -> Image.Image:
    """Tangent-space normal map from Sobel gradients of the albedo's own luminance, baked
    directly in UV space (not view-dependent screen space) so it's a real, reusable asset
    under any lighting. `strength` should already be roughness-scaled by the caller."""
    albedo_arr = np.asarray(albedo_tex.convert("RGB")).astype(np.float32) / 255.0
    lum = gaussian_filter(albedo_arr.mean(axis=-1), sigma=1.0)
    gx = sobel(lum, axis=1)
    gy = sobel(lum, axis=0)

    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny, nz = nx / length, ny / length, nz / length

    normal_map = np.stack([nx, ny, nz], axis=-1)
    normal_map = ((normal_map * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(normal_map, mode="RGB")


def apply_postfactum_pbr(mesh: trimesh.Trimesh, reference_image: Image.Image,
                         bump_base_strength: float = 0.08,
                         progress_callback=None) -> trimesh.Trimesh:
    """Upgrades `mesh`'s existing albedo-only material in place with metallic/roughness
    factors, a baked AO texture, and a baked normal map. `mesh.visual.material` must already
    be a trimesh PBRMaterial with `baseColorTexture` set (i.e. already painted). Returns
    `mesh` for convenience; also mutates it."""
    material = mesh.visual.material
    albedo_tex = material.baseColorTexture.convert("RGB")
    texture_size = albedo_tex.size[0]

    classifier = _get_material_classifier(progress_callback=progress_callback)
    metallic_val, roughness_val = classifier.estimate(reference_image)
    # Compress the raw CLIP roughness estimate away from the extremes (e.g. a glazed-porcelain
    # photo shot without a strong visible specular highlight can score as high as ~0.5-0.6 --
    # verified this isn't a background/lighting artifact of the classifier, it's a genuine
    # judgment call CLIP gets wrong on some photos). This doesn't fix misclassification, it
    # softens its consequence: a compressed range means a wrong guess still lands somewhere
    # plausible instead of fully flat/matte, at the cost of slightly narrowing the real
    # distinction between genuinely glossy and genuinely matte objects.
    roughness_val = 0.05 + 0.7 * roughness_val

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    render = MeshRender(texture_size=texture_size)
    render.load_mesh(mesh)

    ao_tex = _bake_ao(mesh, render)
    ao_image = Image.fromarray((ao_tex * 255).clip(0, 255).astype(np.uint8), mode="L").convert("RGB")

    normal_image = _bake_normal_from_luminance(albedo_tex, strength=bump_base_strength * roughness_val)

    base_color_factor = getattr(material, "baseColorFactor", None)
    if base_color_factor is None:
        base_color_factor = [1.0, 1.0, 1.0, 1.0]

    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=material.baseColorTexture,
        baseColorFactor=base_color_factor,
        # No metallicRoughnessTexture, so these factors ARE the material response (see
        # swift/patches/... for why that distinction matters) -- flat, whole-mesh values.
        metallicFactor=float(metallic_val),
        roughnessFactor=float(roughness_val),
        occlusionTexture=ao_image,
        normalTexture=normal_image,
    )
    return mesh
