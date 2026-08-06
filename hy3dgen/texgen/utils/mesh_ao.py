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

# Shared real-geometry AO raycast, used by both postfactum_pbr.py (realistic PBR
# occlusion, floored/blurred for lighting use) and dither_filter.py (stylized shading
# signal, its own contrast/blur tuning) -- the raycasting and UV baking is identical,
# only the post-processing differs per caller.
#
# Raycasting itself runs on mtlbvh's Metal-accelerated BVH (already an unconditional
# project dependency -- see libraries/mtlbvh), NOT trimesh's `mesh.ray`. That matters:
# this project has no Embree/pyembree binding installed (only bare `rtree`), so
# `mesh.ray` resolves to trimesh's pure-Python RayMeshIntersector fallback, which does
# not scale gracefully with ray count -- confirmed directly (see conversation/commit
# history) that a real generated+simplified mesh with ~25k unique vertices (32 rays/
# vertex = ~800k total rays) never returned from `mesh.ray.intersects_any` in a
# reasonable wait, hanging every AO-driven filter (Dither/Stipple/Riso/Halftone/
# Checkerboard alike, they all call this function). mtlbvh traces the same ~800k rays
# in well under a tenth of a second.

import numpy as np
import torch
import trimesh
from mtlbvh import MtlBVH


def raycast_ao_raw(mesh: trimesh.Trimesh, render, n_rays: int = 32, seed: int = 0) -> np.ndarray:
    """Real geometric AO: dedup by position (UV-seam duplicates share a value for free),
    smooth per-vertex normals (averaged over corner copies -- the raw per-corner normals on
    this unwelded mesh convention are faceted, not smooth), one batched mtlbvh raycast,
    baked to UV space via the same interpolation machinery the pipeline already uses for
    position/normal maps, then filled with the real reference inpaint. Returns HxW
    float32 in [0, 1], NOT floored or blurred -- callers apply their own post-processing.
    `render` is a MeshRender with `mesh` already loaded (shared with the caller)."""
    verts = mesh.vertices
    rounded = verts.round(5)
    uniq_pos, inverse = np.unique(rounded, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_uniq = len(uniq_pos)

    corner_normals = mesh.vertex_normals
    sum_n = np.zeros((n_uniq, 3), dtype=np.float64)
    np.add.at(sum_n, inverse, corner_normals)
    norm = np.linalg.norm(sum_n, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1
    uniq_normals = (sum_n / norm).astype(np.float32)
    uniq_verts = uniq_pos.astype(np.float32)

    rng = np.random.default_rng(seed)
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    eps = bbox_diag * 1e-4

    def cosine_hemisphere_batch(normals, k):
        n = normals
        ref = np.where(np.abs(n[:, :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        t = np.cross(ref, n)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        b = np.cross(n, t)
        u1 = rng.random((len(n), k))
        u2 = rng.random((len(n), k))
        r = np.sqrt(u1)
        theta = 2 * np.pi * u2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(np.maximum(0, 1 - u1))
        return x[..., None] * t[:, None, :] + y[..., None] * b[:, None, :] + z[..., None] * n[:, None, :]

    dirs = cosine_hemisphere_batch(uniq_normals, n_rays)
    origins = np.broadcast_to((uniq_verts[:, None, :] + uniq_normals[:, None, :] * eps), dirs.shape).reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)

    # mesh.vertices/.faces here are this pipeline's usual unwelded-per-corner convention
    # (see corner_normals above) -- fine for BVH construction, which only needs
    # triangle positions, not a watertight/shared-vertex topology.
    bvh = MtlBVH(
        torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(render.device),
        torch.from_numpy(np.asarray(mesh.faces, dtype=np.int32)).to(render.device),
    )
    origins_t = torch.from_numpy(origins.astype(np.float32)).to(render.device)
    dirs_t = torch.from_numpy(dirs.astype(np.float32)).to(render.device)
    _, face_id, _ = bvh.ray_trace(origins_t, dirs_t)
    hits = (face_id >= 0).cpu().numpy()

    ao_uniq = 1.0 - hits.reshape(n_uniq, n_rays).mean(axis=1)
    ao_full = ao_uniq[inverse].astype(np.float32)

    ao_t = torch.from_numpy(ao_full).float().to(render.device).unsqueeze(-1).repeat(1, 3)
    ones_t = torch.ones_like(ao_t)
    ao_map_raw = render.uv_feature_map(ao_t).cpu().numpy()
    mask_map = render.uv_feature_map(ones_t).cpu().numpy()[..., 0]
    covered_mask = (mask_map > 0.5).astype(np.uint8) * 255

    ao_tex = render.uv_inpaint(ao_map_raw, covered_mask)[..., 0].astype(np.float32) / 255.0
    return ao_tex
