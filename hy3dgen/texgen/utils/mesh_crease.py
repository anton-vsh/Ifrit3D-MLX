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

import numpy as np
import torch
import trimesh


def bake_crease_map(mesh: trimesh.Trimesh, render, angle_deg: float = 25.0) -> np.ndarray:
    """Static (view-independent) substitute for a silhouette outline: bakes fixed hard
    mesh creases -- dihedral angle between adjacent faces above a threshold -- to UV
    space. A real silhouette outline is inherently view-dependent (it falls wherever
    the surface normal is perpendicular to the current camera direction, which changes
    every frame) and can't be baked into a static texture; this instead marks genuinely
    sharp, permanent geometric creases (ear rims, horn bases, hard edges) that stay
    consistent from any angle. Returns HxW float32 in [0, 1], 1 = sharp edge.
    `render` is a MeshRender with `mesh` already loaded (shared with the caller)."""
    verts = mesh.vertices
    rounded = verts.round(5)
    uniq_pos, inverse = np.unique(rounded, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_uniq = len(uniq_pos)

    dedup_faces = inverse[mesh.faces]
    dedup_mesh = trimesh.Trimesh(vertices=uniq_pos, faces=dedup_faces, process=False)
    dedup_mesh.update_faces(dedup_mesh.nondegenerate_faces())

    angles = dedup_mesh.face_adjacency_angles  # radians, per adjacent-face-pair
    edges = dedup_mesh.face_adjacency_edges    # [E, 2] vertex indices into uniq_pos
    thresh = np.radians(angle_deg)
    sharp = angles > thresh

    crease_uniq = np.zeros(n_uniq, dtype=np.float32)
    if sharp.any():
        sharp_edges = edges[sharp]
        sharp_weight = np.clip(angles[sharp] / np.pi, 0, 1)  # sharper = closer to 1
        np.maximum.at(crease_uniq, sharp_edges[:, 0], sharp_weight)
        np.maximum.at(crease_uniq, sharp_edges[:, 1], sharp_weight)

    crease_full = crease_uniq[inverse].astype(np.float32)
    crease_t = torch.from_numpy(crease_full).float().to(render.device).unsqueeze(-1).repeat(1, 3)
    crease_map_raw = render.uv_feature_map(crease_t).cpu().numpy()
    ones_t = torch.ones_like(crease_t)
    mask_map = render.uv_feature_map(ones_t).cpu().numpy()[..., 0]
    covered_mask = (mask_map > 0.5).astype(np.uint8) * 255
    crease_tex = render.uv_inpaint(crease_map_raw, covered_mask)[..., 0].astype(np.float32) / 255.0
    return crease_tex
