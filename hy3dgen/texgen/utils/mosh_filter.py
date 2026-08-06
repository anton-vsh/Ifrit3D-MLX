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

# Datamosh-style vertex glitch: picks a handful of contiguous ranges in the mesh's own
# vertex array and, per range, either overwrites them with another random range's
# current positions ("P-frame" leaking stale data) or nudges them by a shared random
# offset. Ported 1:1 from a standalone three.js prototype (index.html's
# moshVertices) -- same seeded LCG-equivalent draw order, same block-size/count
# formula, same per-vertex copy-or-displace coin flip. Purely a position edit: faces
# and UVs are left untouched, so the mesh stays topologically valid (texture just
# stretches/tears across moved vertices, which reads as part of the glitch).
#
# Reads (for the copy branch) and writes into the same evolving position buffer,
# block after block -- a later block can copy from a region an earlier block already
# moved, producing the same cascading "compression artifact" look as the reference.

import numpy as np
import trimesh

DEFAULT_INTENSITY = 0.5
DEFAULT_DISPLACEMENT = 0.5


def apply_mosh_filter(mesh: trimesh.Trimesh,
                       intensity: float = DEFAULT_INTENSITY,
                       displacement: float = DEFAULT_DISPLACEMENT,
                       seed: int = 0) -> trimesh.Trimesh:
    """Glitches `mesh`'s vertex positions in place (see module docstring). Unlike the
    other filters in this package, this one never touches `mesh.visual` -- it works on
    any mesh, painted or not, and can be freely combined with a texture filter (Dither/
    Stipple/Riso/Haring) since it doesn't consume the albedo/AO signal they use.

    `intensity` (0-1) controls both how many vertices land inside each moshed block and
    how many blocks run (1 to 4). `displacement` (0-1) scales the random offset applied
    to the non-copied half of each block. `seed` makes the glitch pattern reproducible
    across runs of the same generation (see app.py's call site, which threads the run's
    resolved seed through here). Returns `mesh` for convenience; also mutates it."""
    if intensity <= 0:
        return mesh

    verts = mesh.vertices
    num = len(verts)
    if num == 0:
        return mesh

    rng = np.random.default_rng(seed & 0xFFFFFFFF)

    block_size = max(8, int(num * 0.1 * intensity))
    num_blocks = max(1, min(4, int(intensity * 4)))
    offset_scale = 0.5 * intensity * displacement * 2

    moshed = np.array(verts, dtype=np.float64, copy=True)
    for _ in range(num_blocks):
        src = int(rng.integers(0, max(1, num - block_size)))
        dst = int(rng.integers(0, max(1, num - block_size)))
        n = min(block_size, num - dst, num - src)
        if n <= 0:
            continue
        offset = (rng.random(3) - 0.5) * offset_scale
        for i in range(n):
            if rng.random() < 0.5:
                moshed[dst + i] = moshed[src + i]
            else:
                moshed[dst + i] += offset

    mesh.vertices = moshed
    return mesh
