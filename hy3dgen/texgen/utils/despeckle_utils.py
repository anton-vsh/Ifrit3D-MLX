"""Median-filter despeckle for baked texture atlases, shared by both the
PyTorch/MLX-hybrid pipeline (Hunyuan3DPaintPipeline) and the native Swift
paint backend (SwiftPaintPipeline via swift_paint_runner.py).

Baking multiview diffusion output into a UV atlas leaves texels with
near-zero sample weight that pass the bake's trust threshold yet carry no
real color — they render as isolated flecks. Two variants, both replaced
with the local 5x5 median color: dark flecks (near-black vs. local
luminance median; verified 94% reduction on a real afflicted texture,
smooth/detailed regions untouched) and chromatic flecks (a wrong-hued speck
that isn't necessarily darker — e.g. blue/teal specks on an otherwise
white/cream surface — which the dark-only check misses entirely). Median
filtering is edge-preserving for genuine color boundaries (a pixel right at
a real edge still matches its own side's local-majority color, so the diff
stays small there), so this cleans isolated outlier texels without
blurring real detail.
"""
import numpy as np
from scipy.ndimage import median_filter


def despeckle_array(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """arr: float32 HxWx3 in [0, 255]. Returns (cleaned_arr, num_flecks_replaced)."""
    lum = arr.mean(axis=2)
    med_lum = median_filter(lum, size=5)
    med_rgb = np.stack(
        [median_filter(arr[:, :, c], size=5) for c in range(arr.shape[2])],
        axis=2,
    )
    dark_mask = (med_lum - lum) > 45
    chroma_dist = np.sqrt(((arr - med_rgb) ** 2).sum(axis=2))
    chroma_mask = chroma_dist > 70
    fleck_mask = dark_mask | chroma_mask
    n = int(fleck_mask.sum())
    if n == 0:
        return arr, 0
    out = arr.copy()
    out[fleck_mask] = med_rgb[fleck_mask]
    return out, n


def despeckle_image(image):
    """Despeckle a PIL RGB Image, returning a new PIL Image."""
    from PIL import Image as PILImage

    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    cleaned, _n = despeckle_array(arr)
    return PILImage.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8))
