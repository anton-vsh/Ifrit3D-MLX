import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import warnings
warnings.filterwarnings("ignore", "resource_tracker: There appear to be.*leaked semaphore.*")

import zipfile
import datetime
import random
import signal
import time
import threading
from pathlib import Path
from argparse import Namespace

import psutil
import trimesh
import gradio as gr
import starlette.status
import torch

starlette.status.HTTP_422_UNPROCESSABLE_ENTITY = starlette.status.HTTP_422_UNPROCESSABLE_CONTENT

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

os.environ.setdefault("HY3DGEN_MODELS", str(ROOT / "models" / "hy3dgen"))

_last_original_input = None
_last_original_inputs = None  # list of all input image paths (1 for sv, 3 for mv)


def _get_memory_stats():
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / 1e9
    ram_total_gb = vm.total / 1e9
    ram_pct = vm.used / vm.total * 100
    mps_gb = 0.0
    if torch.backends.mps.is_available():
        mps_gb = torch.mps.current_allocated_memory() / 1e9
    status = "[OK]" if ram_pct < 70 else ("[WARN]" if ram_pct < 85 else "[HIGH]")
    return f"{status} RAM {ram_pct:.0f}% ({ram_used_gb:.1f}/{ram_total_gb:.0f} GB)  |  MPS {mps_gb:.2f} GB"


def _shutdown_server():
    def _kill():
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return "[STOPPED] Shutting down..."


# Single-model app: 2.0-turbo for both shape and paint (2.1 dropped — slower,
# and it lost every same-input quality comparison run on 2026-07-18).
# Shape defaults to the Swift MLX binary when the local build is present
# (swift/bin is gitignored — fresh clones and the DMG fall back to pytorch).
def _default_shape_backend():
    try:
        from shape.swift_runner import swift_shape_available
        return "swift" if swift_shape_available() else "pytorch"
    except Exception:
        return "pytorch"


SHAPE_BACKEND_DEFAULT = _default_shape_backend()

_sd_pipeline = None
_sd_pipeline_mode = None  # "text2img" or "img2img"
_sd_lock = threading.Lock()


SD_MODEL_PATH = str(ROOT / "models" / "sd-turbo")


def _unload_sd_pipeline():
    global _sd_pipeline, _sd_pipeline_mode
    if _sd_pipeline is not None:
        import torch
        _sd_pipeline.to("cpu")
        _sd_pipeline = None
        _sd_pipeline_mode = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc
        gc.collect()
        print("[app] SD pipeline unloaded, MPS memory freed")


def _load_sd_pipeline(img2img=True):
    global _sd_pipeline, _sd_pipeline_mode
    mode = "img2img" if img2img else "text2img"
    if _sd_pipeline is not None and _sd_pipeline_mode == mode:
        return _sd_pipeline
    with _sd_lock:
        if _sd_pipeline is not None and _sd_pipeline_mode == mode:
            return _sd_pipeline
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        pipeline_cls = AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
        _sd_pipeline = pipeline_cls.from_pretrained(
            SD_MODEL_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to("mps" if torch.backends.mps.is_available() else "cpu")
        _sd_pipeline_mode = mode
        return _sd_pipeline


def generate_image(prompt, sd_steps, progress=gr.Progress()):
    import torch

    progress(0, desc="Loading SD Turbo...")
    pipe = _load_sd_pipeline(img2img=False)

    progress(0.3, desc=f"Generating ({sd_steps} steps)...")
    start = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=sd_steps,
            guidance_scale=0.0,
            height=512,
            width=512,
        ).images[0]
    elapsed = time.time() - start
    print(f"[app] SD generated in {elapsed:.1f}s")

    progress(0.9, desc="Freeing GPU memory...")
    _unload_sd_pipeline()

    progress(1.0, desc="Done!")
    return result


def _export_obj_zip(textured, tmp_dir):
    obj_dir = tmp_dir / "obj"
    obj_dir.mkdir(exist_ok=True)
    obj_path = obj_dir / "model.obj"
    textured.export(str(obj_path), file_type="obj")

    zip_path = tmp_dir / "model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in obj_dir.iterdir():
            zf.write(f, f.name)
    return str(zip_path)


def _scaled_progress(progress, lo, hi):
    def _cb(fraction, desc=None):
        progress(lo + fraction * (hi - lo), desc=desc)
    return _cb


def _free_mps():
    import torch
    import gc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    print("[app] MPS cache cleared")


def _run_retexture(
    glb_path, image_paths_state, use_delight,
    seed, randomize_seed,
    texture_sd_detail=False,
    use_swift_paint=False,
    paint_res=512,
    paint_steps=15,
    paint_guidance=2.0,
    paint_tex=2048,
    paint_superres=False,
    paint_sd_strength=0.3,
    paint_sd_res=768,
    progress=gr.Progress(),
):
    if not glb_path:
        raise gr.Error("Generate a model first!")
    if not image_paths_state:
        raise gr.Error("No input image found — generate a model first!")

    _free_mps()

    from main import run_paint_pipeline as run_paint

    progress(0, desc="Loading mesh...")
    loaded = trimesh.load(glb_path)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded

    os.environ['HY3D_USE_DELIGHT'] = '1' if use_delight else '0'
    os.environ['HY3D_USE_SD_UPSCALE'] = '1' if (texture_sd_detail and not use_swift_paint) else '0'

    resolved_seed = random.randint(0, 999999) if randomize_seed else seed

    args = Namespace(
        no_rembg=False,
        seed=resolved_seed,
        paint_preset="2.0" if use_swift_paint else "2.0-turbo",
        paint_model_repo=None,
        paint_subfolder=None,
        paint_diffusion_backend="swift" if use_swift_paint else "mlx",
        paint_mlx_weights=None,
        paint_basic_texture=True,
        paint_res=paint_res,
        paint_steps=paint_steps,
        paint_guidance=paint_guidance,
        paint_tex=paint_tex,
        paint_superres=paint_superres,
        paint_sd_detail=texture_sd_detail if use_swift_paint else False,
        paint_sd_strength=paint_sd_strength,
        paint_sd_res=paint_sd_res,
    )

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-retexture")
    run_dir = OUTPUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(0.05, desc=f"Re-texturing (seed={resolved_seed})...")
    image_paths = [Path(p) for p in image_paths_state]
    textured = run_paint(mesh, image_paths, args, progress_callback=_scaled_progress(progress, 0.05, 0.9))
    print(f"[app] retexture done (seed={resolved_seed})")

    glb_out = run_dir / "result.glb"
    textured.export(glb_out)

    progress(0.9, desc="Exporting OBJ...")
    obj_zip = _export_obj_zip(textured, run_dir)

    progress(1.0, desc="Done!")
    _free_mps()
    return str(glb_out), str(glb_out), obj_zip, str(glb_out)


def generate(
    image_path,
    use_delight,
    seed,
    shape_steps,
    shape_octree_resolution,
    simplify_before_texturing,
    target_faces,
    skip_texturing=False,
    texture_sd_detail=False,
    use_swift_paint=False,
    paint_res=512,
    paint_steps=15,
    paint_guidance=2.0,
    paint_tex=2048,
    paint_superres=False,
    paint_sd_strength=0.3,
    paint_sd_res=768,
    shape_backend=SHAPE_BACKEND_DEFAULT,
    run_dir=None,
    progress=gr.Progress(),
):
    global _last_original_input, _last_original_inputs
    _last_original_input = image_path

    from PIL import Image
    from shape.runner import run_shape_pipeline as run_shape
    from main import run_paint_pipeline as run_paint

    progress(0, desc="Preparing...")

    if use_delight:
        os.environ['HY3D_USE_DELIGHT'] = '1'
    else:
        os.environ.pop('HY3D_USE_DELIGHT', None)

    os.environ['HY3D_USE_FLASHVDM'] = '0'
    os.environ['HY3D_USE_SD_UPSCALE'] = '1' if (texture_sd_detail and not use_swift_paint) else '0'

    img = Image.open(image_path)
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = OUTPUT_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
    img_path = run_dir / "input.png"
    img.save(img_path, "PNG")

    image_paths = [img_path]
    mode = "sv"

    _last_original_inputs = image_paths

    args = Namespace(
        shape_preset="2.0-turbo",
        shape_model_repo=None,
        shape_subfolder=None,
        shape_variant="fp16",
        no_shape_safetensors=False,
        shape_steps=shape_steps,
        shape_octree_resolution=shape_octree_resolution,
        shape_num_chunks=12000,
        shape_backend=shape_backend,
        no_rembg=False,
        seed=seed,
        paint_preset="2.0" if use_swift_paint else "2.0-turbo",
        paint_model_repo=None,
        paint_subfolder=None,
        paint_diffusion_backend="swift" if use_swift_paint else "mlx",
        paint_mlx_weights=None,
        paint_basic_texture=True,
        paint_res=paint_res,
        paint_steps=paint_steps,
        paint_guidance=paint_guidance,
        paint_tex=paint_tex,
        paint_superres=paint_superres,
        paint_sd_detail=texture_sd_detail if use_swift_paint else False,
        paint_sd_strength=paint_sd_strength,
        paint_sd_res=paint_sd_res,
    )

    progress(0.05, desc="Generating 3D shape...")
    start = time.time()
    mesh = run_shape(
        image_paths, mode, args, forced_preset="2.0-turbo",
        progress_callback=_scaled_progress(progress, 0.05, 0.40),
    )
    print(f"[app] shape done in {time.time() - start:.1f}s")

    if simplify_before_texturing:
        face_before = len(mesh.faces)
        progress(0.40, desc=f"Simplifying {face_before} → {target_faces} faces before texturing...")
        start = time.time()
        mesh = _simplify_mesh(mesh, target_faces)
        print(f"[app] simplify done in {time.time() - start:.1f}s")

    if skip_texturing:
        glb_path = run_dir / "result.glb"
        mesh.export(glb_path)
        progress(0.9, desc="Exporting OBJ...")
        obj_zip_path = _export_obj_zip(mesh, run_dir)
        progress(1.0, desc="Done!")
        _free_mps()
        return str(glb_path), str(glb_path), obj_zip_path, str(glb_path)

    progress(0.45, desc="Texturing...")
    start = time.time()
    textured = run_paint(mesh, image_paths, args, progress_callback=_scaled_progress(progress, 0.45, 0.88))
    print(f"[app] paint done in {time.time() - start:.1f}s")

    glb_path = run_dir / "result.glb"
    textured.export(glb_path)

    progress(0.9, desc="Exporting OBJ...")
    obj_zip_path = _export_obj_zip(textured, run_dir)

    progress(1.0, desc="Done!")
    _free_mps()
    return str(glb_path), str(glb_path), obj_zip_path, str(glb_path)


def _unweld_uvs(mesh):
    import numpy as np
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return mesh

    verts = np.asarray(mesh.vertices)
    uv = np.asarray(mesh.visual.uv) if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None else None

    if uv is None:
        return mesh

    new_verts = verts[faces].reshape(-1, 3)
    new_uv = uv[faces].reshape(-1, 2)
    new_faces = np.arange(len(faces) * 3, dtype=np.int32).reshape(-1, 3)

    img = None
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        mat = mesh.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            img = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image is not None:
            img = mat.image
    if img is None and hasattr(mesh.visual, 'image'):
        img = mesh.visual.image

    result = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        visual=trimesh.visual.TextureVisuals(uv=new_uv, image=img) if img is not None else None,
        process=False,
    )
    result.vertex_normals  # compute per-vertex normals
    return result


def _simplify_mesh(mesh, target_faces):
    import pymeshlab
    import numpy as np
    from trimesh.proximity import closest_point

    face_count = len(mesh.faces)
    if face_count <= target_faces:
        return mesh

    orig_mesh = mesh.copy()

    texture_img = None
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        mat = mesh.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            texture_img = mat.baseColorTexture
        elif hasattr(mat, 'image') and mat.image is not None:
            texture_img = mat.image

    uv = mesh.visual.uv if hasattr(mesh.visual, 'uv') else None

    ms = pymeshlab.MeshSet()
    kwargs = dict(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    if uv is not None:
        kwargs['v_tex_coords_matrix'] = np.asarray(uv, dtype=np.float64)

    ms.add_mesh(pymeshlab.Mesh(**kwargs))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=min(target_faces, face_count),
        preservetopology=True,
        preserveboundary=True,
        preservenormal=True,
    )
    ms.meshing_close_holes(maxholesize=30)
    ms.meshing_repair_non_manifold_edges()

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    result = trimesh.Trimesh(verts, faces)

    if texture_img is not None:
        closest, dist, tri_ids = closest_point(orig_mesh, verts)
        tris = orig_mesh.faces[tri_ids]
        bary = trimesh.triangles.points_to_barycentric(
            orig_mesh.vertices[tris], closest
        )
        new_uv = (orig_mesh.visual.uv[tris] * bary[:, :, None]).sum(axis=1)
        result.visual = trimesh.visual.TextureVisuals(uv=new_uv, image=texture_img)

    result.vertex_normals

    return _unweld_uvs(result)


# Presets differ in geometry detail, reduction target, and texture-detail
# passes — the shape model (2.0-turbo) and steps (8, its distillation
# schedule) are fixed, and target_faces for normal/high are calibrated from
# measured raw (post fragment-filter, pre-simplification) face counts at
# each octree resolution (96->61,172, 192->245,356, 256->436,076 on the same
# test input) — both targets keep the same ~41% retention ratio, so "high"
# carries proportionally more absolute detail than "normal" rather than an
# arbitrary round number. Delight and skip-texturing are off for every
# preset. Paint now runs on the Swift native backend for every preset —
# res/tex scale with how much the mesh's own poly budget can actually show
# (e.g. lowpoly's 500-face mesh gets a small 256 atlas, not a wasted 2048
# one); steps scale lowpoly->high for a real quality ladder; guidance is 1.5
# for lowpoly/normal/high (2.0 measured harsher/oversaturated in an A/B, see
# swift/README.md) but 2 for draft; upscale-texture (SD-Turbo per-view pass)
# mirrors the old on/off split (off for lowpoly/draft, on for normal/high)
# at strength 0.3, resolution 512. use_swift_paint/paint_superres/
# paint_sd_strength/paint_sd_res are no longer user-adjustable in the UI
# (backed by gr.State instead of visible sliders) but stay part of this
# return tuple since presets still set them via the same outputs mechanism.
# Returns: shape_steps, shape_octree_resolution, use_delight, simplify_before,
# target_faces, skip_texturing, texture_sd_detail, use_swift_paint,
# paint_res, paint_steps, paint_guidance, paint_tex, paint_superres,
# paint_sd_strength, paint_sd_res
def _set_preset(name):
    if name == "lowpoly":
        return 8, 96, False, True, 500, False, False, True, 512, 4, 1.5, 256, False, 0.3, 512
    elif name == "draft":
        return 8, 160, False, True, 50000, False, False, True, 384, 10, 2, 1024, False, 0.3, 512
    elif name == "normal":
        return 8, 192, False, True, 100000, False, True, True, 512, 15, 1.5, 2048, False, 0.3, 512
    else:  # high
        return 8, 256, False, True, 180000, False, True, True, 512, 20, 1.5, 2048, False, 0.3, 512



def _simplify_current(glb_path, target_faces, progress=gr.Progress()):
    if not glb_path:
        raise gr.Error("Generate a model first!")

    progress(0, desc="Loading mesh...")
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    face_before = len(mesh.faces)

    progress(0.3, desc=f"Simplifying {face_before} → {target_faces} faces...")
    simplified = _simplify_mesh(mesh, target_faces)

    progress(0.8, desc="Exporting...")
    out_path = Path(glb_path)
    simplified.export(str(out_path))

    progress(0.9, desc="Exporting OBJ...")
    obj_zip = _export_obj_zip(simplified, out_path.parent)

    progress(1.0, desc="Done!")
    return str(out_path), str(out_path), obj_zip, str(out_path)


BRUTAL_THEME = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    body_text_color="#000000",
    body_text_color_dark="#000000",
    body_text_color_subdued="#000000",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#ffffff",
    background_fill_secondary_dark="#ffffff",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_text_color="#000000",
    block_label_text_color_dark="#000000",
    block_title_text_color="#000000",
    block_title_text_color_dark="#000000",
    border_color_primary="#000000",
    border_color_primary_dark="#000000",
    block_border_color="#000000",
    block_border_color_dark="#000000",
    block_border_width="2px",
    block_border_width_dark="2px",
    block_radius="0px",
    block_shadow="none",
    block_shadow_dark="none",
    shadow_drop="none",
    shadow_drop_lg="none",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#000000",
    input_border_color_dark="#000000",
    input_border_width="2px",
    input_radius="0px",
    slider_color="#000000",
    slider_color_dark="#000000",
    button_transition="none",
    button_large_radius="0px",
    button_small_radius="0px",
    button_primary_background_fill="#000000",
    button_primary_background_fill_dark="#000000",
    button_primary_background_fill_hover="#ffffff",
    button_primary_background_fill_hover_dark="#ffffff",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_text_color_hover="#000000",
    button_primary_text_color_hover_dark="#000000",
    button_primary_border_color="#000000",
    button_primary_border_color_dark="#000000",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover="#000000",
    button_secondary_background_fill_hover_dark="#000000",
    button_secondary_text_color="#000000",
    button_secondary_text_color_dark="#000000",
    button_secondary_text_color_hover="#ffffff",
    button_secondary_text_color_hover_dark="#ffffff",
    button_secondary_border_color="#000000",
    button_secondary_border_color_dark="#000000",
    checkbox_background_color="#ffffff",
    checkbox_background_color_dark="#ffffff",
    checkbox_border_color="#000000",
    checkbox_border_color_dark="#000000",
    checkbox_border_width="2px",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_dark="#ffffff",
    checkbox_label_text_color="#000000",
    checkbox_label_text_color_dark="#000000",
)

BRUTAL_CSS = """
.gradio-container {
    background: #ffffff !important;
}

#status-bar {
    align-items: center;
    gap: 0.75rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0 !important;
}
#memory-bar {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex: 1;
}
#memory-bar p {
    font-size: 0.7em;
    letter-spacing: 0.03em;
    opacity: 0.55;
    margin: 0 !important;
}
#shutdown-btn {
    flex: 0 0 auto;
    font-size: 0.7em !important;
    padding: 0.25rem 0.6rem !important;
    opacity: 0.6;
}
#shutdown-btn:hover {
    opacity: 1;
}
#credit-line {
    margin-top: -18px !important;
    margin-bottom: 1rem !important;
}
#credit-line p {
    font-size: 0.7em;
    letter-spacing: 0.03em;
    opacity: 0.55;
    margin: 0 !important;
}
#credit-line a {
    color: #000000 !important;
}

.tab-nav, [role="tablist"] {
    border-bottom: 2px solid #000000 !important;
    gap: 0 !important;
}
.tab-nav button, [role="tablist"] button, [role="tab"] {
    border: 2px solid #000000 !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700 !important;
    background: #ffffff !important;
    color: #000000 !important;
    transition: none !important;
    margin-right: -2px;
}
.tab-nav button.selected, [role="tablist"] button.selected,
[role="tab"][aria-selected="true"] {
    background: #000000 !important;
    color: #ffffff !important;
}

button {
    transition: none !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700 !important;
}

.brutal-box,
.brutal-box .styler,
.brutal-box .form {
    background: transparent !important;
}
.brutal-box {
    border: 2px solid #000000 !important;
    border-radius: 0 !important;
    padding: 0.75rem !important;
    margin-bottom: 0.75rem !important;
}

/* Keep all 4 preset buttons on one line instead of wrapping to a second
   row — Gradio's default Row wraps when buttons' natural min-width doesn't
   fit four across a narrow column. */
.preset-row {
    flex-wrap: nowrap !important;
}
.preset-row button {
    min-width: 0 !important;
    padding: 0.5rem 0.3rem !important;
    font-size: 0.85em !important;
}

.quiet-box,
.quiet-box .styler,
.quiet-box .form {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    /* No margin-bottom here — the parent Column already applies a 28px flex
       `gap` between top-level children (Gradio's own layout spacing); adding
       margin on top of that stacked redundantly and made gaps between
       groups noticeably bigger than gaps between rows within a group. */
    margin-bottom: 0 !important;
}

/* Combined with .quiet-box (elem_classes=["quiet-box", "thin-box"]) on the
   Shape/Texture setting groups: inherits all of quiet-box's per-control
   normalization (Number/Slider/button/dropdown styling below) but restores
   a single outer border — placed after quiet-box's own `border: none
   !important` above so this wins the cascade despite equal specificity
   (both single-class selectors). A standalone .thin-box class (without also
   using quiet-box's nested-control rules) would leave inputs like Seed at
   Gradio's raw default size/border, which reads as a second nested box. */
.thin-box {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    padding: 0.75rem !important;
    /* The parent Column already applies a 28px flex `gap` between top-level
       children (see quiet-box's own comment above) — stacking our own
       margin-bottom on top of that made the gap between the Shape and
       Texture boxes noticeably bigger than intended. */
    margin-bottom: 0 !important;
}
/* A single gr.Group(elem_classes=[..., "thin-box"]) renders as TWO nested
   divs that both receive the class (an outer wrapper + an inner .gr-group),
   so the border above was drawn twice — a box inside a box. Neutralize the
   inner one entirely so only the outermost draws the single border. */
.thin-box .thin-box {
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* Buttons inside a bordered-less Group lose their own border too (Gradio
   merges grouped elements to look seamless with the group's outer border) —
   restore it so they still read as clickable, same as every other button.
   Excludes the slider's tiny reset-icon button: it's not a real "button" in
   the same sense, and a 2px box around a small icon reads as heavy/odd next
   to its unbordered counterpart in the Upscale accordion (.brutal-box never
   strips borders in the first place, so that one was never affected). */
.quiet-box button:not(.reset-button) {
    border-width: 2px !important;
    border-style: solid !important;
    border-color: #000000 !important;
}
/* Normalize every settings-row control to the same visual language. Left as
   Gradio's defaults, some controls got a full 2px box (dropdown) and others
   only a 1px border around one small inner piece (the checkbox's own
   square) — different widths and thicknesses side by side reads as
   inconsistent. One thin rule per row, uniformly, replaces all of that.
   Deliberately NOT touching .tab-like-container (the slider value box) or
   standalone Number inputs (e.g. Seed) — the Upscale accordion (.brutal-box)
   never strips those, and its natural 1px solid box is the look we want
   everywhere, so it's left alone here and matched explicitly below instead
   of being stripped-then-restored. */
.quiet-box .wrap,
.quiet-box input:not([type="checkbox"]):not([type="radio"]):not([type="number"]) {
    border: none !important;
}
/* ...except Dropdowns: without their own box they read as plain text with
   an arrow, not a clickable selector. role="combobox" is Gradio's stable
   marker for a Dropdown's input, unlike its Svelte-hashed class names. */
.quiet-box .wrap:has(input[role="combobox"]) {
    border: 2px solid #000000 !important;
    border-radius: 0 !important;
}
/* Standalone Number fields (e.g. Seed) have no .tab-like-container wrapper
   of their own, so match that same natural 1px box directly on the input.
   Gradio's default Number styling (padding: 14px, font-size: 14px — sized
   for a normal full-width field) is what actually made this box roughly 2x
   taller than a slider's value box; matching the slider input's own compact
   padding/font-size here is what brings the heights in line, not the width
   constraint alone. */
.quiet-box input[type="number"]:not([data-testid="number-input"]) {
    border: 1px solid #000000 !important;
    border-radius: 0 !important;
    width: 105px !important;
    max-width: 105px !important;
    flex-shrink: 0 !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
}
/* Slider value boxes auto-size to their content + reset button, so Target
   faces (4 digits) and Shape Steps (2 digits) end up visibly different
   widths — force one shared width, matching Seed's, across all of them. */
.quiet-box .tab-like-container {
    width: 105px !important;
    flex-shrink: 0 !important;
}
/* ...and lay its label out the same way a slider's "head" row does (label
   text left, value box right, same line) — Gradio's own Number.svelte gives
   the input `display: block`, stacking it under the label full-width by
   default, which is what actually made it look different, not the border. */
.quiet-box label:has(> input[type="number"]:not([data-testid="number-input"])) {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: nowrap !important;
    gap: 0.5rem;
}
/* Gradio wraps consecutive un-Row'd controls (checkboxes, sliders, Number)
   in an extra <div class="styler"><div class="form"> layer, so the actual
   .block elements sit two levels below .quiet-box, not directly under it —
   a plain ".quiet-box > .block" selector silently matches nothing here.
   Scoped to "> .styler > .form" specifically (not just any nested .form):
   an explicit gr.Row() with two side-by-side components (e.g. Shape Model /
   Paint Model) also produces a .row > .form containing both as sibling
   .blocks — matching that too would give each column its own partial-width
   border instead of the one full-width border the .row rule below already
   provides, and :last-child would then strip it from the *right* column
   (last in DOM order) rather than the row as a whole. */
.quiet-box .row:not(:has(button)),
.quiet-box > .styler > .form > .block {
    border-bottom: 1px solid #000000 !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 0.6rem !important;
}
/* Simplify checkbox and the Target-faces slider directly below it read as
   one control (the slider only matters when the checkbox is on), so drop
   the divider between them specifically. */
#simplify-checkbox, #simplify-checkbox-t2 {
    border-bottom: none !important;
    padding-bottom: 0 !important;
}
.quiet-box .row:not(:has(button)):last-child,
.quiet-box > .styler > .form > .block:last-child {
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}
/* Two components sharing one gr.Row() (e.g. Shape Model / Paint Model,
   Paint Backend / Volume Decoder) land as sibling .blocks inside one shared
   .form — give them a vertical divider between columns, matching the
   horizontal ones between rows. */
.quiet-box .row .form > .block:not(:last-child) {
    border-right: 1px solid #000000 !important;
    padding-right: 1rem !important;
}
.quiet-box .row .form > .block:not(:first-child) {
    padding-left: 1rem !important;
}

/* .thin-box groups (Shape / Texture) carry their own 1px outer border, so
   the per-row divider lines that .quiet-box draws between controls become a
   redundant second set of inner borders. Strip every internal divider
   (horizontal row lines + vertical column lines) inside a .thin-box so it
   reads as one clean bordered box, keeping only the divider spacing so
   controls don't jam together. */
.thin-box .row:not(:has(button)),
.thin-box > .styler > .form > .block,
.thin-box .row .form > .block {
    border-bottom: none !important;
    border-right: none !important;
}

/* Extra breathing room between "Randomize seed" and the Re-texture button
   right below it — they otherwise sit at the same tight spacing as every
   other control row. */
#retexture-btn, #retexture-btn-t2 {
    margin-top: 1rem !important;
}

.label-wrap, .accordion .label-wrap span {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700 !important;
}

input[type="range"] {
    accent-color: #000000;
}
input[type="range"]::-webkit-slider-thumb {
    border-radius: 0 !important;
    background: #000000 !important;
}
input[type="checkbox"] {
    accent-color: #000000;
}

/* Unavailable-in-this-mode controls: no gray dimming, strike the label
   instead and keep the rest of the styling exactly as the enabled state. */
button:disabled {
    opacity: 1 !important;
    background: #ffffff !important;
    color: #000000 !important;
    border-color: #000000 !important;
    text-decoration: line-through;
    cursor: not-allowed;
}
input:disabled {
    opacity: 1 !important;
}
.block:has(input:disabled) label,
.block:has(input:disabled) .label-text,
.block:has(input:disabled) span {
    text-decoration: line-through;
    opacity: 1 !important;
    color: #000000 !important;
}

#output-viewer-1, #output-viewer-2 {
    border: 3px solid #000000 !important;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: #ffffff;
}
::-webkit-scrollbar-thumb {
    background: #000000;
    border-radius: 0;
}
"""

with gr.Blocks(title="Ifrit3D MLX") as demo:
    with gr.Row(elem_id="status-bar"):
        memory_label = gr.Markdown(_get_memory_stats(), elem_id="memory-bar")
        shutdown_btn = gr.Button("Shutdown Server", size="sm", variant="secondary", elem_id="shutdown-btn")
    gr.Markdown("Ifrit3D-MLX — Anton Shlyonkin ([shlyonk.in](https://www.shlyonk.in))", elem_id="credit-line")
    shutdown_btn.click(fn=_shutdown_server, outputs=memory_label)

    with gr.Tab("[ IMAGE TO 3D ]"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath",
                    label="Input Image",
                    height=300,
                )
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Row(elem_classes="preset-row"):
                    btn_lowpoly = gr.Button("Lowpoly", size="sm")
                    btn_draft = gr.Button("Draft", size="sm")
                    btn_normal = gr.Button("Normal", size="sm")
                    btn_high = gr.Button("High", size="sm")

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    seed = gr.Number(
                        value=12345,
                        label="Seed",
                        precision=0,
                        minimum=0,
                        maximum=999999,
                    )
                    simplify_before = gr.Checkbox(
                        value=True,
                        label="Simplify mesh before texturing",
                        info="Simplifies the shape mesh before the texturing pass, so UVs are generated for the low-poly geometry",
                        elem_id="simplify-checkbox",
                    )
                    target_faces = gr.Slider(
                        minimum=100,
                        maximum=200000,
                        value=100000,
                        step=100,
                        label="Target faces",
                    )
                    shape_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        label="Shape Steps",
                        interactive=(SHAPE_BACKEND_DEFAULT != "swift"),
                        info=(
                            "Ignored for Swift — fixed at 8 steps (consistency-distilled checkpoint)"
                            if SHAPE_BACKEND_DEFAULT == "swift" else None
                        ),
                    )
                    shape_octree_resolution = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Octree Resolution",
                    )

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    use_delight = gr.Checkbox(
                        value=False,
                        label="Remove lighting (Delight)",
                        info="Pre-processes the input image to remove lighting before texturing, producing more neutral textures",
                    )
                    skip_texturing = gr.Checkbox(
                        value=False,
                        label="Skip texturing (shape only)",
                        info="Generate the 3D shape mesh without texturing — useful for testing or external texturing",
                    )
                    texture_sd_detail = gr.Checkbox(
                        value=False,
                        label="Upscale texture",
                        info="Light generative touch-up on each view before baking",
                    )
                    randomize_seed = gr.Checkbox(value=True, label="Randomize seed")

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    paint_res = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Paint Resolution")
                    paint_steps = gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Paint Steps")
                    paint_tex = gr.Slider(minimum=256, maximum=4096, value=2048, step=256, label="Paint Texture Size")
                    retexture_btn = gr.Button("Re-texture", variant="primary", size="lg", elem_id="retexture-btn")
                    # Hidden — no longer user-adjustable, but still preset-dependent (paint_guidance)
                    # or fixed across all presets (the rest), so kept as plumbing rather than
                    # hardcoded, since presets set these via the same outputs mechanism.
                    use_swift_paint = gr.State(value=True)
                    paint_guidance = gr.State(value=1.5)
                    paint_superres = gr.State(value=False)
                    paint_sd_strength = gr.State(value=0.3)
                    paint_sd_res = gr.State(value=512)

            with gr.Column(scale=1):
                current_mesh = gr.State()
                output_3d = gr.Model3D(label="Result", height=600, elem_id="output-viewer-1")
                output_file = gr.File(label="Download .glb")
                output_obj = gr.File(label="Download .obj (zipped)")

        _preset_outputs = [
            shape_steps, shape_octree_resolution, use_delight, simplify_before,
            target_faces, skip_texturing, texture_sd_detail,
            use_swift_paint, paint_res, paint_steps, paint_guidance, paint_tex,
            paint_superres, paint_sd_strength, paint_sd_res,
        ]
        btn_lowpoly.click(fn=lambda: _set_preset("lowpoly"), outputs=_preset_outputs)
        btn_draft.click(fn=lambda: _set_preset("draft"), outputs=_preset_outputs)
        btn_normal.click(fn=lambda: _set_preset("normal"), outputs=_preset_outputs)
        btn_high.click(fn=lambda: _set_preset("high"), outputs=_preset_outputs)

        generate_btn.click(
            fn=generate,
            inputs=[
                image_input,
                use_delight,
                seed,
                shape_steps,
                shape_octree_resolution,
                simplify_before,
                target_faces,
                skip_texturing,
                texture_sd_detail,
                use_swift_paint,
                paint_res,
                paint_steps,
                paint_guidance,
                paint_tex,
                paint_superres,
                paint_sd_strength,
                paint_sd_res,
            ],
            outputs=[output_3d, output_file, output_obj, current_mesh],
        ).then(fn=_get_memory_stats, outputs=memory_label)

        def _retexture_current(
            glb_path, use_delight,
            seed, randomize_seed,
            texture_sd_detail=False,
            use_swift_paint=False,
            paint_res=512, paint_steps=15, paint_guidance=2.0, paint_tex=2048,
            paint_superres=False, paint_sd_strength=0.3, paint_sd_res=768,
            progress=gr.Progress(),
        ):
            global _last_original_inputs
            if not _last_original_inputs:
                raise gr.Error("Generate a model first!")
            return _run_retexture(
                glb_path, _last_original_inputs, use_delight,
                seed, randomize_seed,
                texture_sd_detail,
                use_swift_paint, paint_res, paint_steps, paint_guidance, paint_tex,
                paint_superres, paint_sd_strength, paint_sd_res,
                progress=progress,
            )

        retexture_btn.click(
            fn=_retexture_current,
            inputs=[
                current_mesh, use_delight,
                seed, randomize_seed,
                texture_sd_detail,
                use_swift_paint, paint_res, paint_steps, paint_guidance, paint_tex,
                paint_superres, paint_sd_strength, paint_sd_res,
            ],
            outputs=[output_3d, output_file, output_obj, current_mesh],
        ).then(fn=_get_memory_stats, outputs=memory_label)

    with gr.Tab("[ TEXT TO 3D ]"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the object you want to generate 3D from...",
                    lines=3,
                    value="fullbody isolated on white",
                )
                sd_steps = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="SD Inference Steps",
                    info="SD Turbo works best with 1-4 steps",
                )
                gen_image_btn = gr.Button("Generate Image", variant="primary", size="lg")

                sd_output = gr.Image(
                    type="pil",
                    label="Generated Image",
                    height=300,
                )
                gen_3d_btn = gr.Button("Generate 3D from Image", variant="primary", size="lg")

                with gr.Row(elem_classes="preset-row"):
                    btn_lowpoly_t2 = gr.Button("Lowpoly", size="sm")
                    btn_draft_t2 = gr.Button("Draft", size="sm")
                    btn_normal_t2 = gr.Button("Normal", size="sm")
                    btn_high_t2 = gr.Button("High", size="sm")

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    seed_t2 = gr.Number(
                        value=12345,
                        label="Seed",
                        precision=0,
                        minimum=0,
                        maximum=999999,
                    )
                    simplify_before_t2 = gr.Checkbox(
                        value=True,
                        label="Simplify mesh before texturing",
                        info="Simplifies the shape mesh before the texturing pass, so UVs are generated for the low-poly geometry",
                        elem_id="simplify-checkbox-t2",
                    )
                    target_faces_t2 = gr.Slider(
                        minimum=100,
                        maximum=200000,
                        value=100000,
                        step=100,
                        label="Target faces",
                    )
                    shape_steps_t2 = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        label="Shape Steps",
                        interactive=(SHAPE_BACKEND_DEFAULT != "swift"),
                        info=(
                            "Ignored for Swift — fixed at 8 steps (consistency-distilled checkpoint)"
                            if SHAPE_BACKEND_DEFAULT == "swift" else None
                        ),
                    )
                    shape_octree_resolution_t2 = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Octree Resolution",
                    )

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    use_delight_t2 = gr.Checkbox(
                        value=False,
                        label="Remove lighting (Delight)",
                        info="Pre-processes the input image to remove lighting before texturing, producing more neutral textures",
                    )
                    skip_texturing_t2 = gr.Checkbox(
                        value=False,
                        label="Skip texturing (shape only)",
                        info="Generate the 3D shape mesh without texturing — useful for testing or external texturing",
                    )
                    texture_sd_detail_t2 = gr.Checkbox(
                        value=False,
                        label="Upscale texture",
                        info="Light generative touch-up on each view before baking",
                    )
                    randomize_seed_t2 = gr.Checkbox(value=True, label="Randomize seed")

                with gr.Group(elem_classes=["quiet-box", "thin-box"]):
                    paint_res_t2 = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Paint Resolution")
                    paint_steps_t2 = gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Paint Steps")
                    paint_tex_t2 = gr.Slider(minimum=256, maximum=4096, value=2048, step=256, label="Paint Texture Size")
                    retexture_btn_t2 = gr.Button("Re-texture", variant="primary", size="lg", elem_id="retexture-btn-t2")
                    use_swift_paint_t2 = gr.State(value=True)
                    paint_guidance_t2 = gr.State(value=1.5)
                    paint_superres_t2 = gr.State(value=False)
                    paint_sd_strength_t2 = gr.State(value=0.3)
                    paint_sd_res_t2 = gr.State(value=512)

            with gr.Column(scale=1):
                current_mesh_t2 = gr.State()
                output_3d_t2 = gr.Model3D(label="Result", height=600, elem_id="output-viewer-2")
                output_file_t2 = gr.File(label="Download .glb")
                output_obj_t2 = gr.File(label="Download .obj (zipped)")

        gen_image_btn.click(
            fn=generate_image,
            inputs=[prompt_input, sd_steps],
            outputs=[sd_output],
        ).then(fn=_get_memory_stats, outputs=memory_label)

        def generate_from_prompt_image(
            sd_output_img,
            use_delight, seed, shape_steps, shape_octree_resolution,
            simplify_before_texturing,
            target_faces, skip_texturing=False,
            texture_sd_detail=False,
            use_swift_paint=False,
            paint_res=512, paint_steps=15, paint_guidance=2.0, paint_tex=2048,
            paint_superres=False, paint_sd_strength=0.3, paint_sd_res=768,
            shape_backend=SHAPE_BACKEND_DEFAULT,
            progress=gr.Progress()
        ):
            if sd_output_img is None:
                raise gr.Error("Generate an image first!")
            img_path = OUTPUT_DIR / "_input.png"
            sd_output_img.save(img_path, "PNG")
            glb_path, glb_file, obj_zip, mesh_state = generate(
                str(img_path),
                use_delight, seed, shape_steps, shape_octree_resolution,
                simplify_before_texturing, target_faces, skip_texturing,
                texture_sd_detail,
                use_swift_paint, paint_res, paint_steps, paint_guidance, paint_tex,
                paint_superres, paint_sd_strength, paint_sd_res,
                shape_backend=shape_backend, progress=progress,
            )
            return glb_path, glb_file, obj_zip, mesh_state

        gen_3d_btn.click(
            fn=generate_from_prompt_image,
            inputs=[
                sd_output,
                use_delight_t2,
                seed_t2,
                shape_steps_t2,
                shape_octree_resolution_t2,
                simplify_before_t2,
                target_faces_t2,
                skip_texturing_t2,
                texture_sd_detail_t2,
                use_swift_paint_t2,
                paint_res_t2,
                paint_steps_t2,
                paint_guidance_t2,
                paint_tex_t2,
                paint_superres_t2,
                paint_sd_strength_t2,
                paint_sd_res_t2,
            ],
            outputs=[output_3d_t2, output_file_t2, output_obj_t2, current_mesh_t2],
        ).then(fn=_get_memory_stats, outputs=memory_label)

        def _retexture_current_t2(
            glb_path, use_delight,
            seed, randomize_seed,
            texture_sd_detail=False,
            use_swift_paint=False,
            paint_res=512, paint_steps=15, paint_guidance=2.0, paint_tex=2048,
            paint_superres=False, paint_sd_strength=0.3, paint_sd_res=768,
            progress=gr.Progress(),
        ):
            global _last_original_inputs
            if not _last_original_inputs:
                raise gr.Error("Generate a model first!")
            return _run_retexture(
                glb_path, _last_original_inputs, use_delight,
                seed, randomize_seed,
                texture_sd_detail,
                use_swift_paint, paint_res, paint_steps, paint_guidance, paint_tex,
                paint_superres, paint_sd_strength, paint_sd_res,
                progress=progress,
            )

        retexture_btn_t2.click(
            fn=_retexture_current_t2,
            inputs=[
                current_mesh_t2, use_delight_t2,
                seed_t2, randomize_seed_t2,
                texture_sd_detail_t2,
                use_swift_paint_t2, paint_res_t2, paint_steps_t2, paint_guidance_t2, paint_tex_t2,
                paint_superres_t2, paint_sd_strength_t2, paint_sd_res_t2,
            ],
            outputs=[output_3d_t2, output_file_t2, output_obj_t2, current_mesh_t2],
        ).then(fn=_get_memory_stats, outputs=memory_label)

        _preset_outputs_t2 = [
            shape_steps_t2, shape_octree_resolution_t2, use_delight_t2, simplify_before_t2,
            target_faces_t2, skip_texturing_t2, texture_sd_detail_t2,
            use_swift_paint_t2, paint_res_t2, paint_steps_t2, paint_guidance_t2, paint_tex_t2,
            paint_superres_t2, paint_sd_strength_t2, paint_sd_res_t2,
        ]
        btn_lowpoly_t2.click(fn=lambda: _set_preset("lowpoly"), outputs=_preset_outputs_t2)
        btn_draft_t2.click(fn=lambda: _set_preset("draft"), outputs=_preset_outputs_t2)
        btn_normal_t2.click(fn=lambda: _set_preset("normal"), outputs=_preset_outputs_t2)
        btn_high_t2.click(fn=lambda: _set_preset("high"), outputs=_preset_outputs_t2)

    demo.queue()


if __name__ == "__main__":
    port = 7860
    while port < 7870:
        try:
            demo.launch(server_name="127.0.0.1", server_port=port, theme=BRUTAL_THEME, css=BRUTAL_CSS, inbrowser=True)
            break
        except OSError:
            port += 1
