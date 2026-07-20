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


import logging
import numpy as np
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List, Union, Optional


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.realesrgan_utils import RealESRGAN_Upscaler
from .utils.sdturbo_upscale_utils import SDTurboUpscaler
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


def _scaled_progress(progress_callback, lo, hi):
    if progress_callback is None:
        return None

    def _cb(fraction, desc=None):
        progress_callback(lo + fraction * (hi - lo), desc)

    return _cb


class Hunyuan3DTexGenConfig:

    def __init__(
        self,
        light_remover_ckpt_path,
        multiview_ckpt_path,
        subfolder_name,
        diffusion_backend='pytorch',
        mlx_weights_path=None,
        mlx_profile=None,
        pbr_albedo_only=False,
    ):
        # Prefer CUDA, then MPS, then CPU.
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.diffusion_backend = diffusion_backend
        self.mlx_weights_path = mlx_weights_path
        self.mlx_profile = mlx_profile
        # Skip "mr" (metallic-roughness) generation for the PBR (2.1) preset —
        # mr is discarded before export today anyway (see multiview_utils.py),
        # so this is a pure compute-saving switch, not a change to output.
        self.pbr_albedo_only = pbr_albedo_only

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 1024
        self.texture_size = 1024
        # Two independent, chainable texture-detail passes applied to each
        # generated view before baking (see __call__): SD Turbo first (a
        # light, ~2-real-step generative touch-up — full-strength/step-count
        # regeneration was tested and made the eye-asymmetry problem worse
        # and the output blurrier, not better), then RealESRGAN (a cheap
        # upscale-then-downsample-back-to-1024 pass used purely as a
        # sharpen/cleanup filter, not for resolution gain — both target the
        # same 1024 texture_size, not the 2048 tested earlier, since that
        # alone didn't look meaningfully better and cost more).
        # Read fresh per-call in __call__ (like HY3D_USE_DELIGHT), not baked
        # in here: self.config is part of a pipeline cached and reused across
        # generations (main.py's _PAINT_CACHE_PIPELINE) whenever the model/
        # backend selection doesn't change, so a flag set once at construction
        # would silently ignore every later checkbox toggle.
        # Higher exponent biases the per-texel weighted blend more strongly
        # toward whichever view has the highest cosine weight there (usually
        # the front view). A low exponent lets a low-weight side/profile view
        # still blend in proportionally wherever it has any coverage, which
        # bleeds that view's (often lower-detail, since profile angles can't
        # resolve fine features well) content into regions the front view
        # already covers well — visible as color/shape drift on bilateral
        # features like eyes even when the front-view generation was correct
        # and symmetric on its own.
        self.bake_exp = 8
        self.merge_method = 'fast'

        self.pipe_dict = {
            'hunyuan3d-paint-v2-0': 'hunyuanpaint',
            'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo',
            'hunyuan3d-paint-v2-1': 'hunyuanpaintpbr',
            'hunyuan3d-paintpbr-v2-1': 'hunyuanpaintpbr',
            'hunyuan3d-paint-v2-1-turbo': 'hunyuanpaintpbr',
        }
        self.pipe_name = self.pipe_dict.get(subfolder_name, 'hunyuanpaint-turbo' if 'turbo' in subfolder_name else 'hunyuanpaint')


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        subfolder='hunyuan3d-paint-v2-0-turbo',
        diffusion_backend='pytorch',
        mlx_weights_path=None,
        pbr_albedo_only=False,
    ):
        def _infer_mlx_profile(subfolder_name: str) -> str:
            if 'paintpbr' in subfolder_name or 'v2-1' in subfolder_name:
                return 'paint-pbr-2.1'
            return 'paint-2.0'

        def _mlx_hf_source(profile: str, subfolder_name: str):
            if profile == 'paint-2.0':
                repo_id = 'zimengxiong/Hunyuan3D-2.0-Paint-MLX'
                subdir = '2.0-turbo' if 'turbo' in subfolder_name else '2.0'
            else:
                repo_id = 'zimengxiong/Hunyuan3D-2.1-Paint-MLX'
                subdir = None
            return repo_id, subdir

        def _default_mlx_weights_path(multiview_model_path: str, profile: str, subfolder_name: str):
            parent = os.path.dirname(multiview_model_path)
            if profile == 'paint-2.0':
                local_subdir = '2.0-turbo' if 'turbo' in subfolder_name else '2.0'
                candidates = [
                    os.path.join(parent, 'hunyuan3d-2.0-mlx', local_subdir),
                    os.path.join(multiview_model_path, 'mlx_weights'),
                ]
            else:
                candidates = [
                    os.path.join(parent, 'hunyuan3d-2.1-mlx'),
                    os.path.join(multiview_model_path, 'mlx_weights'),
                ]
            for c in candidates:
                if os.path.exists(os.path.join(c, 'unet.npz')):
                    return c
            return candidates[0]

        def _download_mlx_weights_from_hf(profile: str, subfolder_name: str) -> str:
            import huggingface_hub

            repo_id, subdir = _mlx_hf_source(profile, subfolder_name)
            if subdir:
                snapshot = huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f'{subdir}/*'],
                )
                return os.path.join(snapshot, subdir)

            snapshot = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                allow_patterns=['*.npz', 'README.md'],
            )
            return snapshot

        original_model_path = model_path
        delight_model_path = None
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            multiview_model_path = os.path.join(model_path, subfolder)
            candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            if os.path.exists(candidate_delight_model_path):
                delight_model_path = candidate_delight_model_path

            if not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[f'{subfolder}/*']
                    )
                    multiview_model_path = os.path.join(model_path, subfolder)
                    candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    if os.path.exists(candidate_delight_model_path):
                        delight_model_path = candidate_delight_model_path
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
        else:
            candidate_delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            if os.path.exists(candidate_delight_model_path):
                delight_model_path = candidate_delight_model_path
            multiview_model_path = os.path.join(model_path, subfolder)

        mlx_profile = _infer_mlx_profile(subfolder)
        resolved_mlx_weights_path = mlx_weights_path
        if diffusion_backend == 'mlx' and resolved_mlx_weights_path is None:
            resolved_mlx_weights_path = _default_mlx_weights_path(multiview_model_path, mlx_profile, subfolder)

            from .mlx.convert_weights import is_mlx_weights_current, convert_and_save

            if not is_mlx_weights_current(resolved_mlx_weights_path):
                # Missing entirely, or stamped from an older conversion
                # (e.g. the v1->v2 fix for a stale-checkpoint bug: the
                # converter used to read diffusion_pytorch_model.bin
                # unconditionally, which for hunyuan3d-paint-v2-0-turbo is
                # an older training snapshot than the .safetensors file
                # diffusers actually loads). multiview_model_path already
                # holds (or was just downloaded to hold) the real PyTorch
                # checkpoint at this point regardless of backend, so
                # converting locally is both correct-by-construction (same
                # source diffusers itself will load) and avoids depending
                # on a third-party pre-converted upload staying in sync
                # with this converter's own fixes. Only fall back to the
                # HF-hosted pre-converted weights if local conversion
                # itself can't run (e.g. torch/mlx not importable here).
                try:
                    resolved_mlx_weights_path, _ = convert_and_save(
                        multiview_model_path,
                        output_dir=resolved_mlx_weights_path,
                        profile=mlx_profile,
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
                    resolved_mlx_weights_path = _download_mlx_weights_from_hf(mlx_profile, subfolder)

        return cls(
            Hunyuan3DTexGenConfig(
                delight_model_path,
                multiview_model_path,
                subfolder,
                diffusion_backend=diffusion_backend,
                mlx_weights_path=resolved_mlx_weights_path,
                mlx_profile=mlx_profile,
                pbr_albedo_only=pbr_albedo_only,
            )
        )
            
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            device=self.config.device,
            raster_mode='auto')

        self.load_models()

    def load_models(self):
        # Empty CUDA cache only when CUDA is active.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # The delight model is a full InstructPix2Pix SD pipeline (multi-GB) and
        # is disabled by default (HY3D_USE_DELIGHT=0), so defer constructing it
        # until it's actually needed instead of loading it on every startup.
        self.models['delight_model'] = None
        self._delight_available = bool(
            self.config.light_remover_ckpt_path
            and os.path.exists(self.config.light_remover_ckpt_path)
        )
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # Both opt-in and kept lazy like the delight model rather than loaded
        # on every startup.
        self.models['sd_upscale_model'] = None
        self.models['esrgan_upscale_model'] = None

    def _get_delight_model(self):
        if self.models['delight_model'] is None and self._delight_available:
            self.models['delight_model'] = Light_Shadow_Remover(self.config)
        return self.models['delight_model']

    def _get_sd_upscale_model(self, progress_callback=None):
        if self.models['sd_upscale_model'] is None:
            self.models['sd_upscale_model'] = SDTurboUpscaler(self.config, progress_callback=progress_callback)
        return self.models['sd_upscale_model']

    def _get_esrgan_upscale_model(self, progress_callback=None):
        if self.models['esrgan_upscale_model'] is None:
            self.models['esrgan_upscale_model'] = RealESRGAN_Upscaler(self.config, progress_callback=progress_callback)
        return self.models['esrgan_upscale_model']

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        if self.models['delight_model'] is not None:
            self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True, resolution=None):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, resolution=resolution, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims, resolution=None):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, resolution=resolution, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='fast'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def _despeckle_texture(self, texture):
        # Baking 512px diffusion views into a larger atlas leaves texels with
        # near-zero sample weight that pass the bake's trust threshold yet
        # carry no real color — they render as isolated flecks. Two variants,
        # both replaced with the local 5x5 median color: dark flecks (near-
        # black vs. local luminance median; verified 94% reduction on a real
        # afflicted texture, smooth/detailed regions untouched) and chromatic
        # flecks (a wrong-hued speck that isn't necessarily darker — e.g.
        # blue/teal specks on an otherwise white/cream surface — which the
        # dark-only check misses entirely). Median filtering is edge-
        # preserving for genuine color boundaries (a pixel right at a real
        # edge still matches its own side's local-majority color, so the
        # diff stays small there), so this cleans isolated outlier texels
        # without blurring real detail.
        from scipy.ndimage import median_filter

        arr = (texture.cpu().numpy() * 255.0).astype(np.float32)
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
            return texture
        arr[fleck_mask] = med_rgb[fleck_mask]
        logger.debug(
            f"Despeckle: replaced {n} fleck texels "
            f"({int(dark_mask.sum())} dark, {int(chroma_mask.sum())} chromatic)"
        )
        return torch.tensor(arr / 255.0).float().to(texture.device)

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image, seed=0, progress_callback=None):

        if progress_callback is not None:
            progress_callback(0.0, "Preparing reference images...")

        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)

        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        # Delight preprocessing runs the InstructPix2Pix model on CPU/float32 on
        # MPS (see Light_Shadow_Remover) to avoid fp16 instability there; on other
        # devices it uses the configured device/dtype. Still opt-in by default.
        use_delight = (
            os.environ.get('HY3D_USE_DELIGHT', '0') == '1'
            and self._delight_available
        )
        if use_delight:
            if progress_callback is not None:
                progress_callback(0.03, "Delight preprocessing...")
            delight_model = self._get_delight_model()
            logger.debug("Delight preprocessing...")
            images_prompt = [delight_model(image_prompt) for image_prompt in images_prompt]
            logger.debug("Delight preprocessing done")

        overlap_uv_unwrap = os.environ.get('HY3D_OVERLAP_UV_UNWRAP', '0') == '1'

        if progress_callback is not None:
            progress_callback(0.08, "UV wrapping...")
        logger.debug("UV wrapping...")
        if overlap_uv_unwrap:
            # Fast path: overlap UV unwrapping on CPU with multiview diffusion on GPU.
            # This improves throughput substantially, but can introduce small texture
            # differences because control views are rendered before UV seam splits.
            raw_mesh = mesh.copy() if hasattr(mesh, 'copy') else mesh
            self.render.load_mesh(raw_mesh)
        else:
            mesh = mesh_uv_wrap(mesh)
            self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        # The multiview diffusion model consumes control images at its own
        # native view_size (512) regardless of render_size, so render them
        # directly at that resolution instead of at render_size (up to 2048)
        # only to have the model immediately downsample them back down.
        view_size = self.models['multiview_model'].view_size
        if progress_callback is not None:
            progress_callback(0.12, "Rendering normal maps...")
        logger.debug("Rendering normal maps...")
        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True, resolution=view_size)
        if progress_callback is not None:
            progress_callback(0.16, "Rendering position maps...")
        logger.debug("Rendering position maps...")
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims, resolution=view_size)

        uv_future = None
        mesh_executor = None
        uv_ready_mesh = mesh
        if overlap_uv_unwrap:
            if hasattr(raw_mesh.visual, 'uv') and raw_mesh.visual.uv is not None:
                uv_ready_mesh = raw_mesh
            else:
                mesh_executor = ThreadPoolExecutor(max_workers=1)
                uv_future = mesh_executor.submit(mesh_uv_wrap, mesh.copy() if hasattr(mesh, 'copy') else mesh)
                uv_ready_mesh = None

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        if progress_callback is not None:
            progress_callback(0.20, "Starting multiview diffusion...")
        multiviews = self.models['multiview_model'](
            images_prompt, normal_maps + position_maps, camera_info, seed=seed,
            progress_callback=_scaled_progress(progress_callback, 0.20, 0.85),
        )

        if uv_future is not None:
            uv_ready_mesh = uv_future.result()
            mesh_executor.shutdown(wait=False)
            self.render.load_mesh(uv_ready_mesh)

        # multiviews are native view_size (512) from the diffusion model —
        # the real content ceiling for texture detail, since Texture Size only
        # resamples this fixed content into the UV atlas. Two optional,
        # chainable passes here before baking can add real detail rather than
        # just resampling it: ESRGAN (cheap sharpen) runs first, then SD Turbo
        # (light generative touch-up) — A/B tested against the reverse order;
        # this order gave SD Turbo's low-strength pass a cleaner starting
        # point and came out visibly sharper (e.g. eyebrow hair strands
        # stayed distinct instead of smudging).
        if os.environ.get('HY3D_USE_ESRGAN_UPSCALE', '0') == '1':
            if progress_callback is not None:
                progress_callback(0.85, "ESRGAN sharpen pass...")
            esrgan_model = self._get_esrgan_upscale_model(progress_callback=_scaled_progress(progress_callback, 0.85, 0.86))
            for i in range(len(multiviews)):
                multiviews[i] = esrgan_model(multiviews[i])

        if os.environ.get('HY3D_USE_SD_UPSCALE', '0') == '1':
            if progress_callback is not None:
                progress_callback(0.86, "SD Turbo detail pass...")
            sd_model = self._get_sd_upscale_model(progress_callback=_scaled_progress(progress_callback, 0.86, 0.88))
            for i in range(len(multiviews)):
                multiviews[i] = sd_model(multiviews[i])

        if progress_callback is not None:
            progress_callback(0.85, "Baking multiview -> UV texture...")
        logger.debug("Baking multiview -> UV texture...")
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        if progress_callback is not None:
            progress_callback(0.92, "Inpainting texture seams...")
        logger.debug("Inpainting texture seams...")
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.texture_inpaint(texture, mask_np)

        texture = self._despeckle_texture(texture)

        if progress_callback is not None:
            progress_callback(0.97, "Exporting mesh...")
        logger.debug("Exporting mesh...")
        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        if progress_callback is not None:
            progress_callback(1.0, "Texturing done")

        return textured_mesh
