import numpy as np
import trimesh
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

mesh = trimesh.load('outputs/demo_shape_mps.glb', force='mesh')
r = MeshRender(default_resolution=512, texture_size=512, device='mps', raster_mode='mtl')
r.load_mesh(mesh)
img_n = r.render_normal(0, 0, return_type='np')
img_p = r.render_position(0, 0, return_type='np')
print('normal', np.nanmin(img_n), np.nanmax(img_n), np.nanmean(img_n))
print('pos', np.nanmin(img_p), np.nanmax(img_p), np.nanmean(img_p))
print('normal nonwhite ratio', np.mean(np.any(img_n < 0.99, axis=-1)))
print('pos nonwhite ratio', np.mean(np.any(img_p < 0.99, axis=-1)))
