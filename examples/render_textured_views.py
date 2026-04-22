import numpy as np
import trimesh
from PIL import Image
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

mesh = trimesh.load('outputs/demo_textured_mps.glb', force='mesh')
r = MeshRender(default_resolution=768, texture_size=1024, device='mps', raster_mode='mtl')
r.load_mesh(mesh)
tex = Image.open('outputs/extracted_tex_0.png').convert('RGB')
r.set_texture(tex)
for az in [0,90,180,270]:
    img = r.render(0, az, keep_alpha=False, return_type='np')
    arr = np.clip(img*255,0,255).astype(np.uint8)
    Image.fromarray(arr).save(f'outputs/view_{az}.png')
    print('saved', f'outputs/view_{az}.png', 'mean', float(arr.mean()), 'std', float(arr.std()))
