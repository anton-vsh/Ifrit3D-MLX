from pygltflib import GLTF2
from PIL import Image
import io
import numpy as np
import sys

p = sys.argv[1] if len(sys.argv) > 1 else 'outputs/demo_textured_mps.glb'
g = GLTF2().load(p)
blob = g.binary_blob()
print('images:', len(g.images or []))
for i, img in enumerate(g.images or []):
    if img.bufferView is None:
        print(i, 'no bufferView')
        continue
    bv = g.bufferViews[img.bufferView]
    data = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
    im = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(im)
    print(i, im.size, 'min', int(arr.min()), 'max', int(arr.max()), 'mean', float(arr.mean()), 'std', float(arr.std()))
    out = f'outputs/extracted_tex_{i}.png'
    im.save(out)
    print('saved', out)
