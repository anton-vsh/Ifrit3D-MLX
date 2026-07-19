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

# RRDBNet is vendored directly (BSD-3-Clause, xinntao/Real-ESRGAN) rather than
# taken from the `realesrgan` PyPI package: that package depends on `basicsr`,
# whose degradations.py imports `torchvision.transforms.functional_tensor`, a
# submodule removed in torchvision 0.17+ — it fails to import on this
# project's torch/torchvision versions. Vendoring just the pure-nn.Module
# architecture (no basicsr import) sidesteps that entirely.

import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REALESRGAN_X4PLUS_URL = (
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Matches basicsr's RRDBNet exactly (module names/shapes) so the public
    RealESRGAN_x4plus.pth checkpoint loads with strict=True."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # x4: two nearest-neighbor 2x upsamples + conv, matching the checkpoint's
        # trained scale — always used regardless of the wrapper's output target,
        # since the input resolution (not the upsample factor) dominates cost.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def _download_checkpoint(dest_path, progress_callback=None):
    tmp_path = dest_path + '.part'
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    last_reported = -1

    def _reporthook(block_num, block_size, total_size):
        nonlocal last_reported
        if progress_callback is None or total_size <= 0:
            return
        fraction = min(1.0, block_num * block_size / total_size)
        percent = int(fraction * 100)
        if percent == last_reported:
            return
        last_reported = percent
        progress_callback(fraction, f"Downloading RealESRGAN_x4plus.pth ({percent}%)")

    urllib.request.urlretrieve(REALESRGAN_X4PLUS_URL, tmp_path, reporthook=_reporthook)
    os.replace(tmp_path, dest_path)


class RealESRGAN_Upscaler():
    """Lightweight (single forward pass, ~64MB) x4 super-resolution for the
    multiview diffusion output, applied before UV baking. Unlike the disabled
    Image_Super_Net (a full SD-based diffusion upscaler), this is a plain CNN
    with no denoising loop, attention, or known fp16/MPS instability — so it
    runs in fp32 on every device rather than needing a per-device fallback."""

    def __init__(self, config, progress_callback=None):
        self.device = config.device
        self.texture_size = config.texture_size

        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        ckpt_path = os.path.expanduser(os.path.join(base_dir, 'realesrgan', 'RealESRGAN_x4plus.pth'))
        if not os.path.exists(ckpt_path):
            _download_checkpoint(ckpt_path, progress_callback=progress_callback)

        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        state = torch.load(ckpt_path, map_location='cpu')
        state = state.get('params_ema', state.get('params', state))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert('RGB')).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray((out * 255).round().astype(np.uint8))
        # Rasterization cost in back_project scales with the input view's own
        # resolution, and the atlas is allocated at texture_size regardless —
        # so downsample the sharpened 2048px result back down before baking
        # rather than feeding the full upscale into the renderer.
        if out_img.size != (self.texture_size, self.texture_size):
            out_img = out_img.resize((self.texture_size, self.texture_size), Image.LANCZOS)
        return out_img
