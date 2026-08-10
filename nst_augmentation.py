import sys
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

CURRENT_DIR = Path(__file__).resolve().parent    # style_detection/
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from nst.utils import calc_mean_std
from nst import adain_model

class NSTRectangularTransform(nn.Module):
    """
    Batchwise AdaIN style transfer for rectangular images.

    Works on [B, C, H, W] tensors on GPU. It can also accept a single image [C, H, W].
    The transform:
      1) randomly selects a subset of images in the batch,
      2) resizes each selected image so its short side becomes patch_size,
      3) tiles the long side into overlapping patches,
      4) stylizes all patches in one batched AdaIN pass,
      5) reconstructs the full image with smooth overlap blending.

    It also supports optional denormalize -> stylize -> renormalize behavior so the
    same class can be used both:
      - before normalization in the classic torchvision path
      - after normalization in a DALI / Lightning hook path
    """
    def __init__(
        self,
        style_feats: torch.Tensor,
        vgg: nn.Module,
        decoder: nn.Module,
        *,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        probability: float = 0.5,
        patch_size: int = 224,
        overlap: int = 32,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.vgg = vgg.eval()
        self.decoder = decoder.eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        for p in self.decoder.parameters(): p.requires_grad_(False)

        self.register_buffer("style_features", torch.as_tensor(style_feats, dtype=torch.float32), persistent=False)

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.probability = float(probability)
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)

        if mean is not None and std is not None:
            self.register_buffer("mean_buf", torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.register_buffer("std_buf", torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.use_normalization = True
        else:
            self.mean_buf = None
            self.std_buf = None
            self.use_normalization = False

    @classmethod
    def from_files(
        cls,
        *,
        style_feats_path: Union[str, Path],
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        probability: float = 0.5,
        patch_size: int = 224,
        overlap: int = 32,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> "NSTRectangularTransform":
        vgg = adain_model.vgg
        decoder = adain_model.decoder

        vgg.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        vgg = nn.Sequential(*list(vgg.children())[:31])
        style_feats_np = np.load(style_feats_path)
        style_feats = torch.from_numpy(style_feats_np).to(dtype=torch.float32)

        return cls(
            style_feats, vgg, decoder,
            alpha_min=alpha_min, alpha_max=alpha_max, probability=probability,
            patch_size=patch_size, overlap=overlap, mean=mean, std=std
        )

    def adaptive_instance_normalization(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    @torch.no_grad()
    def style_transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        alpha = torch.empty((), device=content.device, dtype=content.dtype).uniform_(self.alpha_min, self.alpha_max)
        content_f = self.vgg(content)
        feat = self.adaptive_instance_normalization(content_f, style)
        feat = feat * alpha + content_f * (1.0 - alpha)
        return self.decoder(feat)

    @staticmethod
    def _blend_mask(i: int, j: int, num_h: int, num_w: int, patch_size: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_h == 1 and num_w == 1:
            return torch.ones((3, patch_size, patch_size), device=device, dtype=dtype)
        mask_h = torch.ones((patch_size, patch_size), device=device, dtype=dtype)
        mask_w = torch.ones((patch_size, patch_size), device=device, dtype=dtype)
        if i > 0: mask_h[:overlap, :] = torch.linspace(0, 1, overlap, device=device, dtype=dtype).unsqueeze(1)
        if i < num_h - 1: mask_h[-overlap:, :] = torch.linspace(1, 0, overlap, device=device, dtype=dtype).unsqueeze(1)
        if j > 0: mask_w[:, :overlap] = torch.linspace(0, 1, overlap, device=device, dtype=dtype).unsqueeze(0)
        if j < num_w - 1: mask_w[:, -overlap:] = torch.linspace(1, 0, overlap, device=device, dtype=dtype).unsqueeze(0)
        return (mask_h * mask_w).unsqueeze(0).repeat(3, 1, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_image = x.ndim == 3
        if single_image: x = x.unsqueeze(0)

        # Dynamic Module Alignment Hook for Hidden DDP Attributes
        if self.style_features.device != x.device:
            self.to(x.device)

        x = x.clone()
        device, dtype = x.device, x.dtype
        batch_size, channels, H, W = x.shape

        if self.use_normalization:
            x = x * self.std_buf + self.mean_buf

        ratio = int(math.floor(batch_size * self.probability + torch.rand((), device=device).item()))
        if ratio <= 0:
            if self.use_normalization: x = (x - self.mean_buf) / self.std_buf
            return x.squeeze(0) if single_image else x

        img_idx = torch.randperm(batch_size, device=device)[:ratio]
        style_idx = torch.randint(0, self.style_features.shape[0], (ratio,), device=device)

        selected = x[img_idx]
        was_grayscale = selected.shape[1] == 1
        if was_grayscale: selected = selected.repeat(1, 3, 1, 1)

        patch_size = self.patch_size
        stride = patch_size - self.overlap

        metas, patches, patches_per_image = [], [], []

        for r in range(selected.shape[0]):
            img = selected[r : r + 1]
            _, _, orig_H, orig_W = img.shape

            scale = float(patch_size) / float(min(orig_H, orig_W))
            new_H, new_W = int(round(orig_H * scale)), int(round(orig_W * scale))

            img_resized = F.resize(img, size=[new_H, new_W], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
            num_h = max(1, math.ceil((new_H - self.overlap) / stride))
            num_w = max(1, math.ceil((new_W - self.overlap) / stride))
            metas.append((int(img_idx[r]), int(orig_H), int(orig_W), new_H, new_W, num_h, num_w))

            cnt = 0
            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))
                    patches.append(img_resized[:, :, top : top + patch_size, left : left + patch_size].squeeze(0))
                    cnt += 1
            patches_per_image.append(cnt)

        patches_tensor = torch.stack(patches, dim=0)
        style_batch = torch.cat([self.style_features[sid].unsqueeze(0).repeat(c, 1, 1, 1) for sid, c in zip(style_idx.tolist(), patches_per_image)], dim=0)

        stylized_patches = self.style_transfer(patches_tensor, style_batch)

        proc_ptr = 0
        for meta in metas:
            img_idx_i, orig_H, orig_W, new_H, new_W, num_h, num_w = meta
            recon = torch.zeros((3, new_H, new_W), device=device, dtype=dtype)
            weight = torch.zeros_like(recon)

            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))
                    patch = stylized_patches[proc_ptr]
                    proc_ptr += 1

                    mask = self._blend_mask(i, j, num_h, num_w, patch_size, self.overlap, device, dtype)
                    recon[:, top : top + patch_size, left : left + patch_size] += patch * mask
                    weight[:, top : top + patch_size, left : left + patch_size] += mask

            recon = F.resize(recon / torch.clamp(weight, min=1e-5), size=[orig_H, orig_W], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
            if was_grayscale: recon = F.rgb_to_grayscale(recon, num_output_channels=1)
            x[img_idx_i] = recon

        if self.use_normalization:
            x = (x - self.mean_buf) / self.std_buf

        return x.squeeze(0) if single_image else x
