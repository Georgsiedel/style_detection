import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Union

CURRENT_DIR = Path(__file__).resolve().parent    # style_detection/
SUBMODULE_ROOT = CURRENT_DIR / "MicroAST"       # style_detection/MicroAST/

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

from MicroAST.net_microAST import Encoder, Decoder
DEFAULT_MODELS_DIR = SUBMODULE_ROOT / "models"

class MicroASTAugmentation(nn.Module):
    def __init__(
        self,
        style_feats_path: Union[str, Path],
        content_encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        device: torch.device,
        probability: float = 0.1,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        mean: tuple = None,
        std: tuple = None,
        min_spatial_size: int = 224  # Added minimum resolution parameter
    ):
        super().__init__()
        self.device = device
        self.probability = probability
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_spatial_size = min_spatial_size

        # Instantiate core synthesis components
        self.content_encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)

        self.content_encoder.load_state_dict(torch.load(content_encoder_path, map_location=self.device))
        
        dec_ckpt = torch.load(decoder_path, map_location=self.device)
        dec_state = dec_ckpt.get("state_dict", dec_ckpt)
        cleaned_dec_state = {k.replace("decoder.", ""): v for k, v in dec_state.items()}
        try:
            self.decoder.load_state_dict(cleaned_dec_state)
        except Exception:
            self.decoder.load_state_dict(dec_state)

        self.content_encoder.eval()
        self.decoder.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # OPTIMIZATION: Only register normalization buffers if parameters are explicitly provided
        if mean is not None and std is not None:
            self.register_buffer("img_mean", torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.register_buffer("img_std", torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.use_normalization = True
        else:
            self.img_mean = None
            self.img_std = None
            self.use_normalization = False

        # Build multivariate Gaussian distribution engine
        archive = np.load(style_feats_path)
        mean_tensor = torch.from_numpy(archive["mean"]).to(dtype=torch.float32, device=self.device)
        cov_tensor = torch.from_numpy(archive["covariance"]).to(dtype=torch.float32, device=self.device)
        self.register_buffer("style_mean", mean_tensor, persistent=False)
        self.register_buffer("style_covariance", cov_tensor, persistent=False)
        
        self.distribution = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0:
            return x

        if self.style_mean.device != x.device or self.distribution is None:
            self.to(x.device)
            self.device = x.device  
            self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=self.style_mean, covariance_matrix=self.style_covariance
            )

        batch_size = x.size(0)
        num_to_stylize = int(batch_size * self.probability + random.random())
        if num_to_stylize > batch_size: num_to_stylize = batch_size
        if num_to_stylize <= 0: return x

        out = x.clone()
        indices = torch.randperm(batch_size, device=self.device)[:num_to_stylize]
        content_selected = x[indices].to(torch.float32) # Force FP32 for the generator

        # 1. Conditional Un-normalization (Keeps color profile intact before interpolation)
        if self.use_normalization:
            content_selected = content_selected * self.img_std + self.img_mean

        # 2. Check and Apply Resolution Guardrails
        orig_h, orig_w = content_selected.shape[2], content_selected.shape[3]
        min_side = min(orig_h, orig_w)
        
        requires_upscale = min_side < self.min_spatial_size
        if requires_upscale:
            scale_factor = self.min_spatial_size / min_side
            new_h = int(round(orig_h * scale_factor))
            new_w = int(round(orig_w * scale_factor))
            content_selected = torch.nn.functional.interpolate(
                content_selected, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        # 3. Draw style distribution vector parameters
        samples = self.distribution.sample((num_to_stylize,))
        s0_mean, s0_std = samples[:, 0:64].view(num_to_stylize, 64, 1, 1), samples[:, 64:128].view(num_to_stylize, 64, 1, 1)
        s1_mean, s1_std = samples[:, 128:192].view(num_to_stylize, 64, 1, 1), samples[:, 192:256].view(num_to_stylize, 64, 1, 1)
        w0, w1 = samples[:, 256:320].view(num_to_stylize, 64, 1, 1), samples[:, 320:384].view(num_to_stylize, 64, 1, 1)
        b0, b1 = samples[:, 384:448].view(num_to_stylize, 64, 1, 1), samples[:, 448:512].view(num_to_stylize, 64, 1, 1)

        # Vectorized 4D Concat Grid Generation to satisfy AdaIN constraints
        s0_col = torch.cat([s0_mean - s0_std, s0_mean + s0_std], dim=2)
        s0_dummy = torch.cat([s0_col, s0_col], dim=3)

        s1_col = torch.cat([s1_mean - s1_std, s1_mean + s1_std], dim=2)
        s1_dummy = torch.cat([s1_col, s1_col], dim=3)

        # 4. Stylize content targets
        content_feats = self.content_encoder(content_selected)
        stylized_subset = self.decoder(
            content_feats, 
            [s0_dummy, s1_dummy], 
            [w0, w1], 
            [b0, b1], 
            alpha=random.uniform(self.alpha_min, self.alpha_max)
        )
        stylized_subset = torch.clamp(stylized_subset, 0.0, 1.0)

        # 5. Reverse Resolution Guardrails if applied
        if requires_upscale:
            stylized_subset = torch.nn.functional.interpolate(
                stylized_subset, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        # 6. Conditional Re-normalization
        if self.use_normalization:
            out[indices] = ((stylized_subset - self.img_mean) / self.img_std).to(out.dtype)
        else:
            out[indices] = stylized_subset.to(out.dtype)
            
        return out