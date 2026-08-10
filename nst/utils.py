import torch
from typing import Tuple, Optional, Sequence, Union

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    assert feat.ndim == 4, f"Expected [N,C,H,W], got {tuple(feat.shape)}"
    n, c = feat.shape[:2]
    feat_var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std