import torch
from torch import Tensor

from llm_movies.model import QUANTILES


def pinball_loss(pred: Tensor, target: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    """Sum over targets of the weighted mean (over available rows) of the mean pinball loss over the quantiles.

    Weights are normalised to mean 1 over each target's available rows; a target with no rows adds 0.
    """
    pred, target, weight = pred.float(), target.float(), weight.float()
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)
    diff = torch.where(mask, target, torch.zeros_like(target)).unsqueeze(-1) - pred  # (B, 2, 3)
    per_row = torch.maximum(q * diff, (q - 1) * diff).mean(-1)  # (B, 2)
    m = mask.to(pred.dtype)
    w = torch.where(mask, weight, torch.zeros_like(weight))  # masked cells fully ignored, even NaN/inf
    n = m.sum(0)  # rows available per target
    total = w.sum(0)
    # normalised weight = w * n / total, so the weighted mean is sum(w * loss) / total
    per_target = (w * torch.where(mask, per_row, torch.zeros_like(per_row))).sum(0) / total.clamp_min(torch.finfo(pred.dtype).tiny)
    return torch.where(n > 0, per_target, torch.zeros_like(per_target)).sum()
